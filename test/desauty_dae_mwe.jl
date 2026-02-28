using Test
using ModelingToolkit, OrdinaryDiffEq
using ModelingToolkit: t_nounits as t, D_nounits as D
import SciMLStructures as SS
import SciMLSensitivity
using SymbolicIndexingInterface
using FiniteDiff
using ForwardDiff
using Tracker
using Enzyme
using Mooncake

# DAE with nonlinear algebraic constraints forming an SCC chain.
# Inspired by the De Sauty bridge DAE but written as a flat system
# (no ModelingToolkitStandardLibrary dependency).
#
# D(x) = a*x + y + z        (ODE)
# 0 = y^3 + y - b*x         (algebraic: y from x, parameter b)
# 0 = z^3 + z - c*y         (algebraic: z from y, parameter c)
#
# The cubic equations can't be eliminated by structural_simplify.
# With use_scc=true: init is SCCNonlinearProblem with 2 sub-problems
# With use_scc=false: init is NonlinearProblem
#
# Exact initial values with x(0)=1, b=2, c=1.5:
#   y^3 + y = 2 → y = 1 (exact)
#   z^3 + z = 1.5 → z ≈ 0.8612

@parameters a b c
@variables x(t) y(t) z(t)

eqs = [
    D(x) ~ a * x + y + z,
    0 ~ y^3 + y - b * x,
    0 ~ z^3 + z - c * y,
]

@mtkbuild sys = ODESystem(eqs, t)

@testset "DAE with SCC Initialization" begin
    @testset "use_scc = $use_scc" for use_scc in (false, true)
        prob = ODEProblem(
            sys,
            [x => 1.0],
            (0.0, 0.1),
            [a => -0.5, b => 2.0, c => 1.5],
            guesses = [y => 1.0, z => 0.5];
            use_scc,
        )

        # Verify initialization problem type
        idata = prob.f.initialization_data
        init_type_str = string(typeof(idata.initializeprob))
        if use_scc
            @test occursin("SCC", init_type_str)
        else
            @test !occursin("SCC", init_type_str)
        end

        # Forward solve
        sol = solve(prob, Rodas5P(); abstol = 1e-12, reltol = 1e-12)
        @test SciMLBase.successful_retcode(sol)
        @test sol[y, 1]≈1.0 atol=1e-8
        @test 0.85 < sol[z, 1] < 0.87

        tunables, repack, _ = SS.canonicalize(SS.Tunable(), parameter_values(prob))

        # FiniteDiff ground truth for full ODE solve
        loss = let prob = prob, repack = repack
            p -> begin
                new_prob = remake(prob; p = repack(p))
                sol = solve(new_prob, Rodas5P(); abstol = 1e-12, reltol = 1e-12)
                sum(sol)
            end
        end
        fd_grad = FiniteDiff.finite_difference_gradient(loss, tunables)
        @test any(!iszero, fd_grad)

        # Direct init problem differentiation
        iprob = idata.initializeprob
        itunables, irepack, _ = SS.canonicalize(
            SS.Tunable(), parameter_values(iprob),
        )

        init_loss = let iprob = iprob, irepack = irepack
            p -> begin
                iprob2 = remake(iprob, p = irepack(p))
                sol = solve(iprob2)
                sum(sol.u)
            end
        end

        fd_init_grad = FiniteDiff.finite_difference_gradient(init_loss, itunables)
        @test any(!iszero, fd_init_grad)

        @testset "ForwardDiff through init" begin
            if use_scc
                @test_broken begin
                    fwd_init = ForwardDiff.gradient(init_loss, itunables)
                    isapprox(fwd_init, fd_init_grad, rtol = 0.05)
                end
            else
                fwd_init = ForwardDiff.gradient(init_loss, itunables)
                @test isapprox(fwd_init, fd_init_grad, rtol = 0.05)
            end
        end

        @testset "ForwardDiff through ODE solve" begin
            @test_broken begin
                fwd_grad = ForwardDiff.gradient(loss, tunables)
                isapprox(fwd_grad, fd_grad, rtol = 0.05)
            end
        end

        @testset "Enzyme through init" begin
            @test_broken begin
                igs = Enzyme.gradient(Enzyme.Reverse, init_loss, itunables)
                !iszero(sum(igs))
            end
        end

        @testset "Mooncake through init" begin
            @test_broken begin
                rule = Mooncake.build_rrule(init_loss, itunables)
                _, (_, igs) = Mooncake.value_and_gradient!!(
                    rule, init_loss, itunables,
                )
                !iszero(sum(igs))
            end
        end

        @testset "Tracker + GaussAdjoint through ODE solve" begin
            sensealg = SciMLSensitivity.GaussAdjoint(
                autojacvec = SciMLSensitivity.EnzymeVJP(),
            )
            @test_broken begin
                gs = Tracker.gradient(tunables) do tunables
                    new_prob = remake(prob; p = repack(tunables))
                    sol = solve(new_prob, Rodas5P(); sensealg)
                    sum(sol)
                end
                any(!iszero, gs[1])
            end
        end
    end
end
