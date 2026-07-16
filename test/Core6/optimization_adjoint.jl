using Test, LinearAlgebra
using SciMLSensitivity, Optimization, OptimizationOptimisers, OptimizationNLopt, SciMLBase
using Mooncake, ForwardDiff, FiniteDiff
using SciMLSensitivity: MooncakeVJP, alg_autodiff, diff_type

# A minimal custom SciMLStructure parameter type (a tunable vector + a non-tunable array),
# mimicking MTKParameters without pulling in ModelingToolkit. Used by the structured-parameter
# testset below. Structs must be defined at top level, so it lives here rather than in the
# testset. (Interface pattern taken from test/Core1/scimlstructures_interface.jl.)
const SSTRUCT = SciMLSensitivity.SciMLStructures
mutable struct StructParams{T, C}
    tunable::Vector{T}
    cdata::C   # non-tunable; `replace` copies it by reference (the #1470 aliasing target)
end
SSTRUCT.isscimlstructure(::StructParams) = true
SSTRUCT.ismutablescimlstructure(::StructParams) = true
SSTRUCT.hasportion(::SSTRUCT.Tunable, ::StructParams) = true
function SSTRUCT.canonicalize(::SSTRUCT.Tunable, p::StructParams)
    return copy(p.tunable), (nb -> SSTRUCT.replace(SSTRUCT.Tunable(), p, nb)), false
end
SSTRUCT.replace(::SSTRUCT.Tunable, p::StructParams, nb) = StructParams(collect(nb), p.cdata)
SSTRUCT.replace!(::SSTRUCT.Tunable, p::StructParams, nb) = (copyto!(p.tunable, nb); p)

# Helper: build a NonlinearSolution from an optimization solve using the gradient as the residual,
# and the corresponding SteadyStateAdjoint, matching what _concrete_solve_adjoint does internally.
function build_opt_adjoint_sol(prob, alg, sensealg; kwargs...)
    opt_sol = solve(prob, alg; kwargs...)
    opt_f = prob.f
    grad_fn = if opt_f.grad !== nothing
        opt_f.grad
    elseif sensealg.objective_ad isa Bool && !sensealg.objective_ad
        (G, u, p) -> FiniteDiff.finite_difference_gradient!(G, Base.Fix2(opt_f, p), u)
    else
        (G, u, p) -> ForwardDiff.gradient!(G, Base.Fix2(opt_f, p), u)
    end
    nlprob = NonlinearProblem(grad_fn, opt_sol.u, prob.p)
    sol = SciMLBase.build_solution(
        nlprob, nothing, opt_sol.u, opt_sol.objective;
        retcode = opt_sol.retcode
    )
    steady_sensealg = SteadyStateAdjoint(
        autojacvec = sensealg.autojacvec,
        linsolve = sensealg.linsolve,
        linsolve_kwargs = sensealg.linsolve_kwargs
    )
    return sol, steady_sensealg
end

@testset "Adjoint sensitivities of optimization solver" begin
    @testset "Analytical solution test (Gould et al.)" begin
        # Example from "On Differentiating Parameterized Argmin and Argmax Problems
        # with Application to Bi-level Optimization" Gould, et. al
        # f(u, p) = p[1]*u[1]^4 + 2*p[1]^2*u[1]^3 - 12*u[1]^2
        #
        # Analytical derivative of optimal solution with respect to p[1]:
        # g'(p) = -(u*^3 + 3*p*u*^2) / (3*p*u*^2 + 3*p^2*u* - 6)
        # where u* is the optimal solution

        function f(u, p)
            return p[1] * u[1]^4 + 2 * p[1]^2 * u[1]^3 - 12 * u[1]^2
        end

        u0 = [-2.0]
        p = [1.0]

        opt_f = OptimizationFunction(f, Optimization.AutoForwardDiff())
        prob = OptimizationProblem(opt_f, u0, p)

        opt_sol = solve(prob, Descent(0.01); maxiters = 10000)
        u_star = opt_sol.u[1]

        p_val = p[1]
        g_prime_analytical = -(u_star^3 + 3 * p_val * u_star^2) /
            (3 * p_val * u_star^2 + 3 * p_val^2 * u_star - 6)

        # dgdu for selecting u[1]: dg/du = e_1
        function dgdu!(out, _, _, _, _)
            out[1] = 1.0
        end

        # Default sensealg
        sol, steady = build_opt_adjoint_sol(
            prob, Descent(0.01), UnconstrainedOptimizationAdjoint();
            maxiters = 10000
        )
        dp = adjoint_sensitivities(sol, nothing; sensealg = steady, dgdu = dgdu!)
        @test dp[1] ≈ g_prime_analytical rtol = 1.0e-3

        # ReverseDiffVJP
        sol, steady = build_opt_adjoint_sol(
            prob, Descent(0.01), UnconstrainedOptimizationAdjoint(autojacvec = ReverseDiffVJP());
            maxiters = 10000
        )
        dp = adjoint_sensitivities(sol, nothing; sensealg = steady, dgdu = dgdu!)
        @test dp[1] ≈ g_prime_analytical rtol = 1.0e-3

        # EnzymeVJP
        sol, steady = build_opt_adjoint_sol(
            prob, Descent(0.01), UnconstrainedOptimizationAdjoint(autojacvec = EnzymeVJP());
            maxiters = 10000
        )
        dp = adjoint_sensitivities(sol, nothing; sensealg = steady, dgdu = dgdu!)
        @test dp[1] ≈ g_prime_analytical rtol = 1.0e-3

        # MooncakeVJP
        sol, steady = build_opt_adjoint_sol(
            prob, Descent(0.01), UnconstrainedOptimizationAdjoint(autojacvec = MooncakeVJP());
            maxiters = 10000
        )
        dp = adjoint_sensitivities(sol, nothing; sensealg = steady, dgdu = dgdu!)
        @test dp[1] ≈ g_prime_analytical rtol = 1.0e-3
    end

    @testset "Simple quadratic problem" begin
        # Minimize (u - p)^2
        # Optimal solution: u* = p
        # d(u*)/dp = 1
        f(u, p) = (u[1] - p[1])^2

        u0 = [0.0]
        p = [2.0]

        opt_f = OptimizationFunction(f, Optimization.AutoForwardDiff())
        prob = OptimizationProblem(opt_f, u0, p)
        opt_sol = solve(prob, Descent(0.01); maxiters = 10000)
        @test opt_sol.u[1] ≈ p[1]

        function dgdu!(out, _, _, _, _)
            out[1] = 1.0
        end

        sol, steady = build_opt_adjoint_sol(
            prob, Descent(0.01), UnconstrainedOptimizationAdjoint();
            maxiters = 10000
        )
        dp = adjoint_sensitivities(sol, nothing; sensealg = steady, dgdu = dgdu!)
        @test dp[1] ≈ 1.0
    end

    @testset "Multivariate quadratic" begin
        # Minimize ||u - p||^2
        # Optimal solution: u* = p
        # d(u*_i)/dp_j = δ_ij (identity matrix)
        f(u, p) = sum((u .- p) .^ 2)

        u0 = [0.0, 0.0, 0.0]
        p = [1.0, 2.0, 3.0]

        opt_f = OptimizationFunction(f, Optimization.AutoForwardDiff())
        prob = OptimizationProblem(opt_f, u0, p)
        opt_sol = solve(prob, Descent(0.01); maxiters = 10000)
        @test opt_sol.u ≈ p rtol = 1.0e-2

        sol, steady = build_opt_adjoint_sol(
            prob, Descent(0.01), UnconstrainedOptimizationAdjoint();
            maxiters = 10000
        )

        for i in 1:3
            function dgdu!(out, _, _, _, _)
                fill!(out, 0.0)
                out[i] = 1.0
            end
            dp = adjoint_sensitivities(sol, nothing; sensealg = steady, dgdu = dgdu!)

            expected = zeros(3)
            expected[i] = 1.0
            @test dp ≈ expected rtol = 1.0e-2
        end
    end

    @testset "Linear objective with quadratic constraint" begin
        # Minimize p[1]*u[1] + u[1]^2
        # Optimal solution: u* = -p[1]/2
        # d(u*)/dp = -1/2
        f(u, p) = p[1] * u[1] + u[1]^2

        u0 = [1.0]
        p = [2.0]

        opt_f = OptimizationFunction(f, Optimization.AutoForwardDiff())
        prob = OptimizationProblem(opt_f, u0, p)
        opt_sol = solve(prob, Descent(0.01); maxiters = 10000)
        @test opt_sol.u[1] ≈ -p[1] / 2 rtol = 1.0e-2

        function dgdu!(out, _, _, _, _)
            out[1] = 1.0
        end

        # EnzymeVJP
        sol, steady = build_opt_adjoint_sol(
            prob, Descent(0.01), UnconstrainedOptimizationAdjoint(autojacvec = EnzymeVJP());
            maxiters = 10000
        )
        dp = adjoint_sensitivities(sol, nothing; sensealg = steady, dgdu = dgdu!)
        @test dp[1] ≈ -0.5 rtol = 1.0e-2

        # MooncakeVJP
        sol, steady = build_opt_adjoint_sol(
            prob, Descent(0.01), UnconstrainedOptimizationAdjoint(autojacvec = MooncakeVJP());
            maxiters = 10000
        )
        dp = adjoint_sensitivities(sol, nothing; sensealg = steady, dgdu = dgdu!)
        @test dp[1] ≈ -0.5 rtol = 1.0e-2
    end
end

@testset "OptimizationAdjoint: constrained optimization sensitivities" begin
    @testset "Equality constraint" begin
        let
            # Minimize (u1-1)^2 + (u2-1)^2  s.t.  u1 + u2 = p[1]
            # Optimal solution: u1* = u2* = p[1]/2
            # du1*/dp[1] = 0.5,  du2*/dp[1] = 0.5
            f = (u, p) -> (u[1] - 1)^2 + (u[2] - 1)^2
            cons = (res, u, p) -> (res[1] = u[1] + u[2] - p[1])

            u0 = [1.5, 1.5]  # feasible: u1+u2 = p[1] = 3
            p = [3.0]

            opt_f = OptimizationFunction(f, Optimization.AutoForwardDiff(); cons = cons)
            prob = OptimizationProblem(opt_f, u0, p; lcons = [0.0], ucons = [0.0])

            opt_sol = solve(prob, NLopt.LD_SLSQP())
            @test opt_sol.u[1] ≈ p[1] / 2 rtol = 1.0e-4
            @test opt_sol.u[2] ≈ p[1] / 2 rtol = 1.0e-4
            @test opt_sol.u[1] + opt_sol.u[2] ≈ p[1] rtol = 1.0e-6  # constraint satisfied

            dgdu1!(out, _, _, _, _) = (out[1] = 1.0; out[2] = 0.0)
            dgdu2!(out, _, _, _, _) = (out[1] = 0.0; out[2] = 1.0)
            dp1 = adjoint_sensitivities(opt_sol, nothing; sensealg = OptimizationAdjoint(), dgdu = dgdu1!)
            dp2 = adjoint_sensitivities(opt_sol, nothing; sensealg = OptimizationAdjoint(), dgdu = dgdu2!)
            @test dp1[1] ≈ 0.5 rtol = 1.0e-4   # du1*/dp[1]
            @test dp2[1] ≈ 0.5 rtol = 1.0e-4   # du2*/dp[1]

            # Reverse-mode outer VJPs nest a reverse tape over the forward-mode cons_j, whose
            # nested-AD buffer OptimizationBase currently mis-types. Rejected with a clear error
            # for constrained problems (rather than a deep AD crash); allowed unconstrained.
            @test_throws SciMLSensitivity.OptimizationAdjointUnsupportedVJPError adjoint_sensitivities(
                opt_sol, nothing;
                sensealg = OptimizationAdjoint(autojacvec = ReverseDiffVJP()), dgdu = dgdu1!
            )
            @test_throws SciMLSensitivity.OptimizationAdjointUnsupportedVJPError adjoint_sensitivities(
                opt_sol, nothing;
                sensealg = OptimizationAdjoint(autojacvec = MooncakeVJP()), dgdu = dgdu1!
            )
        end
    end

    @testset "Active inequality constraint" begin
        let
            # Minimize (u - p[1])^2  s.t.  u <= p[2]  where p[2] < p[1] (constraint active)
            # Optimal solution: u* = p[2]
            # du*/dp[1] = 0,  du*/dp[2] = 1
            f = (u, p) -> (u[1] - p[1])^2
            cons = (res, u, p) -> (res[1] = u[1] - p[2])

            u0 = [0.0]
            p = [3.0, 1.0]  # unconstrained min at u=3, constraint forces u<=1

            opt_f = OptimizationFunction(f, Optimization.AutoForwardDiff(); cons = cons)
            prob = OptimizationProblem(opt_f, u0, p; lcons = [-Inf], ucons = [0.0])

            opt_sol = solve(prob, NLopt.LD_SLSQP())
            @test opt_sol.u[1] ≈ p[2] rtol = 1.0e-4
            @test opt_sol.u[1] <= p[2] + 1.0e-6  # constraint satisfied: u <= p[2]

            dgdu!(out, _, _, _, _) = (out[1] = 1.0)
            dp = adjoint_sensitivities(opt_sol, nothing; sensealg = OptimizationAdjoint(), dgdu = dgdu!)
            @test dp[1] ≈ 0.0 atol = 1.0e-4   # du*/dp[1] = 0
            @test dp[2] ≈ 1.0 rtol = 1.0e-4   # du*/dp[2] = 1
        end
    end

    @testset "FiniteDiff vs ForwardDiff consistency" begin
        let
            # Equality-constrained problem, compare autodiff=true vs autodiff=false
            f = (u, p) -> (u[1] - p[1])^2 + (u[2] - p[2])^2
            cons = (res, u, p) -> (res[1] = u[1] + u[2] - p[3])

            u0 = [0.5, 0.5]
            p = [1.0, 2.0, 3.0]

            opt_f = OptimizationFunction(f, Optimization.AutoForwardDiff(); cons = cons)
            prob = OptimizationProblem(opt_f, u0, p; lcons = [0.0], ucons = [0.0])

            opt_sol = solve(prob, NLopt.LD_SLSQP())
            @test opt_sol.u[1] + opt_sol.u[2] ≈ p[3] rtol = 1.0e-6  # constraint satisfied

            dgdu!(out, _, _, _, _) = (out[1] = 1.0; out[2] = 0.0)
            dp_fd = adjoint_sensitivities(
                opt_sol, nothing;
                sensealg = OptimizationAdjoint(autodiff = false), dgdu = dgdu!
            )
            dp_fwd = adjoint_sensitivities(
                opt_sol, nothing;
                sensealg = OptimizationAdjoint(autodiff = true), dgdu = dgdu!
            )
            @test dp_fd ≈ dp_fwd rtol = 1.0e-3
        end
    end

    @testset "Enzyme backend (backend-matched)" begin
        let
            # SLSQP exposes no Hessian, so the Lxx block is built by the AD fallback, which
            # differentiates the stored grad/cons_j. p enters objective and constraint, so the
            # mixed ∇²_xp L term is nonzero. Analytic optimum u* = (13/9, 10/9), μ* = -4/9;
            # differentiating the KKT system gives du*/dp = [4/9 -16/27; 1/9 -7/27].
            f = (u, p) -> (u[1] - 1)^2 + p[2] * (u[2] - 1)^2
            cons = (res, u, p) -> (res[1] = p[2] * u[1] + u[2] - p[1])
            u0 = [1.0, 1.0]
            p = [4.0, 2.0]
            J_exact = [4 / 9 -16 / 27; 1 / 9 -7 / 27]

            dprow(sol, i; kw...) = begin
                dgdu!(out, _, _, _, _) = (out .= 0; out[i] = 1.0; out)
                adjoint_sensitivities(
                    sol, nothing; sensealg = OptimizationAdjoint(; kw...), dgdu = dgdu!
                )
            end

            # ForwardDiff end-to-end (function + adjoint both ForwardDiff).
            opt_f_fwd = OptimizationFunction(f, Optimization.AutoForwardDiff(); cons = cons)
            sol_fwd = solve(
                OptimizationProblem(opt_f_fwd, u0, p; lcons = [0.0], ucons = [0.0]), NLopt.LD_SLSQP()
            )
            J_fwd = vcat(dprow(sol_fwd, 1)', dprow(sol_fwd, 2)')
            @test J_fwd ≈ J_exact rtol = 1.0e-5

            # Enzyme end-to-end: an AutoEnzyme OptimizationFunction (Enzyme-built grad/cons_j)
            # with EnzymeVJP outer ⇒ Enzyme-over-Enzyme for both the residual and the Lxx
            # fallback (which inherits autodiff = AutoEnzyme from the function).
            opt_f_enz = OptimizationFunction(f, Optimization.AutoEnzyme(); cons = cons)
            sol_enz = solve(
                OptimizationProblem(opt_f_enz, u0, p; lcons = [0.0], ucons = [0.0]), NLopt.LD_SLSQP()
            )
            J_enz = vcat(
                dprow(sol_enz, 1; autojacvec = EnzymeVJP())',
                dprow(sol_enz, 2; autojacvec = EnzymeVJP())'
            )
            @test J_enz ≈ J_exact rtol = 1.0e-5
            @test J_enz ≈ J_fwd rtol = 1.0e-5

            # Cross-backend Lxx is rejected, not silently wrong: differentiating a ForwardDiff-built
            # function's grad/cons_j with Enzyme would give a wrong Hessian. Reverse-mode likewise.
            @test_throws ArgumentError dprow(sol_fwd, 1; autodiff = Optimization.AutoEnzyme())
            @test_throws ArgumentError dprow(sol_fwd, 1; autodiff = SciMLSensitivity.AutoReverseDiff())

            # Incompatible outer VJPs raise the structured error rather than crashing deep in AD:
            #   ZygoteVJP can't order nested ForwardDiff tags;
            @test_throws SciMLSensitivity.OptimizationAdjointUnsupportedVJPError dprow(
                sol_fwd, 1; autojacvec = SciMLSensitivity.ZygoteVJP()
            )
            #   EnzymeVJP can't nest over a ForwardDiff-built function;
            @test_throws SciMLSensitivity.OptimizationAdjointUnsupportedVJPError dprow(
                sol_fwd, 1; autojacvec = EnzymeVJP()
            )
            #   an Enzyme-built function needs EnzymeVJP (a non-Enzyme outer can't nest over it).
            @test_throws SciMLSensitivity.OptimizationAdjointUnsupportedVJPError dprow(sol_enz, 1)
        end
    end

    @testset "p only in objective (sensitivity via ∇²_xp L, J_p g = 0)" begin
        let
            # Minimize p[1]*u[1] + u[1]^2 + u[2]^2  s.t.  u[1] + u[2] = 1  (no p in constraint)
            # J_p g = 0; sensitivity flows entirely through ∇²_xp L = [1, 0].
            # KKT → u1* = (2 - p[1])/4,  u2* = (2 + p[1])/4
            # du1*/dp[1] = -1/4,  du2*/dp[1] = 1/4
            f = (u, p) -> p[1] * u[1] + u[1]^2 + u[2]^2
            cons = (res, u, p) -> (res[1] = u[1] + u[2] - 1)

            p = [2.0]
            u0 = [0.0, 1.0]   # feasible: u1+u2 = 1

            opt_f = OptimizationFunction(f, Optimization.AutoForwardDiff(); cons = cons)
            prob = OptimizationProblem(opt_f, u0, p; lcons = [0.0], ucons = [0.0])

            opt_sol = solve(prob, NLopt.LD_SLSQP())
            @test opt_sol.u[1] ≈ (2 - p[1]) / 4 rtol = 1.0e-4
            @test opt_sol.u[2] ≈ (2 + p[1]) / 4 rtol = 1.0e-4
            @test opt_sol.u[1] + opt_sol.u[2] ≈ 1.0 rtol = 1.0e-6  # constraint satisfied

            dgdu1!(out, _, _, _, _) = (out[1] = 1.0; out[2] = 0.0)
            dgdu2!(out, _, _, _, _) = (out[1] = 0.0; out[2] = 1.0)
            dp1 = adjoint_sensitivities(opt_sol, nothing; sensealg = OptimizationAdjoint(), dgdu = dgdu1!)
            dp2 = adjoint_sensitivities(opt_sol, nothing; sensealg = OptimizationAdjoint(), dgdu = dgdu2!)
            @test dp1[1] ≈ -0.25 rtol = 1.0e-3   # du1*/dp[1]
            @test dp2[1] ≈ 0.25 rtol = 1.0e-3   # du2*/dp[1]
        end
    end

    @testset "Inactive inequality constraint" begin
        let
            # Minimize (u - p[1])^2  s.t.  u <= p[2]  where p[2] > p[1] (constraint NOT active)
            # Optimal solution: u* = p[1] (unconstrained min, inequality slack)
            # du*/dp[1] = 1,  du*/dp[2] = 0
            f = (u, p) -> (u[1] - p[1])^2
            cons = (res, u, p) -> (res[1] = u[1] - p[2])

            p = [1.0, 5.0]   # unconstrained min at u=1, well inside bound u<=5
            u0 = [0.0]

            opt_f = OptimizationFunction(f, Optimization.AutoForwardDiff(); cons = cons)
            prob = OptimizationProblem(opt_f, u0, p; lcons = [-Inf], ucons = [0.0])

            opt_sol = solve(prob, NLopt.LD_SLSQP())
            @test opt_sol.u[1] ≈ p[1] rtol = 1.0e-4
            @test opt_sol.u[1] <= p[2] + 1.0e-6  # constraint satisfied (slack)

            dgdu!(out, _, _, _, _) = (out[1] = 1.0)
            dp = adjoint_sensitivities(opt_sol, nothing; sensealg = OptimizationAdjoint(), dgdu = dgdu!)
            @test dp[1] ≈ 1.0 rtol = 1.0e-3   # du*/dp[1] = 1
            @test dp[2] ≈ 0.0 atol = 1.0e-3   # du*/dp[2] = 0 (inactive)
        end
    end

    @testset "Mixed equality + active inequality" begin
        let
            # Minimize (u1-3)^2 + (u2-3)^2  s.t.  u1+u2 = p[1]  and  u1 <= p[2]
            # At p=[4,1]: u1* = p[2] = 1,  u2* = p[1] - p[2] = 3
            # du1*/dp = [0, 1],  du2*/dp = [1, -1]
            f = (u, p) -> (u[1] - 3)^2 + (u[2] - 3)^2
            cons = (res, u, p) -> (res[1] = u[1] + u[2] - p[1]; res[2] = u[1] - p[2])

            p = [4.0, 1.0]
            u0 = [1.0, 3.0]   # feasible: u1+u2=4, u1=1<=1

            opt_f = OptimizationFunction(f, Optimization.AutoForwardDiff(); cons = cons)
            prob = OptimizationProblem(opt_f, u0, p; lcons = [0.0, -Inf], ucons = [0.0, 0.0])

            opt_sol = solve(prob, NLopt.LD_SLSQP())
            @test opt_sol.u[1] ≈ p[2] rtol = 1.0e-4
            @test opt_sol.u[2] ≈ p[1] - p[2] rtol = 1.0e-4
            @test opt_sol.u[1] + opt_sol.u[2] ≈ p[1] rtol = 1.0e-6  # equality satisfied
            @test opt_sol.u[1] <= p[2] + 1.0e-6                      # inequality satisfied

            dgdu1!(out, _, _, _, _) = (out[1] = 1.0; out[2] = 0.0)
            dgdu2!(out, _, _, _, _) = (out[1] = 0.0; out[2] = 1.0)
            dp1 = adjoint_sensitivities(opt_sol, nothing; sensealg = OptimizationAdjoint(), dgdu = dgdu1!)
            dp2 = adjoint_sensitivities(opt_sol, nothing; sensealg = OptimizationAdjoint(), dgdu = dgdu2!)
            @test dp1[1] ≈ 0.0 atol = 1.0e-3   # du1*/dp[1]
            @test dp1[2] ≈ 1.0 rtol = 1.0e-3   # du1*/dp[2]
            @test dp2[1] ≈ 1.0 rtol = 1.0e-3   # du2*/dp[1]
            @test dp2[2] ≈ -1.0 rtol = 1.0e-3   # du2*/dp[2]
        end
    end

    @testset "Multiple equality constraints" begin
        let
            # Minimize (1/2)||u||^2  s.t.  u1+u2 = p[1],  u2+u3 = p[2]
            # Analytical solution: u* = [(2p[1]-p[2])/3, (p[1]+p[2])/3, (-p[1]+2p[2])/3]
            # du1/dp = [2/3, -1/3],  du2/dp = [1/3, 1/3],  du3/dp = [-1/3, 2/3]
            f = (u, p) -> sum(u .^ 2) / 2
            cons = (res, u, p) -> (res[1] = u[1] + u[2] - p[1]; res[2] = u[2] + u[3] - p[2])

            p = [1.0, 1.0]
            u0 = [1.0 / 3, 2.0 / 3, 1.0 / 3]   # feasible

            opt_f = OptimizationFunction(f, Optimization.AutoForwardDiff(); cons = cons)
            prob = OptimizationProblem(opt_f, u0, p; lcons = [0.0, 0.0], ucons = [0.0, 0.0])

            opt_sol = solve(prob, NLopt.LD_SLSQP())
            @test opt_sol.u[1] ≈ (2p[1] - p[2]) / 3 rtol = 1.0e-4
            @test opt_sol.u[2] ≈ (p[1] + p[2]) / 3 rtol = 1.0e-4
            @test opt_sol.u[3] ≈ (-p[1] + 2p[2]) / 3 rtol = 1.0e-4
            @test opt_sol.u[1] + opt_sol.u[2] ≈ p[1] rtol = 1.0e-6
            @test opt_sol.u[2] + opt_sol.u[3] ≈ p[2] rtol = 1.0e-6

            expected = [[2 / 3, -1 / 3], [1 / 3, 1 / 3], [-1 / 3, 2 / 3]]
            for (i, exp_row) in enumerate(expected)
                e = zeros(3); e[i] = 1.0
                dgdui!(out, _, _, _, _) = copyto!(out, e)
                dp = adjoint_sensitivities(
                    opt_sol, nothing;
                    sensealg = OptimizationAdjoint(), dgdu = dgdui!
                )
                @test dp ≈ exp_row rtol = 1.0e-3
            end
        end
    end

    @testset "Active variable bound (lb/ub)" begin
        let
            # Minimize (u1-p)^2 + (u2-p)^2  s.t.  u1 >= 2 (active lb, since p=0 < 2), u2 free
            # u1* = 2 (pinned at bound) → du1*/dp = 0  (without lb in KKT this incorrectly gives 1)
            # u2* = p = 0 (unconstrained) → du2*/dp = 1
            f = (u, p) -> (u[1] - p[1])^2 + (u[2] - p[1])^2

            p = [0.0]
            u0 = [2.0, 0.0]

            opt_f = OptimizationFunction(f, Optimization.AutoForwardDiff())
            prob = OptimizationProblem(opt_f, u0, p; lb = [2.0, -Inf], ub = [Inf, Inf])

            opt_sol = solve(prob, NLopt.LD_SLSQP())
            @test opt_sol.u[1] ≈ 2.0 rtol = 1.0e-4   # pinned at lb
            @test opt_sol.u[2] ≈ p[1] rtol = 1.0e-4  # free, at unconstrained min

            dgdu1!(out, _, _, _, _) = (out[1] = 1.0; out[2] = 0.0)
            dgdu2!(out, _, _, _, _) = (out[1] = 0.0; out[2] = 1.0)
            dp1 = adjoint_sensitivities(opt_sol, nothing; sensealg = OptimizationAdjoint(), dgdu = dgdu1!)
            dp2 = adjoint_sensitivities(opt_sol, nothing; sensealg = OptimizationAdjoint(), dgdu = dgdu2!)
            @test dp1[1] ≈ 0.0 atol = 1.0e-4   # du1*/dp = 0 (pinned at bound)
            @test dp2[1] ≈ 1.0 rtol = 1.0e-4   # du2*/dp = 1 (free variable)
        end
    end

    @testset "p in both objective and constraint (both ∇²_xp L and J_p g nonzero)" begin
        let
            # Minimize (u1 - p[1])^2 + u2^2  s.t.  u1 + u2 = p[2]
            # KKT → u1* = (p[1]+p[2])/2,  u2* = (p[2]-p[1])/2
            # du1*/dp = [1/2, 1/2],  du2*/dp = [-1/2, 1/2]
            f = (u, p) -> (u[1] - p[1])^2 + u[2]^2
            cons = (res, u, p) -> (res[1] = u[1] + u[2] - p[2])

            p = [1.0, 3.0]
            u0 = [1.5, 1.5]   # feasible: u1+u2 = 3 = p[2]

            opt_f = OptimizationFunction(f, Optimization.AutoForwardDiff(); cons = cons)
            prob = OptimizationProblem(opt_f, u0, p; lcons = [0.0], ucons = [0.0])

            opt_sol = solve(prob, NLopt.LD_SLSQP())
            @test opt_sol.u[1] ≈ (p[1] + p[2]) / 2 rtol = 1.0e-4
            @test opt_sol.u[2] ≈ (p[2] - p[1]) / 2 rtol = 1.0e-4
            @test opt_sol.u[1] + opt_sol.u[2] ≈ p[2] rtol = 1.0e-6  # constraint satisfied

            dgdu1!(out, _, _, _, _) = (out[1] = 1.0; out[2] = 0.0)
            dgdu2!(out, _, _, _, _) = (out[1] = 0.0; out[2] = 1.0)
            dp1 = adjoint_sensitivities(opt_sol, nothing; sensealg = OptimizationAdjoint(), dgdu = dgdu1!)
            dp2 = adjoint_sensitivities(opt_sol, nothing; sensealg = OptimizationAdjoint(), dgdu = dgdu2!)
            @test dp1 ≈ [0.5, 0.5] rtol = 1.0e-3
            @test dp2 ≈ [-0.5, 0.5] rtol = 1.0e-3
        end
    end

    @testset "Near-coincident two-sided band folds to equality (no singular KKT)" begin
        let
            # Minimize (u1-1)^2 + (u2-1)^2  s.t.  u1 + u2 = p[1]  ⇒  u* = (p1/2, p1/2).
            # Posing the constraint as a two-sided band [0, 1e-10] narrower than 2*atol makes it
            # numerically an equality: left as an inequality it would register active at *both*
            # bounds, stacking ±J[i,:] into a singular KKT matrix (NaN duals). It must instead be
            # folded into the equality set and give the same finite sensitivity as lcons == ucons.
            f = (u, p) -> (u[1] - 1)^2 + (u[2] - 1)^2
            cons = (res, u, p) -> (res[1] = u[1] + u[2] - p[1])
            p = [1.0]
            opt_f = OptimizationFunction(f, Optimization.AutoForwardDiff(); cons = cons)
            dgdu!(out, _, _, _, _) = (out[1] = 1.0; out[2] = 0.0)

            dp_of(ucons) = begin
                prob = OptimizationProblem(opt_f, [0.3, 0.3], p; lcons = [0.0], ucons = ucons)
                sol = solve(prob, NLopt.LD_SLSQP())
                adjoint_sensitivities(sol, nothing; sensealg = OptimizationAdjoint(), dgdu = dgdu!)
            end

            dp_eq = dp_of([0.0])        # exact equality
            dp_nc = dp_of([1.0e-10])    # near-coincident band (< 2*atol)
            @test all(isfinite, dp_nc)
            @test dp_nc ≈ [0.5] rtol = 1.0e-3
            @test dp_nc ≈ dp_eq rtol = 1.0e-3
        end
    end

    @testset "Inactive inequality without stored cons_j (mu is zero)" begin
        let
            # A constrained problem whose stored function has `grad` but no `cons_j` (built with
            # NoAD + an explicit gradient), and whose lone inequality is strictly inactive at u*.
            # `mu_full` is all-zero, so the constraint-Jacobian term drops out — the residual/Lxx
            # fallback must skip `cons_j` rather than call `_opt_jac_q(nothing, …)` (a MethodError).
            f = (u, p) -> (u[1] - p[1])^2 + (u[2] - p[2])^2
            gradf = (G, u, p) -> (G[1] = 2 * (u[1] - p[1]); G[2] = 2 * (u[2] - p[2]); G)
            consf = (res, u, p) -> (res[1] = u[1] + u[2]; res)
            p = [0.3, 0.4]

            opt_f = OptimizationFunction(f, SciMLBase.NoAD(); grad = gradf, cons = consf)
            prob = OptimizationProblem(opt_f, [0.0, 0.0], p; lcons = [-Inf], ucons = [100.0])
            cache = Optimization.init(prob, NLopt.LD_SLSQP())
            @test cache.f.cons_j === nothing
            @test cache.f.grad !== nothing

            # constraint inactive at the unconstrained optimum u* = p
            sol = SciMLBase.build_solution(
                cache, NLopt.LD_SLSQP(), copy(p), 0.0; retcode = ReturnCode.Success
            )
            dgdu!(out, _, _, _, _) = (out[1] = 1.0; out[2] = 0.0)
            # AutoFiniteDiff avoids nesting an AD backend over the NoAD-built grad in the Lxx fallback
            dp = adjoint_sensitivities(
                sol, nothing;
                sensealg = OptimizationAdjoint(autodiff = Optimization.AutoFiniteDiff()), dgdu = dgdu!
            )
            @test all(isfinite, dp)
            @test dp ≈ [1.0, 0.0] atol = 1.0e-6   # unconstrained ⇒ du*/dp = I
        end
    end

    @testset "Out-of-place constraint (OptimizationFunction{false})" begin
        # A genuinely out-of-place constrained problem. `OptimizationFunction{false}` instantiates
        # `cons` as an out-of-place `(x, p)` that *returns* the constraint vector, whereas the
        # default `iip = true` path instantiates an in-place `(res, x, p)` cons. The KKT residual
        # must dispatch on the function's `isinplace` (`Val{iip}`) to call the right form — the
        # old code hardcoded the 3-arg in-place call and crashed on the out-of-place cons.
        # min (u1-1)^2 + (u2-1)^2 s.t. u1+u2 = p1 ⇒ u* = (p1/2, p1/2); G = u1 ⇒ dG/dp1 = 1/2.
        f = (u, p) -> (u[1] - 1)^2 + (u[2] - 1)^2
        of = OptimizationFunction{false}(
            f, Optimization.AutoForwardDiff(); cons = (u, p) -> [u[1] + u[2] - p[1]]
        )
        cache = Optimization.init(
            OptimizationProblem(of, [0.3, 0.3], [1.0]; lcons = [0.0], ucons = [0.0]), NLopt.LD_SLSQP()
        )
        @test SciMLBase.isinplace(cache.f) == false   # genuinely out-of-place
        # Build the solution at the analytic optimum (a full `solve` of an out-of-place
        # constrained problem is unsupported upstream), then take the adjoint.
        sol = SciMLBase.build_solution(
            cache, NLopt.LD_SLSQP(), [0.5, 0.5], 0.0; retcode = ReturnCode.Success
        )
        dgdu!(out, _, _, _, _) = (out[1] = 1.0; out[2] = 0.0)
        dp = adjoint_sensitivities(sol, nothing; sensealg = OptimizationAdjoint(), dgdu = dgdu!)
        @test dp ≈ [0.5] rtol = 1.0e-6
    end

    @testset "autojacvec = true valid over a FiniteDiff Lxx (decoupled from autodiff)" begin
        # The outer VJP is independent of `autodiff` (which backs only the Lxx fallback), so a
        # ForwardDiff outer (autojacvec = true) is valid over a FiniteDiff Lxx (autodiff = false).
        # SLSQP exposes no Hessian ⇒ Lxx is the AD fallback; the combo must match the exact J.
        f = (u, p) -> (u[1] - 1)^2 + p[2] * (u[2] - 1)^2
        cons = (res, u, p) -> (res[1] = p[2] * u[1] + u[2] - p[1])
        J_exact = [4 / 9 -16 / 27; 1 / 9 -7 / 27]
        opt_f = OptimizationFunction(f, Optimization.AutoForwardDiff(); cons = cons)
        sol = solve(
            OptimizationProblem(opt_f, [1.0, 1.0], [4.0, 2.0]; lcons = [0.0], ucons = [0.0]),
            NLopt.LD_SLSQP()
        )
        dprow(i) = begin
            dg!(out, _, _, _, _) = (out .= 0; out[i] = 1.0; out)
            adjoint_sensitivities(
                sol, nothing;
                sensealg = OptimizationAdjoint(autojacvec = true, autodiff = false), dgdu = dg!
            )
        end
        J = vcat(dprow(1)', dprow(2)')
        @test J ≈ J_exact rtol = 1.0e-5
    end

    @testset "Structured (SciMLStructure) parameters via EnzymeVJP" begin
        # A structured parameter (tunable vector + non-tunable array) mimicking MTKParameters,
        # differentiated with EnzymeVJP. Exercises two structured-param fixes at once:
        #   (1) `_opt_q_eltype` sizes the constraint/gradient buffers from the *tunables'* eltype
        #       — a structured param's `eltype` is `Any`, which would otherwise throw from
        #       `_opt_bufel`'s `oneunit(::Type{Any})`;
        #   (2) the EnzymeVJP shadow is `make_zero(p)`, not `repack(zero(tunables))`, so the
        #       non-tunable `cdata` array is not aliased-and-zeroed across repeated adjoints (#1470).
        # min (u1-1)^2 + (u2-1)^2 s.t. u1+u2 = tunable[1]; u* = (t/2, t/2); G = u1 ⇒ dG/dt = 1/2.
        f = (u, p) -> (u[1] - 1)^2 + (u[2] - 1)^2
        cons = (res, u, p) -> (res[1] = u[1] + u[2] - p.tunable[1])
        p0 = StructParams([3.0], [7.0, 8.0])
        cdata_before = copy(p0.cdata)
        optf = OptimizationFunction(f, Optimization.AutoEnzyme(); cons = cons)
        cache = Optimization.init(
            OptimizationProblem(optf, [1.0, 1.0], p0; lcons = [0.0], ucons = [0.0]), NLopt.LD_SLSQP()
        )
        sol = SciMLBase.build_solution(
            cache, NLopt.LD_SLSQP(), [1.5, 1.5], 0.0; retcode = ReturnCode.Success
        )
        dgdu!(out, _, _, _, _) = (out[1] = 1.0; out[2] = 0.0)
        dp1 = adjoint_sensitivities(
            sol, nothing; sensealg = OptimizationAdjoint(autojacvec = EnzymeVJP()), dgdu = dgdu!
        )
        dp2 = adjoint_sensitivities(
            sol, nothing; sensealg = OptimizationAdjoint(autojacvec = EnzymeVJP()), dgdu = dgdu!
        )
        @test dp1 ≈ [0.5] rtol = 1.0e-6              # dG/d(tunable[1]) = 1/2
        @test dp2 ≈ dp1 rtol = 1.0e-8                # repeatable
        @test p0.cdata == cdata_before              # non-tunable field not corrupted (#1470)
    end

    @testset "alg_autodiff keys on autojacvec, independent of autodiff" begin
        # alg_autodiff selects the outer materialized-Jacobian AD mode (true=ForwardDiff,
        # false=FiniteDiff) for the Bool path. It keys purely on `autojacvec`: the outer VJP
        # differentiates the KKT residual over the stored dual-tolerant grad/cons_j and never
        # touches the `Lxx` fallback that `autodiff` backs, so the two are independent — a
        # ForwardDiff outer is valid over a FiniteDiff `Lxx`.
        @test alg_autodiff(OptimizationAdjoint(autojacvec = false)) == false
        @test alg_autodiff(OptimizationAdjoint(autojacvec = true)) == true
        # `autodiff` does not affect the outer AD mode in either direction
        @test alg_autodiff(OptimizationAdjoint(autojacvec = false, autodiff = false)) == false
        @test alg_autodiff(OptimizationAdjoint(autojacvec = true, autodiff = false)) == true
        @test alg_autodiff(OptimizationAdjoint(autojacvec = true, autodiff = true)) == true
    end
end

@testset "auto-selected default sensealg (constraint-aware)" begin
    let
        # Differentiating a solve without a `sensealg` auto-selects the optimization adjoint:
        # the constrained KKT `OptimizationAdjoint` when the problem has constraints, the
        # stationarity-only `UnconstrainedOptimizationAdjoint` otherwise. Exercised through the
        # solve-differentiation rrule (`_concrete_solve_adjoint` with `sensealg = nothing`)
        # directly — not a full Zygote-through-solve value test, just the dispatch.
        f = (u, p) -> (u[1] - p[1])^2 + (u[2] - p[2])^2
        cons = (res, u, p) -> (res[1] = p[2] * u[1] + u[2] - p[1])
        u0 = [0.5, 0.5]
        p = [1.0, 2.0]
        orig = SciMLBase.ChainRulesOriginator()
        pgrad(g) = only(filter(x -> x isa AbstractArray, g))  # the lone parameter tangent
        dgdu1!(out, _, _, _, _) = (out .= 0; out[1] = 1.0; out)

        # constrained → auto-selects OptimizationAdjoint: matches the explicit KKT result
        # (had it wrongly picked the unconstrained adjoint, the constraint would be ignored).
        optf_c = OptimizationFunction(f, Optimization.AutoForwardDiff(); cons = cons)
        prob_c = OptimizationProblem(optf_c, u0, p; lcons = [0.0], ucons = [0.0])
        out_c, back_c = SciMLBase._concrete_solve_adjoint(
            prob_c, NLopt.LD_SLSQP(), nothing, u0, p, orig; verbose = false
        )
        dp_expl = adjoint_sensitivities(out_c, nothing; sensealg = OptimizationAdjoint(), dgdu = dgdu1!)
        @test pgrad(back_c([1.0, 0.0])) ≈ dp_expl rtol = 1.0e-6

        # unconstrained → auto-selects UnconstrainedOptimizationAdjoint (u* = p ⇒ ∂u1/∂p = [1, 0])
        optf_u = OptimizationFunction(f, Optimization.AutoForwardDiff())
        prob_u = OptimizationProblem(optf_u, u0, p)
        out_u, back_u = SciMLBase._concrete_solve_adjoint(
            prob_u, NLopt.LD_LBFGS(), nothing, u0, p, orig; verbose = false
        )
        @test pgrad(back_u([1.0, 0.0])) ≈ [1.0, 0.0] atol = 1.0e-4
    end
end

@testset "backpass structural tangent (parameter cotangent on the solution)" begin
    # The solve-differentiation pullback returns an `OptimizationSolution`. A ChainRules-native
    # consumer may hand back a *structural* tangent (`Tangent`/`NamedTuple`) rather than the bare
    # `u`-cotangent Zygote produces. `OptimizationSolution` has no `prob` field — its parameters
    # live in the cache at `cache.reinit_cache.p` — so the backpass must read the parameter
    # cotangent from there (not a nonexistent `Δ.prob.p`, which drops it silently and throws on a
    # raw NamedTuple), and must not double the KKT gradient.
    Tangent = SciMLSensitivity.Tangent
    orig = SciMLBase.ChainRulesOriginator()
    pgrad(t) = only(filter(x -> x isa AbstractArray, t))
    f = (u, p) -> (u[1] - p[1])^2 + (u[2] - p[2])^2   # u* = p ⇒ du*/dp = I

    for (alg, sensealg) in (
            (NLopt.LD_LBFGS(), UnconstrainedOptimizationAdjoint()),  # steadystatebackpass
            (NLopt.LD_LBFGS(), OptimizationAdjoint()),               # optimizationbackpass
        )
        optf = OptimizationFunction(f, Optimization.AutoForwardDiff())
        prob = OptimizationProblem(optf, [0.0, 0.0], [1.0, 2.0])
        out, back = SciMLBase._concrete_solve_adjoint(
            prob, alg, sensealg, prob.u0, prob.p, orig; verbose = false
        )
        ST = typeof(out)
        CT = typeof(out.cache)
        RIC = typeof(getfield(out.cache, :reinit_cache))

        # (a) KKT contribution via `Δ.u` matches the bare-array path — no doubling.
        ref = pgrad(back([1.0, 0.0]))
        @test ref ≈ [1.0, 0.0] atol = 1.0e-4
        @test pgrad(back(Tangent{ST}(; u = [1.0, 0.0]))) ≈ ref atol = 1.0e-8

        # (b) an explicit parameter cotangent at the real location `cache.reinit_cache.p` is
        # accumulated (with `Δ.u = 0` the KKT part is zero, so only this survives).
        Δp = [3.0, 5.0]
        tg = Tangent{ST}(;
            u = zeros(2),
            cache = Tangent{CT}(; reinit_cache = Tangent{RIC}(; p = Δp))
        )
        @test pgrad(back(tg)) ≈ Δp atol = 1.0e-8

        # (c) a raw NamedTuple tangent must not throw (the old `Δ.prob.p` did).
        nt = (; u = [1.0, 0.0], cache = (; reinit_cache = (; p = zeros(2))))
        local rnt
        @test (rnt = pgrad(back(nt)); true)
        @test rnt ≈ [1.0, 0.0] atol = 1.0e-4
    end
end
