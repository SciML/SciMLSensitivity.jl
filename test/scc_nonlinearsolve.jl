using Test
using NonlinearSolve, SCCNonlinearSolve
using SciMLSensitivity
using Enzyme
using Mooncake
using ForwardDiff
using FiniteDiff
import SciMLStructures as SS

# Two-component SCC problem with parameter coupling through explicitfuns!
#
# Component 1: u1^2 - p[1] = 0           → u1 = sqrt(p[1])
# Component 2: u2 - cache * p[2] = 0     → u2 = sqrt(p[1]) * p[2]
#   where cache is set to sol1[1] by explicitfun2!
#
# loss = u1 + u2 = sqrt(p[1]) + sqrt(p[1]) * p[2]
# dloss/dp[1] = (1 + p[2]) / (2*sqrt(p[1]))
# dloss/dp[2] = sqrt(p[1])

function f1(du, u, p)
    du[1] = u[1]^2 - p[1]
end
explicitfun1!(p, sols) = nothing

function f2(du, u, p)
    du[1] = u[1] - p[1] * p[2]
end
function explicitfun2!(p, sols)
    p[1] = sols[1].u[1]
    return nothing
end

function make_scc(p_val)
    p1 = copy(p_val)
    p2 = copy(p_val)
    prob1 = NonlinearProblem(
        NonlinearFunction{true, SciMLBase.NoSpecialize}(f1), [1.0], p1,
    )
    prob2 = NonlinearProblem(
        NonlinearFunction{true, SciMLBase.NoSpecialize}(f2), [1.0], p2,
    )
    return SciMLBase.SCCNonlinearProblem(
        [prob1, prob2],
        SciMLBase.Void{Any}.([explicitfun1!, explicitfun2!]),
    )
end

alg = SCCNonlinearSolve.SCCAlg(nlalg = NewtonRaphson())

function loss(p_val)
    sccprob = make_scc(p_val)
    sol = solve(sccprob, alg)
    sum(sol.u)
end

p_test = [4.0, 3.0]

@testset "SCCNonlinearProblem differentiation" begin
    # Forward solve
    sol = solve(make_scc(p_test), alg)
    @test SciMLBase.successful_retcode(sol)
    @test sol.u[1] ≈ 2.0 atol = 1.0e-10
    @test sol.u[2] ≈ 6.0 atol = 1.0e-10

    # FiniteDiff ground truth
    fd = FiniteDiff.finite_difference_gradient(loss, p_test)
    @test fd[1] ≈ (1 + 3) / (2 * sqrt(4)) atol = 1.0e-6
    @test fd[2] ≈ sqrt(4) atol = 1.0e-6

    @testset "ForwardDiff" begin
        # ForwardDiff through SCCNonlinearProblem fails because the
        # explicitfuns! mutate Float64 buffers which can't hold Dual numbers.
        @test_broken begin
            fwd = ForwardDiff.gradient(loss, p_test)
            isapprox(fwd, fd, rtol = 0.05)
        end
    end

    @testset "Enzyme" begin
        # Enzyme through the full SCC loop hits EnzymeNoTypeError from
        # the complex dispatch chain in _scc_solve/iteratively_build_sols.
        # Individual sub-problem solves work with Enzyme (see use_scc=false
        # in desauty_dae_mwe.jl). Tracked at Enzyme.jl#3021.
        @test_broken begin
            g = Enzyme.gradient(
                Enzyme.set_runtime_activity(Enzyme.Reverse), loss, copy(p_test),
            )
            isapprox(g[1], fd, rtol = 0.05)
        end
    end

    @testset "Mooncake" begin
        rule = Mooncake.build_rrule(loss, copy(p_test))
        _, (_, dp_mc) = Mooncake.value_and_gradient!!(
            rule, loss, copy(p_test),
        )
        @test isapprox(collect(dp_mc), fd, rtol = 0.05)
    end
end
