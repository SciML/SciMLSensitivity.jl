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
    return du[1] = u[1]^2 - p[1]
end
explicitfun1!(p, sols) = nothing

function f2(du, u, p)
    return du[1] = u[1] - p[1] * p[2]
end
function explicitfun2!(p, sols)
    p[1] = sols[1].u[1]
    return nothing
end

function make_scc(p_val)
    p1 = copy(p_val)
    p2 = copy(p_val)
    prob1 = NonlinearProblem(f1, [1.0], p1)
    prob2 = NonlinearProblem(f2, [1.0], p2)
    # Use Tuple (not Vector) for sub-problems and explicitfuns so that
    # each element has a concrete type. Enzyme requires concrete types
    # to specialize through the SCC dispatch chain.
    return SciMLBase.SCCNonlinearProblem(
        (prob1, prob2), (explicitfun1!, explicitfun2!),
    )
end

alg = SCCNonlinearSolve.SCCAlg(nlalg = NewtonRaphson())

function loss(p_val)
    sccprob = make_scc(p_val)
    sol = solve(sccprob, alg)
    return sum(sol.u)
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
        fwd = ForwardDiff.gradient(loss, p_test)
        @test isapprox(fwd, fd, rtol = 0.05)
    end

    # Enzyme test skipped: Enzyme produces correct gradients with Tuple-based
    # SCCNonlinearProblem (verified manually: [1.0, 2.0] matches FiniteDiff),
    # but intermittently segfaults due to GC corruption on Julia 1.10,
    # crashing the test process. Vector-based SCC fails because heterogeneous
    # function types get erased to Any.
    # See Enzyme.jl#3021.
    @testset "Enzyme" begin
        @test_skip true
    end

    @testset "Mooncake" begin
        rule = Mooncake.build_rrule(loss, copy(p_test))
        _, (_, dp_mc) = Mooncake.value_and_gradient!!(
            rule, loss, copy(p_test),
        )
        @test isapprox(collect(dp_mc), fd, rtol = 0.05)
    end
end
