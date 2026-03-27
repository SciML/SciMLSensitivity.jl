using Test
using NonlinearSolve, SCCNonlinearSolve
using SciMLSensitivity
using Enzyme
using FiniteDiff
import SciMLStructures as SS

# Two-component SCC problem with parameter coupling through caches.
# Component 1: u1^2 - p[1] = 0 (root: u1 = sqrt(p[1]))
# Component 2: u2 - p[2]*u1 = 0 (root: u2 = p[2]*u1)
#   where u1 from component 1 is passed via explicitfun into component 2's
#   parameter cache.

@testset "SCCNonlinearProblem Enzyme differentiation" begin
    # Sub-problem 1: u^2 - p = 0
    function f1(du, u, p)
        du[1] = u[1]^2 - p[1]
    end
    explicitfun1!(p, sols) = nothing

    # Sub-problem 2: u - cache[1] * p[2] = 0
    # cache[1] will be set to sol1[1] (= sqrt(p[1])) by explicitfun2
    function f2(du, u, p)
        du[1] = u[1] - p[1] * p[2]  # p[1] is cache, p[2] is tunable
    end
    function explicitfun2!(p, sols)
        p[1] = sols[1].u[1]  # transfer u1 from component 1 into cache
        return nothing
    end

    p_shared = [0.0, 2.0]  # p[1] = cache (written by explicitfun2), p[2] = tunable
    prob1 = NonlinearProblem(
        NonlinearFunction{true, SciMLBase.NoSpecialize}(f1), [1.0], p_shared,
    )
    prob2 = NonlinearProblem(
        NonlinearFunction{true, SciMLBase.NoSpecialize}(f2), [1.0], p_shared,
    )

    sccprob = SciMLBase.SCCNonlinearProblem(
        [prob1, prob2],
        SciMLBase.Void{Any}.([explicitfun1!, explicitfun2!]),
    )

    alg = SCCNonlinearSolve.SCCAlg(nlalg = NewtonRaphson())

    # Forward solve works
    p_test = [4.0, 3.0]
    p_shared .= p_test
    sol = solve(sccprob, alg)
    @test SciMLBase.successful_retcode(sol)
    @test sol.u[1] ≈ 2.0 atol = 1.0e-10  # sqrt(4)
    @test sol.u[2] ≈ 6.0 atol = 1.0e-10  # 3 * 2

    # FiniteDiff ground truth
    function loss(p_val)
        p_shared .= p_val
        sol = solve(sccprob, alg)
        sum(sol.u)
    end
    fd = FiniteDiff.finite_difference_gradient(loss, p_test)
    @test any(!iszero, fd)

    # Enzyme gradient
    @testset "Enzyme through SCC" begin
        loss_enzyme = let sccprob = sccprob, alg = alg, p_shared = p_shared
            p_val -> begin
                p_shared .= p_val
                sol = solve(sccprob, alg)
                sum(sol.u)
            end
        end

        dloss = Enzyme.make_zero(loss_enzyme)
        dp = zeros(length(p_test))
        Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse),
            Enzyme.Duplicated(loss_enzyme, dloss),
            Enzyme.Active,
            Enzyme.Duplicated(copy(p_test), dp),
        )
        @test isapprox(dp, fd, rtol = 0.05)
    end
end
