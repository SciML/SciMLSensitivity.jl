using Test, LinearAlgebra
using SciMLSensitivity, Optimization, OptimizationOptimisers, SciMLBase
using Mooncake, ForwardDiff
using SciMLSensitivity: MooncakeVJP

# Helper: build a NonlinearSolution from an optimization solve using the gradient as the residual,
# and the corresponding SteadyStateAdjoint, matching what _concrete_solve_adjoint does internally.
function build_opt_adjoint_sol(prob, alg, sensealg; kwargs...)
    opt_sol = solve(prob, alg; kwargs...)
    opt_f = prob.f
    grad_fn = if sensealg.objective_ad isa Bool && !sensealg.objective_ad
        (u, p) -> FiniteDiff.finite_difference_gradient(Base.Fix2(opt_f, p), u)
    else
        (u, p) -> ForwardDiff.gradient(Base.Fix2(opt_f, p), u)
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
