using Test, LinearAlgebra
using SciMLSensitivity, Optimization, OptimizationOptimisers, OptimizationNLopt, SciMLBase
using Mooncake, ForwardDiff, Zygote
using SciMLSensitivity: MooncakeVJP

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
            f    = (u, p) -> (u[1] - 1)^2 + (u[2] - 1)^2
            # Constraint: u1 + u2 - p[1] = 0  (p flows through cons for correct adjoint)
            cons = (res, u, p) -> (res[1] = u[1] + u[2] - p[1])

            u0 = [1.5, 1.5]  # feasible starting point: u1 + u2 = p[1] = 3
            p  = [3.0]

            opt_f = OptimizationFunction(f, Optimization.AutoForwardDiff(); cons = cons)
            prob  = OptimizationProblem(opt_f, u0, p; lcons = [0.0], ucons = [0.0])

            # Verify the forward solve
            opt_sol = solve(prob, NLopt.LD_SLSQP())
            @test opt_sol.u[1] ≈ p[1] / 2 rtol = 1e-4
            @test opt_sol.u[2] ≈ p[1] / 2 rtol = 1e-4
            @test opt_sol.u[1] + opt_sol.u[2] ≈ p[1] rtol = 1e-6  # constraint satisfied

            # d(u1* + u2*)/dp[1] = d(p[1])/dp[1] = 1
            dp = Zygote.gradient(p) do p
                _prob = remake(prob; p = p)
                sol = solve(_prob, NLopt.LD_SLSQP(); sensealg = OptimizationAdjoint())
                sol.u[1] + sol.u[2]
            end[1]
            @test dp[1] ≈ 1.0 rtol = 1e-4

            # du1*/dp[1] = 0.5
            dp1 = Zygote.gradient(p) do p
                _prob = remake(prob; p = p)
                sol = solve(_prob, NLopt.LD_SLSQP(); sensealg = OptimizationAdjoint())
                sol.u[1]
            end[1]
            @test dp1[1] ≈ 0.5 rtol = 1e-4
        end
    end

    @testset "Active inequality constraint" begin
        let
            # Minimize (u - p[1])^2  s.t.  u <= p[2]  where p[2] < p[1] (constraint active)
            # Optimal solution: u* = p[2]
            # du*/dp[1] = 0,  du*/dp[2] = 1
            f    = (u, p) -> (u[1] - p[1])^2
            # Constraint: u[1] - p[2] <= 0  (p[2] flows through cons for correct adjoint)
            cons = (res, u, p) -> (res[1] = u[1] - p[2])

            u0 = [0.0]
            p  = [3.0, 1.0]  # unconstrained min at u=3, constraint forces u<=1

            opt_f = OptimizationFunction(f, Optimization.AutoForwardDiff(); cons = cons)
            prob  = OptimizationProblem(opt_f, u0, p; lcons = [-Inf], ucons = [0.0])

            opt_sol = solve(prob, NLopt.LD_SLSQP())
            @test opt_sol.u[1] ≈ p[2] rtol = 1e-4
            @test opt_sol.u[1] <= p[2] + 1e-6  # constraint satisfied: u <= p[2]

            dp = Zygote.gradient(p) do p
                _prob = remake(prob; p = p)
                sol = solve(_prob, NLopt.LD_SLSQP(); sensealg = OptimizationAdjoint())
                sol.u[1]
            end[1]
            @test dp[1] ≈ 0.0 atol = 1e-4  # du*/dp[1] = 0 (u* doesn't depend on p[1])
            @test dp[2] ≈ 1.0 rtol = 1e-4  # du*/dp[2] = 1 (u* = p[2])
        end
    end

    @testset "FiniteDiff vs ForwardDiff consistency" begin
        let
            # Equality-constrained problem, compare autodiff=true vs autodiff=false
            f    = (u, p) -> (u[1] - p[1])^2 + (u[2] - p[2])^2
            cons = (res, u, p) -> (res[1] = u[1] + u[2] - p[3])

            u0 = [0.5, 0.5]
            p  = [1.0, 2.0, 3.0]

            opt_f = OptimizationFunction(f, Optimization.AutoForwardDiff(); cons = cons)
            prob  = OptimizationProblem(opt_f, u0, p; lcons = [0.0], ucons = [0.0])

            opt_sol = solve(prob, NLopt.LD_SLSQP())
            @test opt_sol.u[1] + opt_sol.u[2] ≈ p[3] rtol = 1e-6  # constraint satisfied

            dp_fd = Zygote.gradient(p) do p
                _prob = remake(prob; p = p)
                sol = solve(_prob, NLopt.LD_SLSQP(); sensealg = OptimizationAdjoint(autodiff = false))
                sol.u[1]
            end[1]
            dp_fwd = Zygote.gradient(p) do p
                _prob = remake(prob; p = p)
                sol = solve(_prob, NLopt.LD_SLSQP(); sensealg = OptimizationAdjoint(autodiff = true))
                sol.u[1]
            end[1]
            @test dp_fd ≈ dp_fwd rtol = 1e-3
        end
    end

    @testset "Lemma 4.2 (Gould et al.): L2 projection onto hyperplane" begin
        let
            # Minimize (1/2)||u - p||^2  s.t.  u1 + u2 + u3 = 1
            # Analytical solution (Lemma 4.2 with A = [1 1 1], H = I):
            #   g'(p) = I - A^T(AA^T)^{-1}A = I - (1/3)J
            #   dg_i/dp_j = δ_ij - 1/3
            f    = (u, p) -> sum((u .- p) .^ 2) / 2
            cons = (res, u, p) -> (res[1] = u[1] + u[2] + u[3] - 1)

            p  = [2.0, 0.0, 0.0]
            u0 = [1.0 / 3, 1.0 / 3, 1.0 / 3]   # feasible starting point

            opt_f = OptimizationFunction(f, Optimization.AutoForwardDiff(); cons = cons)
            prob  = OptimizationProblem(opt_f, u0, p; lcons = [0.0], ucons = [0.0])

            # Verify forward solve: u* = p - (sum(p)-1)/3 * [1,1,1] = [5/3, -1/3, -1/3]
            opt_sol = solve(prob, NLopt.LD_SLSQP())
            @test opt_sol.u[1] ≈ 5.0 / 3 rtol = 1e-4
            @test opt_sol.u[2] ≈ -1.0 / 3 rtol = 1e-4
            @test opt_sol.u[3] ≈ -1.0 / 3 rtol = 1e-4
            @test sum(opt_sol.u) ≈ 1.0 rtol = 1e-6  # constraint satisfied

            # Verify adjoint: dg_i/dp_j = δ_ij - 1/3
            for i in 1:3
                dp = Zygote.gradient(p) do p
                    _prob = remake(prob; p = p)
                    sol = solve(_prob, NLopt.LD_SLSQP(); sensealg = OptimizationAdjoint())
                    sol.u[i]
                end[1]

                expected = [-1.0 / 3, -1.0 / 3, -1.0 / 3]
                expected[i] += 1.0   # δ_ij term
                @test dp ≈ expected rtol = 1e-3
            end
        end
    end
end
