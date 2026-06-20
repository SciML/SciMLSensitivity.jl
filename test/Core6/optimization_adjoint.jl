using Test, LinearAlgebra
using SciMLSensitivity, Optimization, OptimizationOptimisers, OptimizationNLopt, SciMLBase
using Mooncake, ForwardDiff, FiniteDiff
using SciMLSensitivity: MooncakeVJP, alg_autodiff, diff_type

# Helper: build a NonlinearSolution from an optimization solve using the gradient as the residual,
# and the corresponding SteadyStateAdjoint, matching what _concrete_solve_adjoint does internally.
function build_opt_adjoint_sol(prob, alg, sensealg; kwargs...)
    opt_sol = solve(prob, alg; kwargs...)
    opt_f = prob.f
    grad_fn = if opt_f.grad !== nothing
        opt_f.grad
    elseif alg_autodiff(sensealg)
        (G, u, p) -> ForwardDiff.gradient!(G, Base.Fix2(opt_f, p), u)
    else
        (G, u, p) -> FiniteDiff.finite_difference_gradient!(
            G, Base.Fix2(opt_f, p), u, diff_type(sensealg))
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

    @testset "Enzyme forward backend" begin
        let
            # SLSQP exposes no Hessian, so the Lxx block is built by the AD fallback — this
            # exercises the Enzyme forward path for BOTH the residual stationarity gradient
            # and Lxx. p enters objective and constraint, so the mixed ∇²_xp L term is
            # nonzero. Analytic optimum u* = (13/9, 10/9), μ* = -4/9; differentiating the KKT
            # system gives du*/dp = [4/9 -16/27; 1/9 -7/27].
            f = (u, p) -> (u[1] - 1)^2 + p[2] * (u[2] - 1)^2
            cons = (res, u, p) -> (res[1] = p[2] * u[1] + u[2] - p[1])
            u0 = [1.0, 1.0]
            p = [4.0, 2.0]

            opt_f = OptimizationFunction(f, Optimization.AutoForwardDiff(); cons = cons)
            prob = OptimizationProblem(opt_f, u0, p; lcons = [0.0], ucons = [0.0])
            opt_sol = solve(prob, NLopt.LD_SLSQP())

            J_exact = [4/9 -16/27; 1/9 -7/27]
            function dprow(i; kw...)
                dgdu!(out, _, _, _, _) = (out .= 0; out[i] = 1.0; out)
                adjoint_sensitivities(
                    opt_sol, nothing; sensealg = OptimizationAdjoint(; kw...), dgdu = dgdu!
                )
            end

            J_enz = vcat(
                dprow(1; autodiff = Optimization.AutoEnzyme())',
                dprow(2; autodiff = Optimization.AutoEnzyme())'
            )
            J_fwd = vcat(
                dprow(1; autodiff = Optimization.AutoForwardDiff())',
                dprow(2; autodiff = Optimization.AutoForwardDiff())'
            )
            @test J_enz ≈ J_exact rtol = 1.0e-5
            @test J_enz ≈ J_fwd rtol = 1.0e-6

            # Reverse-mode backends are rejected (these derivatives are forward-mode).
            @test_throws ArgumentError adjoint_sensitivities(
                opt_sol, nothing;
                sensealg = OptimizationAdjoint(autodiff = SciMLSensitivity.AutoReverseDiff()),
                dgdu = (out, _, _, _, _) -> (out .= 0; out[1] = 1.0; out)
            )
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
end
