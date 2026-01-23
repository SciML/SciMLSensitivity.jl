using Test, LinearAlgebra
using SciMLSensitivity, Optimization, OptimizationOptimisers
using SciMLSensitivity: MooncakeVJP
using Zygote

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

        # Solve to get optimal solution
        sol = solve(prob, Descent())
        u_star = sol.u[1]

        # Analytical gradient
        p_val = p[1]
        g_prime_analytical = -(u_star^3 + 3 * p_val * u_star^2) /
                             (3 * p_val * u_star^2 + 3 * p_val^2 * u_star - 6)

        # Compute gradient using adjoint
        g(p) = solve(prob, Descent(), p = p).u[1]

        res_adj = Zygote.gradient(g, p)[1]

        # Test that adjoint matches analytical solution
        @test res_adj[1]≈g_prime_analytical rtol=1e-3

        # Test with explicit sensealg (default)
        res_adj_explicit = Zygote.gradient(
            p -> solve(prob, Descent(), p = p,
            sensealg = UnconstrainedOptimizationAdjoint()).u[1],
            p)[1]

        @test res_adj_explicit[1] ≈ g_prime_analytical

        # Test with different VJP methods

        res_reversediff = Zygote.gradient(
            p -> solve(prob, Descent(), p = p,
            sensealg = UnconstrainedOptimizationAdjoint(autojacvec = ReverseDiffVJP())).u[1],
            p)[1]
        @test res_reversediff[1] ≈ g_prime_analytical

        res_enzyme = Zygote.gradient(
            p -> solve(prob, Descent(), p = p,
            sensealg = UnconstrainedOptimizationAdjoint(autojacvec = EnzymeVJP())).u[1],
            p)[1]
        @test res_enzyme[1] ≈ g_prime_analytical

        res_mooncake = Zygote.gradient(
            p -> solve(prob, Descent(), p = p,
            sensealg = UnconstrainedOptimizationAdjoint(autojacvec = MooncakeVJP())).u[1],
            p)[1]
        @test res_mooncake[1] ≈ g_prime_analytical

    end

    @testset "Simple quadratic problem" begin
        # Minimize (u - p)^2
        # Optimal solution: u* = p
        # d(u*)/dp = 1
        f(u, p) = (u[1] - p[1])^2

        u0 = [0.0]
        p = [2.0]

        prob = OptimizationProblem(f, u0, p)
        sol = solve(prob, Descent())

        @test sol.u[1] ≈ p[1] rtol=1e-2

        # Test gradient
        g(p) = solve(prob, Descent(), p = p).u[1]

        res_adj = Zygote.gradient(g, p)[1]

        # Analytical: du*/dp = 1
        @test res_adj[1] ≈ 1.0 rtol=1e-2

        # Test with explicit sensealg
        res_explicit = Zygote.gradient(
            p -> solve(prob, Descent(), p = p,
                      sensealg = UnconstrainedOptimizationAdjoint()).u[1],
            p)[1]
        @test res_explicit[1] ≈ 1.0 rtol=1e-2
    end

    @testset "Multivariate quadratic" begin
        # Minimize ||u - p||^2
        # Optimal solution: u* = p
        # d(u*_i)/dp_j = δ_ij (identity matrix)
        f(u, p) = sum((u .- p) .^ 2)

        u0 = [0.0, 0.0, 0.0]
        p = [1.0, 2.0, 3.0]

        prob = OptimizationProblem(f, u0, p)
        sol = solve(prob, Descent())

        @test sol.u ≈ p rtol=1e-2

        # Test Jacobian: d(u*)/dp should be identity
        for i in 1:3
            g(p) = solve(prob, Descent(), p = p).u[i]
            res_adj = Zygote.gradient(g, p)[1]

            # Should be close to i-th unit vector
            expected = zeros(3)
            expected[i] = 1.0
            @test res_adj ≈ expected rtol=1e-2
        end
    end

    @testset "Linear objective with quadratic constraint" begin
        # Minimize p[1]*u[1] subject to being near optimum
        # This tests that parameters affect the optimal solution
        f(u, p) = p[1] * u[1] + u[1]^2

        u0 = [1.0]
        p = [2.0]

        prob = OptimizationProblem(f, u0, p)
        sol = solve(prob, Descent())

        # Optimal solution: u* = -p[1]/2
        @test sol.u[1] ≈ -p[1]/2 rtol=1e-2

        # Test gradient: d(u*)/dp = -1/2
        g(p) = solve(prob, Descent(), p = p).u[1]
        res_adj = Zygote.gradient(g, p)[1]

        @test res_adj[1] ≈ -0.5 rtol=1e-2

        # Test with different VJP methods
        res_enzyme = Zygote.gradient(
            p -> solve(prob, Descent(), p = p,
                      sensealg = UnconstrainedOptimizationAdjoint(autojacvec = EnzymeVJP())).u[1],
            p)[1]
        @test res_enzyme[1] ≈ -0.5 rtol=1e-2

        res_mooncake = Zygote.gradient(
            p -> solve(prob, Descent(), p = p,
                      sensealg = UnconstrainedOptimizationAdjoint(autojacvec = MooncakeVJP())).u[1],
            p)[1]
        @test res_mooncake[1] ≈ -0.5 rtol=1e-2
    end

    @testset "save_idxs tests" begin
        # Test that save_idxs works correctly
        f(u, p) = sum((u .- p) .^ 2)

        u0 = [1.0, 2.0, 3.0]
        p = [2.0, 3.0, 4.0]

        prob = OptimizationProblem(f, u0, p)

        # Test with single index
        res1 = Zygote.gradient(
            p -> solve(prob, Descent(), p = p, save_idxs = 1)[1],
            p)[1]
        @test length(res1) == 3
        @test res1[1] ≈ 1.0 rtol=1e-2

        # Test with range
        res2 = Zygote.gradient(
            p -> sum(solve(prob, Descent(), p = p, save_idxs = 1:2)),
            p)[1]
        @test length(res2) == 3
        @test res2[1] ≈ 1.0 rtol=1e-2
        @test res2[2] ≈ 1.0 rtol=1e-2
    end
end
