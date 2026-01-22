using Test, LinearAlgebra
using SciMLSensitivity, Optimization, OptimizationOptimJL
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
        sol = solve(prob, NelderMead())
        u_star = sol.u[1]

        # Analytical gradient
        p_val = p[1]
        g_prime_analytical = -(u_star^3 + 3 * p_val * u_star^2) /
                             (3 * p_val * u_star^2 + 3 * p_val^2 * u_star - 6)

        # Compute gradient using adjoint
        g(p) = solve(prob, NelderMead(), p = p).u[1]

        res_adj = Zygote.gradient(g, p)[1]

        # Test that adjoint matches analytical solution
        @test res_adj[1]≈g_prime_analytical rtol=1e-3

        # Test with explicit sensealg (default)
        res_adj_explicit = Zygote.gradient(
            p -> solve(prob, NelderMead(), p = p,
            sensealg = UnconstrainedOptimizationAdjoint()).u[1],
            p)[1]

        @test res_adj_explicit[1] ≈ g_prime_analytical

        # Test with different VJP methods

        res_reversediff = Zygote.gradient(
            p -> solve(prob, NelderMead(), p = p,
            sensealg = UnconstrainedOptimizationAdjoint(autojacvec = ReverseDiffVJP())).u[1],
            p)[1]
        @test res_reversediff[1] ≈ g_prime_analytical

        res_enzyme = Zygote.gradient(
            p -> solve(prob, NelderMead(), p = p,
            sensealg = UnconstrainedOptimizationAdjoint(autojacvec = EnzymeVJP())).u[1],
            p)[1]
        @test res_enzyme[1] ≈ g_prime_analytical

        res_mooncake = Zygote.gradient(
            p -> solve(prob, NelderMead(), p = p,
            sensealg = UnconstrainedOptimizationAdjoint(autojacvec = MooncakeVJP())).u[1],
            p)[1]
        @test res_mooncake[1] ≈ g_prime_analytical

    end
end
