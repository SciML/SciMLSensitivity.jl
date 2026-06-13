using Test, SciMLSensitivity, Enzyme, ComponentArrays, OrdinaryDiffEq

@testset "EnzymeVJP with view ComponentArray parameters" begin
    # When p is θ.p_all (a ComponentArray view backed by SubArray), zero(p) returns a
    # Vector-backed ComponentArray — a different concrete type. Before the fix,
    # Enzyme.Duplicated(p, zero(p)) would throw a MethodError because primal and shadow
    # had different types. The fix pre-allocates a dense primal buffer at cache build time
    # and copies p into it at each adjoint step.

    function f!(du, u, p, t)
        du[1] = p.a * u[1] + p.b * u[2]
        du[2] = p.c * u[1] - p.a * u[2]
    end

    # θ.p is a ComponentArray view into θ
    θ = ComponentArray(p = ComponentArray(a = 0.5, b = -0.3, c = 0.1), u0 = [1.0, 0.0])
    prob = ODEProblem(f!, θ.u0, (0.0, 1.0), θ.p)

    function loss(θ, prob)
        sol = solve(
            remake(prob, u0 = θ.u0, p = θ.p), Tsit5();
            sensealg = InterpolatingAdjoint(autojacvec = EnzymeVJP()),
            saveat = 0.0:0.25:1.0
        )
        return sum(Array(sol))
    end

    # Both gradient paths must work without error and return finite values
    g = Enzyme.gradient(Enzyme.Reverse, loss, θ, Enzyme.Const(prob))
    @test all(isfinite, g[1])

    # Verify that plain (non-view) ComponentArrays still work correctly
    p_plain = ComponentArray(a = 0.5, b = -0.3, c = 0.1)
    prob_plain = ODEProblem(f!, [1.0, 0.0], (0.0, 1.0), p_plain)
    function loss_plain(p, prob)
        sol = solve(
            prob, Tsit5(); p = p,
            sensealg = InterpolatingAdjoint(autojacvec = EnzymeVJP()),
            saveat = 0.0:0.25:1.0
        )
        return sum(Array(sol))
    end
    g_plain = Enzyme.gradient(Enzyme.Reverse, loss_plain, p_plain, Enzyme.Const(prob_plain))
    @test all(isfinite, g_plain[1])

    # Gradient values should match between view and non-view parameters
    @test g[1].p ≈ g_plain[1]
end
