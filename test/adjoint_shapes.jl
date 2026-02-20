using OrdinaryDiffEq, SciMLSensitivity, ForwardDiff
using SciMLStructures
using Test, Reactant

# This test uses Zygote and inplace_vjp which selects EnzymeVJP on Julia 1.12
# Enzyme has issues on Julia 1.12+, so skip these tests
if VERSION >= v"1.12"
    @info "Skipping adjoint_shapes.jl tests on Julia 1.12+ due to Enzyme compatibility issues"
    @testset "Adjoint Shapes (skipped on Julia 1.12+)" begin
        @test_skip false
    end
else

    using Zygote

    tspan = (0.0, 1.0)
    X = randn(3, 4)
    p = randn(3, 4)
    f(u, p, t) = u .* p
    f(du, u, p, t) = (du .= u .* p)
    prob_ube = ODEProblem{false}(f, X, tspan, p)
    Zygote.gradient(p -> sum(solve(prob_ube, Midpoint(); u0 = X, p)), p)

    prob_ube = ODEProblem{true}(f, X, tspan, p)
    Zygote.gradient(p -> sum(solve(prob_ube, Midpoint(); u0 = X, p)), p)

    # ReactantVJP with matrix-shaped parameters
    prob_oop_r = ODEProblem{false}(f, X, tspan, p)
    g_default = Zygote.gradient(p -> sum(solve(prob_oop_r, Midpoint(); u0 = X, p)), p)[1]
    g_reactant = Zygote.gradient(
        p -> sum(solve(prob_oop_r, Midpoint(); u0 = X, p,
            sensealg = InterpolatingAdjoint(autojacvec = ReactantVJP()))), p)[1]
    @test g_default ≈ g_reactant rtol = 1.0e-6

    prob_iip_r = ODEProblem{true}(f, X, tspan, p)
    g_default_iip = Zygote.gradient(p -> sum(solve(prob_iip_r, Midpoint(); u0 = X, p)), p)[1]
    g_reactant_iip = Zygote.gradient(
        p -> sum(solve(prob_iip_r, Midpoint(); u0 = X, p,
            sensealg = InterpolatingAdjoint(autojacvec = ReactantVJP()))), p)[1]
    @test g_default_iip ≈ g_reactant_iip rtol = 1.0e-6

    function aug_dynamics!(dz, z, K, t)
        x = @view z[2:end]
        u = -K * x
        dz[1] = x' * x + u' * u
        dz[2:end] = x + u
        return nothing
    end

    policy_params = ones(2, 2)
    z0 = zeros(3)
    fwd_sol = solve(
        ODEProblem(aug_dynamics!, z0, (0.0, 1.0), policy_params),
        Tsit5(), u0 = z0, p = policy_params
    )
    _, repack, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), policy_params)

    sensealg = InterpolatingAdjoint()
    sensealg = SciMLSensitivity.setvjp(
        sensealg,
        SciMLSensitivity.inplace_vjp(
            fwd_sol.prob, fwd_sol.prob.u0, fwd_sol.prob.p, true, repack
        )
    )

    solve(
        ODEAdjointProblem(
            fwd_sol, sensealg, Tsit5(),
            [1.0], (out, x, p, t, i) -> (out .= 1)
        ),
        Tsit5()
    )

    A = ones(2, 2)
    B = ones(2, 2)
    Q = ones(2, 2)
    R = ones(2, 2)

    function aug_dynamics!(dz, z, K, t)
        x = @view z[2:end]
        u = -K * x
        dz[1] = x' * Q * x + u' * R * u
        dz[2:end] = A * x + B * u # or just `x + u`
        return nothing
    end

    policy_params = ones(2, 2)
    z0 = zeros(3)
    fwd_sol = solve(
        ODEProblem(aug_dynamics!, z0, (0.0, 1.0), policy_params), u0 = z0,
        p = policy_params, Tsit5()
    )

    solve(
        ODEAdjointProblem(
            fwd_sol, sensealg, Tsit5(), [1.0],
            (out, x, p, t, i) -> (out .= 1)
        ),
        Tsit5()
    )

    # https://github.com/SciML/SciMLSensitivity.jl/issues/581

    p = rand(1)

    function dudt(u, p, t)
        return u .* p
    end

    function loss(p)
        prob = ODEProblem(dudt, [3.0], (0.0, 1.0), p)
        sol = solve(prob, Tsit5(), dt = 0.01, sensealg = ReverseDiffAdjoint())
        return sum(abs2, Array(sol))
    end
    Zygote.gradient(loss, p)[1][1] ≈ ForwardDiff.gradient(loss, p)[1]

end  # VERSION < v"1.12" else block
