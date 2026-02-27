using SciMLSensitivity, OrdinaryDiffEq, ForwardDiff, LinearAlgebra, Zygote, SciMLBase
using Test

# Lotka-Volterra system with user-provided VJP, paramjac, and vjp_p
function lotka_volterra!(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = α * x - β * x * y
    return du[2] = -δ * y + γ * x * y
end

# State VJP: (df/du)^T * v
function lv_vjp!(dλ, λ, u, p, t)
    x, y = u
    α, β, δ, γ = p
    dλ[1] = λ[1] * (α - β * y) + λ[2] * (γ * y)
    return dλ[2] = λ[1] * (-β * x) + λ[2] * (-δ + γ * x)
end

# Full parameter Jacobian: df/dp (n_states × n_params matrix)
function lv_paramjac!(pJ, u, p, t)
    x, y = u
    pJ .= 0
    pJ[1, 1] = x
    pJ[1, 2] = -x * y
    pJ[2, 3] = -y
    return pJ[2, 4] = x * y
end

# Parameter VJP: (df/dp)^T * v (vector of length n_params)
function lv_vjp_p!(Jpv, v, u, p, t)
    x, y = u
    Jpv[1] = x * v[1]
    Jpv[2] = -x * y * v[1]
    Jpv[3] = -y * v[2]
    return Jpv[4] = x * y * v[2]
end

p = [1.5, 1.0, 3.0, 1.0]
u0 = [1.0, 1.0]
tspan = (0.0, 10.0)
solver_kwargs = (abstol = 1.0e-12, reltol = 1.0e-12)

# ForwardDiff baseline
function loss_fwd(p)
    prob = ODEProblem(lotka_volterra!, u0, tspan, p)
    sol = solve(prob, Tsit5(); saveat = 0.1, solver_kwargs...)
    return sum(sol)
end
grad_fwd = ForwardDiff.gradient(loss_fwd, p)

# Helper to compute gradient for a given ODEFunction and sensealg
function compute_grad(f, sensealg)
    prob = ODEProblem(f, u0, tspan, p)
    loss = p -> sum(
        solve(
            prob, Tsit5(); p = p, saveat = 0.1, solver_kwargs...,
            sensealg = sensealg
        )
    )
    return Zygote.gradient(loss, p)[1]
end

@testset "User-provided VJP dispatch" begin
    @testset "VJP + vjp_p: $name" for (name, sensealg) in [
            ("GaussAdjoint", GaussAdjoint(autojacvec = EnzymeVJP())),
            (
                "QuadratureAdjoint",
                QuadratureAdjoint(
                    autojacvec = EnzymeVJP(), abstol = 1.0e-12, reltol = 1.0e-12
                ),
            ),
            ("InterpolatingAdjoint", InterpolatingAdjoint(autojacvec = EnzymeVJP())),
            ("BacksolveAdjoint", BacksolveAdjoint(autojacvec = EnzymeVJP())),
        ]
        f = ODEFunction(lotka_volterra!; vjp = lv_vjp!, vjp_p = lv_vjp_p!)
        grad = compute_grad(f, sensealg)
        @test isapprox(grad, grad_fwd, rtol = 1.0e-5)
    end

    @testset "VJP + paramjac: $name" for (name, sensealg) in [
            ("GaussAdjoint", GaussAdjoint(autojacvec = EnzymeVJP())),
            (
                "QuadratureAdjoint",
                QuadratureAdjoint(
                    autojacvec = EnzymeVJP(), abstol = 1.0e-12, reltol = 1.0e-12
                ),
            ),
            ("InterpolatingAdjoint", InterpolatingAdjoint(autojacvec = EnzymeVJP())),
            ("BacksolveAdjoint", BacksolveAdjoint(autojacvec = EnzymeVJP())),
        ]
        f = ODEFunction(lotka_volterra!; vjp = lv_vjp!, paramjac = lv_paramjac!)
        grad = compute_grad(f, sensealg)
        @test isapprox(grad, grad_fwd, rtol = 1.0e-5)
    end

    @testset "vjp_p matches paramjac exactly: $name" for (name, sensealg) in [
            ("GaussAdjoint", GaussAdjoint(autojacvec = EnzymeVJP())),
            (
                "QuadratureAdjoint",
                QuadratureAdjoint(
                    autojacvec = EnzymeVJP(), abstol = 1.0e-12, reltol = 1.0e-12
                ),
            ),
            ("InterpolatingAdjoint", InterpolatingAdjoint(autojacvec = EnzymeVJP())),
            ("BacksolveAdjoint", BacksolveAdjoint(autojacvec = EnzymeVJP())),
        ]
        f_vjpp = ODEFunction(lotka_volterra!; vjp = lv_vjp!, vjp_p = lv_vjp_p!)
        f_pjac = ODEFunction(lotka_volterra!; vjp = lv_vjp!, paramjac = lv_paramjac!)
        grad_vjpp = compute_grad(f_vjpp, sensealg)
        grad_pjac = compute_grad(f_pjac, sensealg)
        @test isapprox(grad_vjpp, grad_pjac, rtol = 1.0e-10)
    end

    @testset "vjp_p takes priority over paramjac" begin
        calls_vjp_p = Ref(0)
        calls_paramjac = Ref(0)

        function counting_vjp_p!(Jpv, v, u, p, t)
            calls_vjp_p[] += 1
            lv_vjp_p!(Jpv, v, u, p, t)
        end
        function counting_paramjac!(pJ, u, p, t)
            calls_paramjac[] += 1
            lv_paramjac!(pJ, u, p, t)
        end

        f = ODEFunction(
            lotka_volterra!;
            vjp = lv_vjp!,
            paramjac = counting_paramjac!,
            vjp_p = counting_vjp_p!
        )

        grad = compute_grad(f, GaussAdjoint(autojacvec = EnzymeVJP()))
        @test isapprox(grad, grad_fwd, rtol = 1.0e-5)
        @test calls_vjp_p[] > 0
        @test calls_paramjac[] == 0
    end

    @testset "VJP-only (no paramjac or vjp_p): $name" for (name, sensealg) in [
            ("GaussAdjoint", GaussAdjoint(autojacvec = ReverseDiffVJP())),
            ("InterpolatingAdjoint", InterpolatingAdjoint(autojacvec = ReverseDiffVJP())),
        ]
        f = ODEFunction(lotka_volterra!; vjp = lv_vjp!)
        grad = compute_grad(f, sensealg)
        @test isapprox(grad, grad_fwd, rtol = 1.0e-3)
    end
end
