# Adjoints of ODEs with matrix states. The adjoint methods store λ (and, for BacksolveAdjoint, the
# state) in flat vectors, so the VJPs must reshape them as the state.
using SciMLSensitivity, OrdinaryDiffEqLowOrderRK, ForwardDiff, Zygote, LinearAlgebra, Test
using SciMLSensitivity: MooncakeVJP
using Mooncake

const A = [0.0 1.0; -1.0 -0.1]

# du/dt = p₁ A u + p₂ u Aᵀ
function f!(du, u, p, t)
    mul!(du, A, u, p[1], false)
    mul!(du, u, A', p[2], true)
    return nothing
end

# user-provided VJPs, with λ and dλ given as vectors
function f_vjp!(dλ, λ, u, p, t)
    Λ, dΛ = reshape(λ, size(u)), reshape(dλ, size(u))
    mul!(dΛ, A', Λ, p[1], false)
    mul!(dΛ, Λ, A, p[2], true)
    return dλ
end
function f_vjp_p!(dp, λ, u, p, t)
    Λ = reshape(λ, size(u))
    dp[1] = dot(Λ, A * u)
    dp[2] = dot(Λ, u * A')
    return dp
end

const u0 = [1.0 2.0; 0.3 -0.5]
const p0 = [1.0, 0.5]

function loss(p, f, sensealg)
    prob = ODEProblem{true}(f, eltype(p).(u0), (0.0, 1.0), p)
    sol = solve(prob, DP5(); abstol = 1.0e-10, reltol = 1.0e-10, sensealg)
    return sum(abs2, sol.u[end])
end

const ref = ForwardDiff.gradient(p -> loss(p, f!, nothing), p0)

@testset "$(nameof(typeof(sensealg))) with MooncakeVJP" for sensealg in (
        InterpolatingAdjoint(autojacvec = MooncakeVJP()), BacksolveAdjoint(autojacvec = MooncakeVJP()),
    )
    @test Zygote.gradient(p -> loss(p, f!, sensealg), p0)[1] ≈ ref rtol = 1.0e-6
end

@testset "$(nameof(typeof(sensealg))) with autojacvec = false" for sensealg in (
        InterpolatingAdjoint(autojacvec = false), BacksolveAdjoint(autojacvec = false),
    )
    @test Zygote.gradient(p -> loss(p, f!, sensealg), p0)[1] ≈ ref rtol = 1.0e-6
end

@testset "$(nameof(typeof(sensealg))) with a user-provided VJP" for sensealg in (
        InterpolatingAdjoint(), BacksolveAdjoint(),
    )
    f = ODEFunction{true}(f!; vjp = f_vjp!, vjp_p = f_vjp_p!)
    @test Zygote.gradient(p -> loss(p, f, sensealg), p0)[1] ≈ ref rtol = 1.0e-6
end
