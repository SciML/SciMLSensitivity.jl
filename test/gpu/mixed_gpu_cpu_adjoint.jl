using SciMLSensitivity, OrdinaryDiffEq
using Lux, LuxCUDA, Test, Zygote, Random, LinearAlgebra, ComponentArrays

const gdev = gpu_device()

CUDA.allowscalar(false)

H = CuArray(rand(Float32, 2, 2))
ann = Lux.Chain(Lux.Dense(1, 4, tanh))
rng = Random.default_rng()
p, st = Lux.setup(rng, ann)
p = ComponentArray(p)
const _st = st

function func(x, p, t)
    CuArray(reshape(first(ann([t], p, _st)), 2, 2)) * H * x
end

x0 = CuArray(rand(Float32, 2))
x1 = CuArray(rand(Float32, 2))

prob = ODEProblem(func, x0, (0.0f0, 1.0f0))

function evolve(p)
    solve(prob, Tsit5(), p = p, save_start = false,
    save_everystep = false, abstol = 1e-4, reltol = 1e-4,
    sensealg = QuadratureAdjoint(autojacvec = ZygoteVJP())).u[1]
end

function cost(p)
    x = evolve(p)
    c = sum(abs, x - x1)
    #println(c)
    c
end

grad = Zygote.gradient(cost, p)[1]
@test !iszero(grad)

###
# https://github.com/SciML/SciMLSensitivity.jl/issues/632
###

rng = MersenneTwister(1234)
m = 32
n = 16
Z = randn(rng, Float32, (n, m)) |> gdev
𝒯 = 2.0f0
Δτ = 1.0f-1
ca_init = [zeros(1); ones(m)] |> gdev

function f(ca, Z, t)
    a = ca[2:end]
    a_unit = a / sum(a)
    w_unit = Z * a_unit
    Ka_unit = Z' * w_unit
    z_unit = dot(abs.(Ka_unit), a_unit)
    aKa_over_z = a .* Ka_unit / z_unit
    [sum(aKa_over_z) / m; -abs.(aKa_over_z)] |> gdev
end

function c(Z)
    prob = ODEProblem(f, ca_init, (0.0f0, 𝒯), Z, saveat = Δτ)
    sol = solve(prob, Tsit5(), sensealg = BacksolveAdjoint(), saveat = Δτ)
    sum(last(sol.u))
end

println("forward:", c(Z))
println("backward: ", Zygote.gradient(c, Z))
