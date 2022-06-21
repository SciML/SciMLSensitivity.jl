using DiffEqSensitivity, OrdinaryDiffEq
using Flux, CUDA, Test, Zygote, Random, LinearAlgebra

CUDA.allowscalar(false)

H = CuArray(rand(Float32, 2, 2))
ann = Chain(Dense(1, 4, tanh))
p,re = Flux.destructure(ann)

function func(x, p, t)
    (re(p)([t])[1]*H)*x
end

x0 = CuArray(rand(Float32, 2))
x1 = CuArray(rand(Float32, 2))

prob = ODEProblem(func, x0, (0.0f0, 1.0f0))

function evolve(p)
    solve(prob, Tsit5(), p=p, save_start=false,
          save_everystep=false, abstol=1e-4, reltol=1e-4,
          sensealg=QuadratureAdjoint(autojacvec=ZygoteVJP())).u[1]
end

function cost(p)
    x = evolve(p)
    c = sum(abs,x - x1)
    #println(c)
    c
end

grad = Zygote.gradient(cost,p)[1]
@test !iszero(grad[1])
@test iszero(grad[2:4])
@test !iszero(grad[5])
@test iszero(grad[6:end])

###
# https://github.com/SciML/DiffEqSensitivity.jl/issues/632
###

rng = MersenneTwister(1234)
m = 32
n = 16
Z = randn(rng, Float32, (n,m)) |> gpu
𝒯 = 2f0
Δτ = 1f-1
ca_init = [zeros(1) ; ones(m)] |> gpu

function f(ca, Z, t)
  a = ca[2:end]
  a_unit = a / sum(a)
  w_unit = Z*a_unit
  Ka_unit = Z'*w_unit
  z_unit = dot(abs.(Ka_unit), a_unit)
  aKa_over_z = a .* Ka_unit / z_unit
  [sum(aKa_over_z) / m; -abs.(aKa_over_z)] |> gpu
end

function c(Z)
  prob = ODEProblem(f, ca_init, (0f0,𝒯), Z, saveat=Δτ)
  sol = solve(prob, Tsit5(), sensealg=BacksolveAdjoint(), saveat=Δτ)
  sum(last(sol.u))
end

println("forward:", c(Z))
println("backward: ", Zygote.gradient(c, Z))
