using Test, LinearAlgebra
using DiffEqSensitivity, StochasticDiffEq
using Random

@info "SDE Checkpointing"

seed = 100
Random.seed!(seed)
abstol = 1e-4
reltol = 1e-4

u₀ = [0.5]
tstart = 0.0
tend = 0.1
dt = 0.005
trange = (tstart, tend)
t = tstart:dt:tend
tarray = collect(t)

function g(u,p,t)
  sum(u.^2.0/2.0)
end

function dg!(out,u,p,t,i)
  (out.=-u)
end

p2 = [1.01,0.87]



f_oop_linear(u,p,t) = p[1]*u
σ_oop_linear(u,p,t) = p[2]*u

dt1 = tend/1e3

Random.seed!(seed)
prob_oop = SDEProblem(f_oop_linear,σ_oop_linear,u₀,trange,p2)
sol_oop = solve(prob_oop,RKMil(interpretation=:Stratonovich),dt=dt1,adaptive=false,save_noise=true)


res_u0, res_p = adjoint_sensitivities(sol_oop,EulerHeun(),dg!,Array(t)
      ,dt=dt1,adaptive=false,sensealg=InterpolatingAdjoint(noise=DiffEqSensitivity.ZygoteNoise()))
