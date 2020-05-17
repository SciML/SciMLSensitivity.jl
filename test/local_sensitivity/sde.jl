using Test, LinearAlgebra
using OrdinaryDiffEq
using DiffEqSensitivity, StochasticDiffEq, DiffEqBase
using ForwardDiff, Calculus, ReverseDiff
using Random
using Plots


seed = 100
Random.seed!(seed)
abstol = 1e-4
reltol = 1e-4


numtraj = 10
u₀ = [0.5]
tstart = 0.0
tend = 1.0
dt = 0.05
trange = (tstart, tend)
t = tstart:dt:tend


f_oop_linear(u,p,t) = p[1]*u
σ_oop_linear(u,p,t) = p[2]*u
linear_analytic_ode(u0,p,t) = @.(u0*exp(p[1]t))
linear_analytic(u0,p,t,W) = @.(u0*exp((p[1]-p[2]^2/2)t+p[2]*W))
linear_analytic_du_du0(u0,p,t,W) = @.(exp((p[1]-p[2]^2/2)t+p[2]*W))
linear_analytic_du_dp1(u0,p,t,W) = @.(t*u0*exp((p[1]-p[2]^2/2)t+p[2]*W))
linear_analytic_du_dp2(u0,p,t,W) = @.((W-p[2]*t)*u0*exp((p[1]-p[2]^2/2)t+p[2]*W))

function g(u,p,t)
  sum(u.^2/2)
end

function dg!(out,u,p,t,i)
  (out.=-u)
end

p = [1.01,0.0]

# generate ODE adjoint results

prob_oop_ode = ODEProblem(f_oop_linear,u₀,(tstart,tend),p)
sol_oop_ode = solve(prob_oop_ode,Tsit5(),saveat=t,abstol=abstol,reltol=reltol)
res_ode_u0, res_ode_p = adjoint_sensitivities(sol_oop_ode,Tsit5(),dg!,t
	,abstol=abstol,reltol=reltol,sensealg=BacksolveAdjoint())

function G(p)
  tmp_prob = remake(prob_oop_ode,u0=eltype(p).(prob_oop_ode.u0),p=p,
                    tspan=eltype(p).(prob_oop_ode.tspan),abstol=abstol, reltol=reltol)
  sol = solve(tmp_prob,Tsit5(),saveat=t,abstol=abstol, reltol=reltol)
  #@show sol
  res = g(sol,p,nothing)
  @show res
  res
end
res_ode_forward = ForwardDiff.gradient(G,p)
#res_ode_reverse = ReverseDiff.gradient(G,p)
@test isapprox(res_ode_forward[1], sum(@. u₀^2*exp(2*p[1]*t)*t), rtol = 1e-4)
#@test isapprox(res_ode_reverse[1], sum(@. u₀^2*exp(2*p[1]*t)*t), rtol = 1e-4)
@test isapprox(res_ode_p'[1], sum(@. u₀^2*exp(2*p[1]*t)*t), rtol = 1e-4)


# SDE adjoint results (with noise == 0, so should agree with above)

Random.seed!(seed)
prob_oop_sde = SDEProblem(f_oop_linear,σ_oop_linear,u₀,trange,p)
sol_oop_sde = solve(prob_oop_sde,RKMil(interpretation=:Stratonovich),dt=tend/20,adaptive=false,save_noise=true)
# res_sde_u0, res_sde_p = adjoint_sensitivities(sol_oop_sde,RKMil(interpretation=:Stratonovich),dg!,t
# 	,abstol=abstol,reltol=reltol,sensealg=BacksolveAdjoint()) #

adj_prob2, tstops2 = adjoint_sensitivities(sol_oop_sde,RKMil(interpretation=:Stratonovich),dg!,t
 	,abstol=abstol,reltol=reltol,sensealg=BacksolveAdjoint())
solve(adj_prob2, RKMil(interpretation=:Stratonovich); tstops=tstops2)


adj_prob1, tstops1 = adjoint_sensitivities(sol_oop_ode,Tsit5(),dg!,t
	,abstol=abstol,reltol=reltol,sensealg=BacksolveAdjoint())
solve(adj_prob1, Tsit5())


prob_oop_sde
adj_prob2



prob_oop_ode
adj_prob1












function GSDE(p)
  Random.seed!(seed)
  tmp_prob = remake(prob_oop_sde,u0=eltype(p).(prob_oop_sde.u0),p=p,
                    tspan=eltype(p).(prob_oop_sde.tspan)
					#,abstol=abstol, reltol=reltol
					)
  sol = solve(tmp_prob,RKMil(interpretation=:Stratonovich),dt=tend/10000,adaptive=false,saveat=t)
  A = convert(Array,sol)
  res = g(A,p,nothing)
  @show res
  res
end
res_sde_forward = ForwardDiff.gradient(GSDE,p)
res_sde_reverse = ReverseDiff.gradient(GSDE,p)
@test isapprox(res_sde_forward[1], sum(@. u₀^2*exp(2*p[1]*t)*t), rtol = 1e-2)
@test isapprox(res_sde_reverse[1], sum(@. u₀^2*exp(2*p[1]*t)*t), rtol = 1e-2)






# using StochasticDiffEq, DiffEqNoiseProcess, Plots
# α=1
# β=1
# u₀=1/2
# f(u,p,t) = α*u
# g(u,p,t) = β*u
# dt = 1//2^(4)
# tspan = (0.0,1.0)
# prob = SDEProblem(f,g,u₀,(0.0,1.0))
# sol = solve(prob,EulerHeun(),dt=0.01,save_noise=true)
# _sol = deepcopy(sol) # to make sure the plot is correct
# W3 = NoiseGrid(reverse!(_sol.t),reverse!(_sol.W))
#
# f!(du,u,p,t) = @. du=α*u
# g!(du,u,p,t) = @. du=β*u
#
# prob3 = SDEProblem(f!,g!,[0.0,0.0,0.0,sol[end]],(1.0,0.0),noise=W3)
# sol2 = solve(prob3,EulerHeun(),dt=0.01)
# plot(sol.t,sol.u)
# plot!(sol2.t,sol2.u)
