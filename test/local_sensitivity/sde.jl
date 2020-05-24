using Test, LinearAlgebra
using OrdinaryDiffEq
using DiffEqSensitivity, StochasticDiffEq, DiffEqBase
using ForwardDiff, Calculus, ReverseDiff
using Random
import Tracker, Zygote

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

f_oop_linear(u,p,t) = p[1]*u
σ_oop_linear(u,p,t) = p[2]*u

function f_oop_linear(u::Tracker.TrackedArray,p,t)
  dx = p[1]*u[1]
  Tracker.collect([dx])
end

function σ_oop_linear(u::Tracker.TrackedArray,p,t)
  dx = p[2]*u[1]
  Tracker.collect([dx])
end

function g(u,p,t)
  sum(u.^2.0/2.0)
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
  sol = solve(tmp_prob,Tsit5(),saveat=Array(t),abstol=abstol, reltol=reltol)
  res = g(sol,p,nothing)
end
res_ode_forward = ForwardDiff.gradient(G,p)
#res_ode_reverse = ReverseDiff.gradient(G,p)

res_ode_trackeru0, res_ode_trackerp = Zygote.gradient((u0,p)->sum(concrete_solve(prob_oop_ode,Tsit5(),u0,p,abstol=abstol,reltol=reltol,saveat=Array(t),sensealg=TrackerAdjoint()).^2.0/2.0),u₀,p)

@test isapprox(res_ode_forward[1], sum(@. u₀^2*exp(2*p[1]*t)*t), rtol = 1e-4)
#@test isapprox(res_ode_reverse[1], sum(@. u₀^2*exp(2*p[1]*t)*t), rtol = 1e-4)
@test isapprox(res_ode_p'[1], sum(@. u₀^2*exp(2*p[1]*t)*t), rtol = 1e-4)
@test isapprox(res_ode_p', res_ode_trackerp, rtol = 1e-4)

# SDE adjoint results (with noise == 0, so should agree with above)

Random.seed!(seed)
prob_oop_sde = SDEProblem(f_oop_linear,σ_oop_linear,u₀,trange,p)
sol_oop_sde = solve(prob_oop_sde,RKMil(interpretation=:Stratonovich),dt=1e-4,adaptive=false,save_noise=true)
res_sde_u0, res_sde_p = adjoint_sensitivities(sol_oop_sde,
	EulerHeun(),dg!,t,dt=1e-2,sensealg=BacksolveAdjoint())



function GSDE1(p)
  Random.seed!(seed)
  tmp_prob = remake(prob_oop_sde,u0=eltype(p).(prob_oop_sde.u0),p=p,
                    tspan=eltype(p).(prob_oop_sde.tspan)
					#,abstol=abstol, reltol=reltol
					)
  sol = solve(tmp_prob,RKMil(interpretation=:Stratonovich),dt=tend/10000,adaptive=false,saveat=t)
  A = convert(Array,sol)
  res = g(A,p,nothing)
end
res_sde_forward = ForwardDiff.gradient(GSDE1,p)
res_sde_reverse = ReverseDiff.gradient(GSDE1,p)

res_sde_trackeru0, res_sde_trackerp = Zygote.gradient((u0,p)->sum(concrete_solve(prob_oop_sde,RKMil(interpretation=:Stratonovich),dt=tend/1400,adaptive=false,u0,p,saveat=Array(t),sensealg=TrackerAdjoint()).^2.0/2.0),u₀,p)

noise = vec((@. sol_oop_sde.W(tarray)))
Wfix = [W[1][1] for W in noise]
@test isapprox(res_sde_forward[1], sum(@. u₀^2*exp(2*p[1]*t)*t), rtol = 1e-4)
@test isapprox(res_sde_reverse[1], sum(@. u₀^2*exp(2*p[1]*t)*t), rtol = 1e-4)
@test isapprox(res_sde_p'[1], sum(@. u₀^2*exp(2*p[1]*t)*t), rtol = 1e-4)
@test isapprox(res_sde_p'[2], sum(@. (Wfix)*u₀^2*exp(2*(p[1])*tarray+2*p[2]*Wfix)), rtol = 1e-4)
@test isapprox(res_sde_p'[1], res_sde_trackerp[1], rtol = 1e-4)

# SDE adjoint results (with noise != 0)

p2 = [1.01,0.87]

Random.seed!(seed)
prob_oop_sde2 = SDEProblem(f_oop_linear,σ_oop_linear,u₀,trange,p2)
sol_oop_sde2 = solve(prob_oop_sde2,RKMil(interpretation=:Stratonovich),dt=tend/1e6,adaptive=false,save_noise=true)
res_sde_u02, res_sde_p2 = adjoint_sensitivities(sol_oop_sde2,EulerHeun(),dg!,Array(t)
 	,dt=tend/1e6,adaptive=false,sensealg=BacksolveAdjoint())

function GSDE2(p)
  Random.seed!(seed)
  tmp_prob = remake(prob_oop_sde2,u0=eltype(p).(prob_oop_sde2.u0),p=p,
                    tspan=eltype(p).(prob_oop_sde2.tspan)
					#,abstol=abstol, reltol=reltol
					)
  sol = solve(tmp_prob,RKMil(interpretation=:Stratonovich),dt=tend/1e6,adaptive=false,saveat=Array(t))
  A = convert(Array,sol)
  res = g(A,p,nothing)
end
res_sde_forward2 = ForwardDiff.gradient(GSDE2,p2)
res_sde_reverse2 = ReverseDiff.gradient(GSDE2,p2)


Random.seed!(seed)
res_sde_trackeru02, res_sde_trackerp2 = Zygote.gradient((u0,p)->sum(concrete_solve(prob_oop_sde2,RKMil(interpretation=:Stratonovich),dt=tend/1e3,adaptive=false,u0,p,saveat=Array(t),sensealg=TrackerAdjoint()).^2.0/2.0),u₀,p2)


noise = vec((@. sol_oop_sde2.W(tarray)))
Wfix = [W[1][1] for W in noise]
resp1 = sum(@. tarray*u₀^2*exp(2*(p2[1])*tarray+2*p2[2]*Wfix))
resp2 = sum(@. (Wfix)*u₀^2*exp(2*(p2[1])*tarray+2*p2[2]*Wfix))
resp = [resp1, resp2]

@test isapprox(res_sde_forward2, resp, rtol = 1e-6)
@test isapprox(res_sde_reverse2, resp, rtol = 1e-6)
@test isapprox(res_sde_trackerp2, resp, rtol = 4e-1)

@test isapprox(res_sde_p2', res_sde_forward2, rtol = 1e-6)
@test isapprox(res_sde_p2', resp, rtol = 1e-6)

@info "ForwardDiff" res_sde_forward2
@info "ReverseDiff" res_sde_reverse2
@info "Exact" resp
@info "Adjoint SDE" res_sde_p2


# consistency check with respect to tracker
Random.seed!(seed)
prob_oop_sde2 = SDEProblem(f_oop_linear,σ_oop_linear,u₀,trange,p2)
sol_oop_sde2 = solve(prob_oop_sde2,RKMil(interpretation=:Stratonovich),dt=tend/1e3,adaptive=false,save_noise=true)
res_sde_u02, res_sde_p2 = adjoint_sensitivities(sol_oop_sde2,EulerHeun(),dg!,Array(t)
 	,dt=tend/1e3,adaptive=false,sensealg=BacksolveAdjoint())

@test isapprox(res_sde_p2', res_sde_trackerp2, rtol = 3e-4)

# Free memory to help Travis

noise = nothing
Wfix = nothing
res_sde_forward2 = nothing
res_sde_reverse2 = nothing
resp = nothing
res_sde_trackerp2 = nothing
res_sde_u02 = nothing
sol_oop_sde2 = nothing
res_sde_p2 = nothing
sol_oop_sde = nothing
GC.gc()

# SDE adjoint results with diagonal noise

Random.seed!(seed)
prob_oop_sde2 = SDEProblem(f_oop_linear,σ_oop_linear,[u₀;u₀;u₀],trange,p2)
sol_oop_sde2 = solve(prob_oop_sde2,EulerHeun(),
	dt=tend/1e6,adaptive=false,save_noise=true)

@info "Diagonal Adjoint"

res_sde_u02, res_sde_p2 = adjoint_sensitivities(sol_oop_sde2,EulerHeun(),dg!,Array(t)
 	,dt=tend/1e6,adaptive=false,sensealg=BacksolveAdjoint())

sol_oop_sde2 = nothing
GC.gc()

@info "Diagonal ForwardDiff"
res_sde_forward2 = ForwardDiff.gradient(GSDE2,p2)
@info "Diagonal ReverseDiff"
res_sde_reverse2 = ReverseDiff.gradient(GSDE2,p2)

@test isapprox(res_sde_forward2, res_sde_reverse2, rtol = 1e-6)
@test isapprox(res_sde_p2', res_sde_forward2, rtol = 1e-3)
@test isapprox(res_sde_p2', res_sde_reverse2, rtol = 1e-3)


# u0
function GSDE3(u)
  Random.seed!(seed)
  tmp_prob = remake(prob_oop_sde2,u0=u,p=eltype(p).(prob_oop_sde2.p),
                    tspan=eltype(p).(prob_oop_sde2.tspan)
					#,abstol=abstol, reltol=reltol
					)
  sol = solve(tmp_prob,RKMil(interpretation=:Stratonovich),dt=tend/1e5,adaptive=false,saveat=Array(t))
  A = convert(Array,sol)
  res = g(A,p,nothing)
end

res_sde_forward2 = ForwardDiff.gradient(GSDE3,[u₀;u₀;u₀])

@test isapprox(res_sde_u02, res_sde_forward2, rtol = 1e-5)
