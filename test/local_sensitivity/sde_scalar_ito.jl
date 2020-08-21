using Test, LinearAlgebra
using DiffEqSensitivity, StochasticDiffEq
using Random

@info "SDE scalar Adjoints"

seed = 5
Random.seed!(seed)

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

p2 = [1.01,0.37]

#using DiffEqNoiseProcess

dtscalar = tend/1e4

f(u,p,t) = p[1]*u
σ(u,p,t) = p[2]*u
linear_analytic(u0,p,t,W) = @.(u0*exp((p[1]-p[2]^2/2)*t+p[2]*W))
corfunc(u,p,t) = p[2]^2*u

"""
1D oop
"""

Random.seed!(seed)
#W = WienerProcess(0.0,0.0,0.0)
u0 = rand(1)


Random.seed!(seed)
prob = SDEProblem(SDEFunction(f,σ,
  #analytic=linear_analytic
  ),σ,u0,trange,p2,
  #noise=W
 )
sol = solve(prob,EM(), dt=dtscalar, adaptive=false, save_noise=true)
#@test isapprox(sol.u_analytic,sol.u, atol=1e-4)

res_sde_u0a, res_sde_pa = adjoint_sensitivities(sol,EM(),dg!,Array(t)
  ,dt=dtscalar,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=DiffEqSensitivity.ReverseDiffVJP()))

@info res_sde_u0a, res_sde_pa

res_sde_u0b, res_sde_pb = adjoint_sensitivities(sol,EM(),dg!,Array(t)
  ,dt=dtscalar,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=false))

@info res_sde_u0b, res_sde_pb

@test isapprox(res_sde_u0a, res_sde_u0b, rtol=1e-9)
@test isapprox(res_sde_pa, res_sde_pb, rtol=1e-9)

res_sde_u0b, res_sde_pb = adjoint_sensitivities(sol,EM(),dg!,Array(t)
  ,dt=dtscalar,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=false), corfunc_analytical=corfunc)

@info res_sde_u0b, res_sde_pb

@test isapprox(res_sde_u0a, res_sde_u0b, rtol=1e-9)
@test isapprox(res_sde_pa, res_sde_pb, rtol=1e-9)


using ForwardDiff
function GSDE(p)
  Random.seed!(seed)
  tmp_prob = remake(prob,u0=eltype(p).(prob.u0),p=p,
                  tspan=eltype(p).(prob.tspan))
  _sol = solve(tmp_prob,EM(),dt=dtscalar,adaptive=false,saveat=Array(t), sensealg=DiffEqBase.SensitivityADPassThrough())
  A = convert(Array,_sol)
  res = g(A,p,nothing)
end

res_sde_forward = ForwardDiff.gradient(GSDE,p2)

@info res_sde_forward

Wfix = [sol.W(t)[1][1] for t in tarray]
resp1 = sum(@. tarray*u0^2*exp(2*(p2[1]-p2[2]^2/2)*tarray+2*p2[2]*Wfix))
resp2 = sum(@. (Wfix-p2[2]*tarray)*u0^2*exp(2*(p2[1]-p2[2]^2/2)*tarray+2*p2[2]*Wfix))
resp = [resp1, resp2]

@test isapprox(resp, res_sde_pa', rtol=1e-3)
@test isapprox(resp, res_sde_forward, rtol=1e-3)
