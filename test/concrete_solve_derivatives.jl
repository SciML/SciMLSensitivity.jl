using DiffEqSensitivity, OrdinaryDiffEq, Zygote
using Test

function f(du,u,p,t)
  du[1] = dx = p[1]*u[1] - p[2]*u[1]*u[2]
  du[2] = dy = -p[3]*u[2] + p[4]*u[1]*u[2]
end
function foop(u,p,t)
  dx = p[1]*u[1] - p[2]*u[1]*u[2]
  dy = -p[3]*u[2] + p[4]*u[1]*u[2]
  [dx,dy]
end

p = [1.5,1.0,3.0,1.0]; u0 = [1.0;1.0]
prob = ODEProblem(f,u0,(0.0,10.0),p)
proboop = ODEProblem(foop,u0,(0.0,10.0),p)

sol = DiffEqSensitivity.concrete_solve(prob,Tsit5(),abstol=1e-14,reltol=1e-14)
sum(DiffEqSensitivity.concrete_solve(prob,Tsit5(),u0,p,abstol=1e-14,reltol=1e-14))
sum(DiffEqSensitivity.concrete_solve(prob,Tsit5(),u0,p,abstol=1e-14,reltol=1e-14,sensealg=ForwardDiffSensitivity()))
sum(DiffEqSensitivity.concrete_solve(prob,Tsit5(),u0,p,abstol=1e-14,reltol=1e-14,sensealg=BacksolveAdjoint()))

du01,dp1 = Zygote.gradient((u0,p)->sum(DiffEqSensitivity.concrete_solve(prob,Tsit5(),u0,p,abstol=1e-14,reltol=1e-14,sensealg=QuadratureAdjoint())),u0,p)
du02,dp2 = Zygote.gradient((u0,p)->sum(DiffEqSensitivity.concrete_solve(prob,Tsit5(),u0,p,abstol=1e-14,reltol=1e-14,sensealg=InterpolatingAdjoint())),u0,p)
du03,dp3 = Zygote.gradient((u0,p)->sum(DiffEqSensitivity.concrete_solve(prob,Tsit5(),u0,p,abstol=1e-14,reltol=1e-14,sensealg=BacksolveAdjoint())),u0,p)
du04,dp4 = Zygote.gradient((u0,p)->sum(DiffEqSensitivity.concrete_solve(prob,Tsit5(),u0,p,abstol=1e-14,reltol=1e-14,sensealg=TrackerAdjoint())),u0,p)
@test_broken Zygote.gradient((u0,p)->sum(DiffEqSensitivity.concrete_solve(prob,Tsit5(),u0,p,abstol=1e-14,reltol=1e-14,sensealg=ZygoteAdjoint())),u0,p) isa Tuple
du06,dp6 = Zygote.gradient((u0,p)->sum(DiffEqSensitivity.concrete_solve(prob,Tsit5(),u0,p,abstol=1e-14,reltol=1e-14,sensealg=ForwardSensitivity())),u0,p)
du07,dp7 = Zygote.gradient((u0,p)->sum(DiffEqSensitivity.concrete_solve(prob,Tsit5(),u0,p,abstol=1e-14,reltol=1e-14,sensealg=ForwardDiffSensitivity())),u0,p)
