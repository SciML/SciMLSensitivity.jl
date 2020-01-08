using DiffEqSensitivity, OrdinaryDiffEq, Zygote
using DiffEqSensitivity: concrete_solve
using RecursiveArrayTools: DiffEqArray
using Test

function fiip(du,u,p,t)
  du[1] = dx = p[1]*u[1] - p[2]*u[1]*u[2]
  du[2] = dy = -p[3]*u[2] + p[4]*u[1]*u[2]
end
function foop(u,p,t)
  dx = p[1]*u[1] - p[2]*u[1]*u[2]
  dy = -p[3]*u[2] + p[4]*u[1]*u[2]
  [dx,dy]
end

p = [1.5,1.0,3.0,1.0]; u0 = [1.0;1.0]
prob = ODEProblem(fiip,u0,(0.0,10.0),p)
proboop = ODEProblem(foop,u0,(0.0,10.0),p)

sol = concrete_solve(prob,Tsit5(),abstol=1e-14,reltol=1e-14)
@test sol isa DiffEqArray
sumsol = sum(sol)
@test sum(concrete_solve(prob,Tsit5(),u0,p,abstol=1e-14,reltol=1e-14)) == sumsol
@test sum(concrete_solve(prob,Tsit5(),u0,p,abstol=1e-14,reltol=1e-14,sensealg=ForwardDiffSensitivity())) == sumsol
@test sum(concrete_solve(prob,Tsit5(),u0,p,abstol=1e-14,reltol=1e-14,sensealg=BacksolveAdjoint())) == sumsol


###
### adjoint
###

_sol = solve(prob,Tsit5(),abstol=1e-14,reltol=1e-14)
ū0,adj = adjoint_sensitivities_u0(_sol,Tsit5(),((out,u,p,t,i) -> out .= -1),sol.t,abstol=1e-14,
                                  reltol=1e-14,iabstol=1e-14,ireltol=1e-12)
#du01,dp1 = Zygote.gradient((u0,p)->sum(concrete_solve(prob,Tsit5(),u0,p,abstol=1e-14,reltol=1e-14,sensealg=QuadratureAdjoint())),u0,p) # Can't get sensitivities of u0 with quadrature.
du02,dp2 = Zygote.gradient((u0,p)->sum(concrete_solve(prob,Tsit5(),u0,p,abstol=1e-14,reltol=1e-14,sensealg=InterpolatingAdjoint())),u0,p)
du03,dp3 = Zygote.gradient((u0,p)->sum(concrete_solve(prob,Tsit5(),u0,p,abstol=1e-14,reltol=1e-14,sensealg=BacksolveAdjoint())),u0,p)
du04,dp4 = Zygote.gradient((u0,p)->sum(concrete_solve(prob,Tsit5(),u0,p,abstol=1e-14,reltol=1e-14,sensealg=TrackerAdjoint())),u0,p)
@test_broken Zygote.gradient((u0,p)->sum(concrete_solve(prob,Tsit5(),u0,p,abstol=1e-14,reltol=1e-14,sensealg=ZygoteAdjoint())),u0,p) isa Tuple

#@test ū0 ≈ du01 rtol=1e-15
@test ū0 == du02
@test ū0 ≈ du03 rtol=1e-12
@test_broken ū0 ≈ du04 rtol=1e-12
#@test adj ≈ dp1' rtol=1e-15
@test adj == dp2'
@test adj ≈ dp3' rtol=1e-12
@test_broken adj ≈ dp4' rtol=1e-12

###
### forward
###

du06,dp6 = Zygote.gradient((u0,p)->sum(concrete_solve(prob,Tsit5(),u0,p,abstol=1e-14,reltol=1e-14,saveat=sol.t,sensealg=ForwardSensitivity())),u0,p)
du07,dp7 = Zygote.gradient((u0,p)->sum(concrete_solve(prob,Tsit5(),u0,p,abstol=1e-14,reltol=1e-14,saveat=sol.t,sensealg=ForwardDiffSensitivity())),u0,p)

@test du06 === du07 === nothing
@test adj ≈ dp6' rtol=1e-12
@test adj ≈ dp7' rtol=1e-12
