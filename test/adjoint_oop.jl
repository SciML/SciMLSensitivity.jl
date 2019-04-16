using DiffEqSensitivity,OrdinaryDiffEq, ParameterizedFunctions,
      RecursiveArrayTools, DiffEqBase, ForwardDiff, Calculus, QuadGK,
      LinearAlgebra
using Test

A = rand(2)
f(u,p,t) = p[1].*A.*u./p[2].^2
p = [1.5,3.0]

prob = ODEProblem{false}(f,[1.0;1.0],(0.0,10.0),p)
sol = solve(prob,Tsit5(),abstol=1e-14,reltol=1e-14)
sol_end = solve(prob,Tsit5(),abstol=1e-14,reltol=1e-14,
          save_everystep=false,save_start=false)

# Do a discrete adjoint problem
println("Calculate discrete adjoint sensitivities")
t = 0.0:0.5:10.0 # TODO: Add end point handling for callback
# g(t,u,i) = (1-u)^2/2, L2 away from 1
function dg(out,u,p,t,i)
  (out.=2.0.-u)
end

easy_res = adjoint_sensitivities(sol,Tsit5(),dg,t,abstol=1e-14,
                                 reltol=1e-14,iabstol=1e-14,ireltol=1e-12)
easy_res2 = adjoint_sensitivities(sol,Tsit5(),dg,t,abstol=1e-14,
                                 reltol=1e-14,iabstol=1e-14,ireltol=1e-12,sensealg=SensitivityAlg(quad=true,backsolve=false))
easy_res3 = adjoint_sensitivities(sol,Tsit5(),dg,t,abstol=1e-14,
                                  reltol=1e-14,iabstol=1e-14,ireltol=1e-12,sensealg=SensitivityAlg(quad=false,backsolve=false))
easy_res4 = adjoint_sensitivities(sol,Tsit5(),dg,t,abstol=1e-14,
                                  reltol=1e-14,iabstol=1e-14,ireltol=1e-12,sensealg=SensitivityAlg(backsolve=true))
easy_res5 = adjoint_sensitivities(sol,Kvaerno5(nlsolve=NLAnderson(), smooth_est=false),dg,t,abstol=1e-12,
                                 reltol=1e-10,iabstol=1e-14,ireltol=1e-12,sensealg=SensitivityAlg(backsolve=true))
easy_res6 = adjoint_sensitivities(sol,Tsit5(),dg,t,abstol=1e-14,
                                  reltol=1e-14,iabstol=1e-14,ireltol=1e-12,
                                  sensealg=SensitivityAlg(checkpointing=true,quad=true),
                                  checkpoints=sol.t[1:5:end])
easy_res7 = adjoint_sensitivities(sol,Tsit5(),dg,t,abstol=1e-14,
                                  reltol=1e-14,iabstol=1e-14,ireltol=1e-12,
                                  sensealg=SensitivityAlg(checkpointing=true,quad=false),
                                  checkpoints=sol.t[1:5:end])

adj_prob = ODEAdjointProblem(sol,dg,t)
adj_sol = solve(adj_prob,Tsit5(),abstol=1e-14,reltol=1e-14)
integrand = AdjointSensitivityIntegrand(sol,adj_sol)
res,err = quadgk(integrand,0.0,10.0,atol=1e-14,rtol=1e-12)

@test isapprox(res, easy_res, rtol = 1e-10)
@test isapprox(res, easy_res2, rtol = 1e-10)
@test isapprox(res, easy_res3, rtol = 1e-10)
@test isapprox(res, easy_res4, rtol = 1e-10)
@test isapprox(res, easy_res5, rtol = 1e-7)
@test isapprox(res, easy_res6, rtol = 1e-9)
@test isapprox(res, easy_res7, rtol = 1e-9)

function G(p)
  tmp_prob = remake(prob,u0=convert.(eltype(p),prob.u0),p=p)
  sol = solve(tmp_prob,Tsit5(),abstol=1e-14,reltol=1e-14,saveat=t)
  sum(((2 .- Array(sol)).^2)./2)
end
G([1.5,3.0])
res2 = ForwardDiff.gradient(G,[1.5,3.0])
res3 = Calculus.gradient(G,[1.5,3.0])

@test norm(res' .- res2) < 1e-8
@test norm(res' .- res3) < 1e-6
