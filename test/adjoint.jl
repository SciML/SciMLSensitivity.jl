using DiffEqSensitivity, OrdinaryDiffEq, RecursiveArrayTools, DiffEqBase,
      ForwardDiff, Calculus, QuadGK, LinearAlgebra, Zygote
using Test

function fb(du,u,p,t)
  du[1] = dx = p[1]*u[1] - p[2]*u[1]*u[2]
  du[2] = dy = -p[3]*u[2] + p[4]*u[1]*u[2]
end
function foop(u,p,t)
  dx = p[1]*u[1] - p[2]*u[1]*u[2]
  dy = -p[3]*u[2] + p[4]*u[1]*u[2]
  [dx,dy]
end
function jac(J,u,p,t)
  (x, y, a, b, c) = (u[1], u[2], p[1], p[2], p[3])
  J[1,1] = a + y * b * -1
  J[2,1] = y
  J[1,2] = b * x * -1
  J[2,2] = c * -1 + x
end

f = ODEFunction(fb,jac=jac)
p = [1.5,1.0,3.0,1.0]; u0 = [1.0;1.0]
prob = ODEProblem(f,u0,(0.0,10.0),p)
sol = solve(prob,Tsit5(),abstol=1e-14,reltol=1e-14)
probb = ODEProblem(fb,u0,(0.0,10.0),p)
proboop = ODEProblem(foop,u0,(0.0,10.0),p)

solb = solve(probb,Tsit5(),abstol=1e-14,reltol=1e-14)
sol_end = solve(probb,Tsit5(),abstol=1e-14,reltol=1e-14,
          save_everystep=false,save_start=false)

sol_nodense = solve(probb,Tsit5(),abstol=1e-14,reltol=1e-14,dense=false)
soloop = solve(proboop,Tsit5(),abstol=1e-14,reltol=1e-14)
soloop_nodense = solve(proboop,Tsit5(),abstol=1e-14,reltol=1e-14,dense=false)

_p = copy(p)
function foop_zygote(u,p,t)
  dx = _p[1]*u[1] - _p[2]*u[1]*u[2]
  dy = -_p[3]*u[2] + _p[4]*u[1]*u[2]
  [dx,dy]
end
pp = Zygote.Params([_p])
prob_zygote = ODEProblem(foop_zygote,u0,(0.0,10.0),pp)
soloop_zygote = solve(prob_zygote,Tsit5(),abstol=1e-14,reltol=1e-14)

# Do a discrete adjoint problem
println("Calculate discrete adjoint sensitivities")
t = 0.0:0.5:10.0
# g(t,u,i) = (1-u)^2/2, L2 away from 1
function dg(out,u,p,t,i)
  (out.=2.0.-u)
end

easy_res = adjoint_sensitivities(sol,Tsit5(),dg,t,abstol=1e-14,
                                 reltol=1e-14,iabstol=1e-14,ireltol=1e-12)
easy_res2 = adjoint_sensitivities(solb,Tsit5(),dg,t,abstol=1e-14,
                                 reltol=1e-14,iabstol=1e-14,ireltol=1e-12,
                                 sensealg=QuadratureAdjoint())
easy_res22 = adjoint_sensitivities(solb,Tsit5(),dg,t,abstol=1e-14,
                                  reltol=1e-14,iabstol=1e-14,ireltol=1e-12,
                                  sensealg=QuadratureAdjoint(autojacvec=false))
easy_res3 = adjoint_sensitivities(solb,Tsit5(),dg,t,abstol=1e-14,
                                  reltol=1e-14,
                                  sensealg=InterpolatingAdjoint())
easy_res32 = adjoint_sensitivities(solb,Tsit5(),dg,t,abstol=1e-14,
                                  reltol=1e-14,
                                  sensealg=InterpolatingAdjoint(autojacvec=false))
easy_res4 = adjoint_sensitivities(solb,Tsit5(),dg,t,abstol=1e-14,
                                  reltol=1e-14,
                                  sensealg=BacksolveAdjoint())
easy_res42 = adjoint_sensitivities(solb,Tsit5(),dg,t,abstol=1e-14,
                                   reltol=1e-14,
                                   sensealg=BacksolveAdjoint(autojacvec=false))
easy_res5 = adjoint_sensitivities(sol,Kvaerno5(nlsolve=NLAnderson(), smooth_est=false),
                                 dg,t,abstol=1e-12,
                                 reltol=1e-10,
                                 sensealg=BacksolveAdjoint())
easy_res6 = adjoint_sensitivities(sol_nodense,Tsit5(),dg,t,abstol=1e-14,
                                  reltol=1e-14,
                                  sensealg=InterpolatingAdjoint(checkpointing=true),
                                  checkpoints=sol.t[1:5:end])
easy_res62 = adjoint_sensitivities(sol_nodense,Tsit5(),dg,t,abstol=1e-14,
                                   reltol=1e-14,
                                   sensealg=InterpolatingAdjoint(checkpointing=true,autojacvec=false),
                                   checkpoints=sol.t[1:5:end])

# It should automatically be checkpointing since the solution isn't dense
easy_res7 = adjoint_sensitivities(sol_nodense,Tsit5(),dg,t,abstol=1e-14,
                                  reltol=1e-14,
                                  sensealg=InterpolatingAdjoint(),
                                  checkpoints=sol.t[1:5:end])

adj_prob = ODEAdjointProblem(sol,QuadratureAdjoint(),dg,t)
adj_sol = solve(adj_prob,Tsit5(),abstol=1e-14,reltol=1e-14)
integrand = AdjointSensitivityIntegrand(sol,adj_sol,QuadratureAdjoint())
res,err = quadgk(integrand,0.0,10.0,atol=1e-14,rtol=1e-12)

@test isapprox(res, easy_res, rtol = 1e-10)
@test isapprox(res, easy_res2, rtol = 1e-10)
@test isapprox(res, easy_res22, rtol = 1e-10)
@test isapprox(res, easy_res3, rtol = 1e-10)
@test isapprox(res, easy_res32, rtol = 1e-10)
@test isapprox(res, easy_res4, rtol = 1e-10)
@test isapprox(res, easy_res42, rtol = 1e-10)
@test isapprox(res, easy_res5, rtol = 1e-7)
@test isapprox(res, easy_res6, rtol = 1e-9)
@test isapprox(res, easy_res62, rtol = 1e-9)
@test all(easy_res6 .== easy_res7)  # should be the same!

println("OOP adjoint sensitivities ")

easy_res = adjoint_sensitivities(soloop,Tsit5(),dg,t,abstol=1e-14,
                                 reltol=1e-14,iabstol=1e-14,ireltol=1e-12)
easy_res2 = adjoint_sensitivities(soloop,Tsit5(),dg,t,abstol=1e-14,
                                 reltol=1e-14,iabstol=1e-14,ireltol=1e-12,
                                 sensealg=QuadratureAdjoint())
@test_broken easy_res22 = adjoint_sensitivities(soloop,Tsit5(),dg,t,abstol=1e-14,
                                  reltol=1e-14,iabstol=1e-14,ireltol=1e-12,
                                  sensealg=QuadratureAdjoint(autojacvec=false)) isa AbstractArray
easy_res3 = adjoint_sensitivities(soloop,Tsit5(),dg,t,abstol=1e-14,
                                  reltol=1e-14,
                                  sensealg=InterpolatingAdjoint())
@test_broken easy_res32 = adjoint_sensitivities(soloop,Tsit5(),dg,t,abstol=1e-14,
                                  reltol=1e-14,
                                  sensealg=InterpolatingAdjoint(autojacvec=false)) isa AbstractArray
easy_res4 = adjoint_sensitivities(soloop,Tsit5(),dg,t,abstol=1e-14,
                                  reltol=1e-14,
                                  sensealg=BacksolveAdjoint())
@test_broken easy_res42 = adjoint_sensitivities(soloop,Tsit5(),dg,t,abstol=1e-14,
                                   reltol=1e-14,
                                   sensealg=BacksolveAdjoint(autojacvec=false)) isa AbstractArray
easy_res5 = adjoint_sensitivities(soloop,Kvaerno5(nlsolve=NLAnderson(), smooth_est=false),
                                 dg,t,abstol=1e-12,
                                 reltol=1e-10,
                                 sensealg=BacksolveAdjoint())
easy_res6 = adjoint_sensitivities(soloop_nodense,Tsit5(),dg,t,abstol=1e-14,
                                  reltol=1e-14,
                                  sensealg=InterpolatingAdjoint(checkpointing=true),
                                  checkpoints=soloop_nodense.t[1:5:end])
@test_broken easy_res62 = adjoint_sensitivities(soloop_nodense,Tsit5(),dg,t,abstol=1e-14,
                                   reltol=1e-14,
                                   sensealg=InterpolatingAdjoint(checkpointing=true,autojacvec=false),
                                   checkpoints=soloop_nodense.t[1:5:end])

@test isapprox(res, easy_res, rtol = 1e-10)
@test isapprox(res, easy_res2, rtol = 1e-10)
@test isapprox(res, easy_res22, rtol = 1e-10)
@test isapprox(res, easy_res3, rtol = 1e-10)
@test isapprox(res, easy_res32, rtol = 1e-10)
@test isapprox(res, easy_res4, rtol = 1e-10)
@test isapprox(res, easy_res42, rtol = 1e-10)
@test isapprox(res, easy_res5, rtol = 1e-9)
@test isapprox(res, easy_res6, rtol = 1e-10)
@test isapprox(res, easy_res62, rtol = 1e-9)

println("Calculate adjoint sensitivities ")

easy_res8 = adjoint_sensitivities(solb,Tsit5(),dg,t,abstol=1e-14,
                                  reltol=1e-14,iabstol=1e-14,ireltol=1e-12,
                                  save_everystep=false,save_start=false,
                                  sensealg=BacksolveAdjoint())

@test isapprox(res, easy_res8, rtol = 1e-9)

end_only_res = adjoint_sensitivities(sol_end,Tsit5(),dg,t,abstol=1e-14,
                                  reltol=1e-14,iabstol=1e-14,ireltol=1e-12,
                                  save_everystep=false,save_start=false,
                                  sensealg=BacksolveAdjoint())

@test isapprox(res, end_only_res, rtol = 1e-9)

println("Calculate adjoint sensitivities from autodiff & numerical diff")
function G(p)
  tmp_prob = remake(prob,u0=convert.(eltype(p),prob.u0),p=p)
  sol = solve(tmp_prob,Tsit5(),abstol=1e-10,reltol=1e-10,saveat=t)
  A = convert(Array,sol)
  sum(((2 .- A).^2)./2)
end
G([1.5,1.0,3.0,1.0])
res2 = ForwardDiff.gradient(G,[1.5,1.0,3.0,1.0])
res3 = Calculus.gradient(G,[1.5,1.0,3.0,1.0])

import Tracker
res4 = Tracker.gradient(G,[1.5,1.0,3.0,1.0])[1]

import ReverseDiff
res5 = ReverseDiff.gradient(G,[1.5,1.0,3.0,1.0])

@test norm(res' .- res2) < 1e-8
@test norm(res' .- res3) < 1e-5
@test norm(res' .- res4) < 1e-6
@test norm(res' .- res5) < 1e-6

# check other t handling

t2 = [0.5, 1.0]
t3 = [0.0, 0.5, 1.0]
t4 = [0.5, 1.0, 10.0]

easy_res2 = adjoint_sensitivities(sol,Tsit5(),dg,t2,abstol=1e-14,
                                  reltol=1e-14,iabstol=1e-14,ireltol=1e-12)
easy_res3 = adjoint_sensitivities(sol,Tsit5(),dg,t3,abstol=1e-14,
                                  reltol=1e-14,iabstol=1e-14,ireltol=1e-12)
easy_res4 = adjoint_sensitivities(sol,Tsit5(),dg,t4,abstol=1e-14,
                                  reltol=1e-14,iabstol=1e-14,ireltol=1e-12)

function G(p,ts)
  tmp_prob = remake(prob,u0=convert.(eltype(p),prob.u0),p=p)
  sol = solve(tmp_prob,Tsit5(),abstol=1e-10,reltol=1e-10,saveat=ts)
  A = convert(Array,sol)
  sum(((2 .- A).^2)./2)
end
res2 = ForwardDiff.gradient(p->G(p,t2),[1.5,1.0,3.0,1.0])
res3 = ForwardDiff.gradient(p->G(p,t3),[1.5,1.0,3.0,1.0])
res4 = ForwardDiff.gradient(p->G(p,t4),[1.5,1.0,3.0,1.0])

@test easy_res2' ≈ res2
@test easy_res3' ≈ res3
@test easy_res4' ≈ res4

println("Adjoints of u0")

function dg(out,u,p,t,i)
  out .= 1 .- u
end

ū0,adj = adjoint_sensitivities_u0(sol,Tsit5(),dg,t,abstol=1e-14,
                         reltol=1e-14,iabstol=1e-14,ireltol=1e-12)

adjnou0 = adjoint_sensitivities(sol,Tsit5(),dg,t,abstol=1e-14,
                        reltol=1e-14,iabstol=1e-14,ireltol=1e-12)

ū02,adj2 = adjoint_sensitivities_u0(sol,Tsit5(),dg,t,abstol=1e-14,
                                    sensealg=BacksolveAdjoint(),
                                    reltol=1e-14,iabstol=1e-14,ireltol=1e-12)

ū022,adj22 = adjoint_sensitivities_u0(sol,Tsit5(),dg,t,abstol=1e-14,
                                    sensealg=BacksolveAdjoint(autojacvec=false),
                                    reltol=1e-14,iabstol=1e-14,ireltol=1e-12)

ū03,adj3 = adjoint_sensitivities_u0(sol,Tsit5(),dg,t,abstol=1e-14,
                                    sensealg=InterpolatingAdjoint(),
                                    reltol=1e-14,iabstol=1e-14,ireltol=1e-12)

ū032,adj32 = adjoint_sensitivities_u0(sol,Tsit5(),dg,t,abstol=1e-14,
                                    sensealg=InterpolatingAdjoint(autojacvec=false),
                                    reltol=1e-14,iabstol=1e-14,ireltol=1e-12)

ū04,adj4 = adjoint_sensitivities_u0(sol,Tsit5(),dg,t,abstol=1e-14,
                                    sensealg=InterpolatingAdjoint(checkpointing=true),
                                    checkpoints=sol.t[1:10:end],
                                    reltol=1e-14,iabstol=1e-14,ireltol=1e-12)

@test_throws Any adjoint_sensitivities_u0(sol,Tsit5(),dg,t,abstol=1e-14,
                         sensealg=InterpolatingAdjoint(checkpointing=true),
                         checkpoints=sol.t[1:5:end],
                         reltol=1e-14,iabstol=1e-14,ireltol=1e-12)

ū042,adj42 = adjoint_sensitivities_u0(sol,Tsit5(),dg,t,abstol=1e-14,
                                    sensealg=InterpolatingAdjoint(checkpointing=true,autojacvec=false),
                                    checkpoints=sol.t[1:10:end],
                                    reltol=1e-14,iabstol=1e-14,ireltol=1e-12)

ū0args,adjargs = adjoint_sensitivities_u0(sol,Tsit5(),dg,t,abstol=1e-14,
                        save_everystep=false, save_start=false,
                        sensealg=BacksolveAdjoint(),
                        reltol=1e-14,iabstol=1e-14,ireltol=1e-12)

ū0args2,adjargs2 = adjoint_sensitivities_u0(sol,Tsit5(),dg,t,abstol=1e-14,
                        save_everystep=false, save_start=false,
                        sensealg=InterpolatingAdjoint(),
                        reltol=1e-14,iabstol=1e-14,ireltol=1e-12)

res = ForwardDiff.gradient(prob.u0) do u0
  tmp_prob = remake(prob,u0=u0)
  sol = solve(tmp_prob,Tsit5(),abstol=1e-14,reltol=1e-14,saveat=t)
  A = convert(Array,sol)
  sum(((1 .- A).^2)./2)
end

@test ū0 ≈ res rtol = 1e-10
@test ū02 ≈ res rtol = 1e-10
@test ū022 ≈ res rtol = 1e-10
@test ū03 ≈ res rtol = 1e-10
@test ū032 ≈ res rtol = 1e-10
@test ū04 ≈ res rtol = 1e-10
@test ū042 ≈ res rtol = 1e-10
@test adj ≈ adjnou0 rtol = 1e-10
@test adj ≈ adj2 rtol = 1e-10
@test adj ≈ adj22 rtol = 1e-10
@test adj ≈ adj3 rtol = 1e-10
@test adj ≈ adj32 rtol = 1e-10
@test adj ≈ adj4 rtol = 1e-10
@test adj ≈ adj42 rtol = 1e-10

@test ū0args ≈ res rtol = 1e-10
@test adjargs ≈ adj rtol = 1e-10
@test ū0args2 ≈ res rtol = 1e-10
@test adjargs2 ≈ adj rtol = 1e-10

println("Zygote OOP adjoint sensitivities ")

zy_ū0, zy_adj = adjoint_sensitivities_u0(soloop_zygote,Tsit5(),dg,t,
                                         abstol=1e-10,reltol=1e-10)

zy_ū02, zy_adj2 = adjoint_sensitivities_u0(soloop_zygote,Tsit5(),dg,t,
                                           abstol=1e-10,reltol=1e-10,
                                           sensealg=BacksolveAdjoint())

@test zy_ū0 ≈ res rtol = 1e-8
@test zy_ū02 ≈ res rtol = 1e-8
@test zy_adj ≈ adjnou0 rtol = 1e-8
@test zy_adj2 ≈ adjnou0 rtol = 1e-8

println("Do a continuous adjoint problem")

# Energy calculation
g(u,p,t) = (sum(u).^2) ./ 2
# Gradient of (u1 + u2)^2 / 2
function dg(out,u,p,t)
  out[1]= u[1] + u[2]
  out[2]= u[1] + u[2]
end

adj_prob = ODEAdjointProblem(sol,QuadratureAdjoint(),g,nothing,dg)
adj_sol = solve(adj_prob,Tsit5(),abstol=1e-14,reltol=1e-10)
integrand = AdjointSensitivityIntegrand(sol,adj_sol,QuadratureAdjoint())
res,err = quadgk(integrand,0.0,10.0,atol=1e-14,rtol=1e-10)

println("Test the `adjoint_sensitivities` utility function")
easy_res = adjoint_sensitivities(sol,Tsit5(),g,nothing,dg,abstol=1e-14,
                                 reltol=1e-14,iabstol=1e-14,ireltol=1e-12)
println("2")
easy_res2 = adjoint_sensitivities(sol,Tsit5(),g,nothing,dg,abstol=1e-14,
                                  reltol=1e-14,iabstol=1e-14,ireltol=1e-12,
                                  sensealg=InterpolatingAdjoint())
easy_res22 = adjoint_sensitivities(sol,Tsit5(),g,nothing,dg,abstol=1e-14,
                                   reltol=1e-14,iabstol=1e-14,ireltol=1e-12,
                                   sensealg=InterpolatingAdjoint(autojacvec=false))
println("23")
easy_res23 = adjoint_sensitivities(sol,Tsit5(),g,nothing,dg,abstol=1e-14,
                                   reltol=1e-14,iabstol=1e-14,ireltol=1e-12,
                                   sensealg=QuadratureAdjoint())
easy_res24 = adjoint_sensitivities(sol,Tsit5(),g,nothing,dg,abstol=1e-14,
                                   reltol=1e-14,iabstol=1e-14,ireltol=1e-12,
                                   sensealg=QuadratureAdjoint(autojacvec=false))
println("25")
easy_res25 = adjoint_sensitivities(sol,Tsit5(),g,nothing,dg,abstol=1e-14,
                                   reltol=1e-14,iabstol=1e-14,ireltol=1e-12,
                                   sensealg=BacksolveAdjoint())
easy_res26 = adjoint_sensitivities(sol,Tsit5(),g,nothing,dg,abstol=1e-14,
                                   reltol=1e-14,iabstol=1e-14,ireltol=1e-12,
                                   sensealg=BacksolveAdjoint(autojacvec=false))
println("27")
easy_res27 = adjoint_sensitivities(sol,Tsit5(),g,nothing,dg,abstol=1e-14,
                                  reltol=1e-14,iabstol=1e-14,ireltol=1e-12,
                                  checkpoints=sol.t[1:10:end],
                                  sensealg=InterpolatingAdjoint(checkpointing=true))
easy_res28 = adjoint_sensitivities(sol,Tsit5(),g,nothing,dg,abstol=1e-14,
                                  reltol=1e-14,iabstol=1e-14,ireltol=1e-12,
                                  checkpoints=sol.t[1:10:end],
                                  sensealg=InterpolatingAdjoint(checkpointing=true,autojacvec=false))
println("3")
easy_res3 = adjoint_sensitivities(sol,Tsit5(),g,nothing,abstol=1e-14,
                                  reltol=1e-14,iabstol=1e-14,ireltol=1e-12,
                                  sensealg=InterpolatingAdjoint())
easy_res32 = adjoint_sensitivities(sol,Tsit5(),g,nothing,abstol=1e-14,
                                   reltol=1e-14,iabstol=1e-14,ireltol=1e-12,
                                   sensealg=InterpolatingAdjoint(autojacvec=false))
println("33")
easy_res33 = adjoint_sensitivities(sol,Tsit5(),g,nothing,abstol=1e-14,
                                   reltol=1e-14,iabstol=1e-14,ireltol=1e-12,
                                   sensealg=QuadratureAdjoint())
easy_res34 = adjoint_sensitivities(sol,Tsit5(),g,nothing,abstol=1e-14,
                                   reltol=1e-14,iabstol=1e-14,ireltol=1e-12,
                                   sensealg=QuadratureAdjoint(autojacvec=false))
println("35")
easy_res35 = adjoint_sensitivities(sol,Tsit5(),g,nothing,abstol=1e-14,
                                   reltol=1e-14,iabstol=1e-14,ireltol=1e-12,
                                   sensealg=BacksolveAdjoint())
easy_res36 = adjoint_sensitivities(sol,Tsit5(),g,nothing,abstol=1e-14,
                                   reltol=1e-14,iabstol=1e-14,ireltol=1e-12,
                                   sensealg=BacksolveAdjoint(autojacvec=false))
println("37")
easy_res37 = adjoint_sensitivities(sol,Tsit5(),g,nothing,abstol=1e-14,
                                  reltol=1e-14,iabstol=1e-14,ireltol=1e-12,
                                  checkpoints=sol.t[1:10:end],
                                  sensealg=InterpolatingAdjoint(checkpointing=true))
easy_res38 = adjoint_sensitivities(sol,Tsit5(),g,nothing,abstol=1e-14,
                                  reltol=1e-14,iabstol=1e-14,ireltol=1e-12,
                                  checkpoints=sol.t[1:10:end],
                                  sensealg=InterpolatingAdjoint(checkpointing=true,autojacvec=false))

@test norm(easy_res .- res) < 1e-8
@test norm(easy_res2 .- res) < 1e-8
@test norm(easy_res22 .- res) < 1e-8
@test norm(easy_res23 .- res) < 1e-8
@test norm(easy_res24 .- res) < 1e-8
@test norm(easy_res25 .- res) < 1e-8
@test norm(easy_res26 .- res) < 1e-8
@test norm(easy_res27 .- res) < 1e-8
@test norm(easy_res28 .- res) < 1e-8
@test norm(easy_res3 .- res) < 1e-8
@test norm(easy_res32 .- res) < 1e-8
@test norm(easy_res33 .- res) < 1e-8
@test norm(easy_res34 .- res) < 1e-8
@test norm(easy_res35 .- res) < 1e-8
@test norm(easy_res36 .- res) < 1e-8
@test norm(easy_res37 .- res) < 1e-8
@test norm(easy_res38 .- res) < 1e-8

println("Calculate adjoint sensitivities from autodiff & numerical diff")
function G(p)
  tmp_prob = remake(prob,u0=eltype(p).(prob.u0),p=p,
                    tspan=eltype(p).(prob.tspan))
  sol = solve(tmp_prob,Tsit5(),abstol=1e-14,reltol=1e-14)
  res,err = quadgk((t)-> (sum(sol(t)).^2)./2,0.0,10.0,atol=1e-14,rtol=1e-10)
  res
end
res2 = ForwardDiff.gradient(G,[1.5,1.0,3.0,1.0])
res3 = Calculus.gradient(G,[1.5,1.0,3.0,1.0])

@test norm(res' .- res2) < 1e-8
@test norm(res' .- res3) < 1e-6

# Buffer length test
f = (du, u, p, t) -> du .= 0
p = zeros(3); u = zeros(50)
prob = ODEProblem(f,u,(0.0,10.0),p)
sol = solve(prob,Tsit5(),abstol=1e-14,reltol=1e-14)
@test_nowarn res = adjoint_sensitivities(sol,Tsit5(),dg,t,abstol=1e-14,
                                 reltol=1e-14,iabstol=1e-14,ireltol=1e-12)
