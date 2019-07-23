using DiffEqSensitivity,OrdinaryDiffEq, ParameterizedFunctions,
      RecursiveArrayTools, DiffEqBase, ForwardDiff, Calculus, QuadGK,
      LinearAlgebra, Flux, ReverseDiff
using Test

fb = @ode_def_bare begin
  dx = a*x - b*x*y
  dy = -c*y + x*y
end a b c
f = @ode_def begin
  dx = a*x - b*x*y
  dy = -c*y + x*y
end a b c

p = [1.5,1.0,3.0]
prob = ODEProblem(f,[1.0;1.0],(0.0,10.0),p)
sol = solve(prob,Tsit5(),abstol=1e-14,reltol=1e-14)
probb = ODEProblem(fb,[1.0;1.0],(0.0,10.0),p)
solb = solve(probb,Tsit5(),abstol=1e-14,reltol=1e-14)
sol_end = solve(probb,Tsit5(),abstol=1e-14,reltol=1e-14,
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
easy_res2 = adjoint_sensitivities(solb,Tsit5(),dg,t,abstol=1e-14,
                                 reltol=1e-14,iabstol=1e-14,ireltol=1e-12,sensealg=SensitivityAlg(quad=true,backsolve=false))
easy_res3 = adjoint_sensitivities(solb,Tsit5(),dg,t,abstol=1e-14,
                                  reltol=1e-14,iabstol=1e-14,ireltol=1e-12,sensealg=SensitivityAlg(quad=false,backsolve=false))
easy_res4 = adjoint_sensitivities(solb,Tsit5(),dg,t,abstol=1e-14,
                                  reltol=1e-14,iabstol=1e-14,ireltol=1e-12,sensealg=SensitivityAlg(backsolve=true))
easy_res5 = adjoint_sensitivities(sol,Kvaerno5(nlsolve=NLAnderson(), smooth_est=false),dg,t,abstol=1e-12,
                                 reltol=1e-10,iabstol=1e-14,ireltol=1e-12,sensealg=SensitivityAlg(backsolve=true))
easy_res6 = adjoint_sensitivities(solb,Tsit5(),dg,t,abstol=1e-14,
                                  reltol=1e-14,iabstol=1e-14,ireltol=1e-12,
                                  sensealg=SensitivityAlg(checkpointing=true,quad=true),
                                  checkpoints=sol.t[1:5:end])
easy_res7 = adjoint_sensitivities(solb,Tsit5(),dg,t,abstol=1e-14,
                                  reltol=1e-14,iabstol=1e-14,ireltol=1e-12,
                                  sensealg=SensitivityAlg(checkpointing=true,quad=false),
                                  checkpoints=sol.t[1:5:end])
easy_res8 = adjoint_sensitivities(solb,Tsit5(),dg,t,abstol=1e-14,
                                  reltol=1e-14,iabstol=1e-14,ireltol=1e-12,
                                  sensealg=SensitivityAlg(quad=true,autojacvec=false))
easy_res9 = adjoint_sensitivities(sol, Tsit5(),dg,t,abstol=1e-14,
                                  reltol=1e-14,iabstol=1e-14,ireltol=1e-12,
                                  sensealg=SensitivityAlg(quad=true,autojacvec=false))

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
@test isapprox(res, easy_res8, rtol = 1e-9)
@test isapprox(res, easy_res9, rtol = 1e-9)

println("Calculate adjoint sensitivities ")

easy_res8 = adjoint_sensitivities(solb,Tsit5(),dg,t,abstol=1e-14,
                                  reltol=1e-14,iabstol=1e-14,ireltol=1e-12,
                                  save_everystep=false,save_start=false,
                                  sensealg=SensitivityAlg(backsolve=true))

@test isapprox(res, easy_res8, rtol = 1e-9)

end_only_res = adjoint_sensitivities(sol_end,Tsit5(),dg,t,abstol=1e-14,
                                  reltol=1e-14,iabstol=1e-14,ireltol=1e-12,
                                  save_everystep=false,save_start=false,
                                  sensealg=SensitivityAlg(backsolve=true))

@test isapprox(res, end_only_res, rtol = 1e-9)

println("Calculate adjoint sensitivities from autodiff & numerical diff")
function G(p)
  tmp_prob = remake(prob,u0=convert.(eltype(p),prob.u0),p=p)
  sol = solve(tmp_prob,Tsit5(),abstol=1e-14,reltol=1e-14,saveat=t)
  A = convert(Array,sol)
  sum(((2 .- A).^2)./2)
end
G([1.5,1.0,3.0])
res2 = ForwardDiff.gradient(G,[1.5,1.0,3.0])
res3 = Calculus.gradient(G,[1.5,1.0,3.0])
res4 = Flux.Tracker.gradient(G,[1.5,1.0,3.0])[1]
res5 = ReverseDiff.gradient(G,[1.5,1.0,3.0])

@test norm(res' .- res2) < 1e-8
@test norm(res' .- res3) < 1e-6
@test norm(res' .- res4) < 1e-6
@test norm(res' .- res5) < 1e-6

# Do a continuous adjoint problem

# Energy calculation
g(u,p,t) = (sum(u).^2) ./ 2
# Gradient of (u1 + u2)^2 / 2
function dg(out,u,p,t)
  out[1]= u[1] + u[2]
  out[2]= u[1] + u[2]
end

adj_prob = ODEAdjointProblem(sol,g,nothing,dg)
adj_sol = solve(adj_prob,Tsit5(),abstol=1e-14,reltol=1e-10)
integrand = AdjointSensitivityIntegrand(sol,adj_sol)
res,err = quadgk(integrand,0.0,10.0,atol=1e-14,rtol=1e-10)

println("Test the `adjoint_sensitivities` utility function")
easy_res = adjoint_sensitivities(sol,Tsit5(),g,nothing,dg,abstol=1e-14,
                                 reltol=1e-14,iabstol=1e-14,ireltol=1e-12)
println("2")
easy_res2 = adjoint_sensitivities(sol,Tsit5(),g,nothing,dg,abstol=1e-14,
                                  reltol=1e-14,iabstol=1e-14,ireltol=1e-12,
                                  sensealg=SensitivityAlg(quad=false))
println("3")
easy_res3 = adjoint_sensitivities(sol,Tsit5(),g,nothing,abstol=1e-14,
                                 reltol=1e-14,iabstol=1e-14,ireltol=1e-12)
println("4")
easy_res4 = adjoint_sensitivities(sol,Tsit5(),g,nothing,abstol=1e-14,
                                  reltol=1e-14,iabstol=1e-14,ireltol=1e-12,
                                  sensealg=SensitivityAlg(autodiff=false))
println("5")
easy_res5 = adjoint_sensitivities(sol,Tsit5(),g,nothing,abstol=1e-14,
                                  reltol=1e-14,iabstol=1e-14,ireltol=1e-12,
                                  sensealg=SensitivityAlg(checkpointing=true))
@test norm(easy_res .- res) < 1e-8
@test norm(easy_res2 .- res) < 1e-8
@test norm(easy_res3 .- res) < 1e-8
@test norm(easy_res4 .- res) < 1e-8
@test norm(easy_res5 .- res) < 1e-8

println("Calculate adjoint sensitivities from autodiff & numerical diff")
function G(p)
  tmp_prob = remake(prob,u0=eltype(p).(prob.u0),p=p,
                    tspan=eltype(p).(prob.tspan))
  sol = solve(tmp_prob,Tsit5(),abstol=1e-14,reltol=1e-14)
  res,err = quadgk((t)-> (sum(sol(t)).^2)./2,0.0,10.0,atol=1e-14,rtol=1e-10)
  res
end
res2 = ForwardDiff.gradient(G,[1.5,1.0,3.0])
res3 = Calculus.gradient(G,[1.5,1.0,3.0])

@test norm(res' .- res2) < 1e-8
@test norm(res' .- res3) < 1e-6

# Buffer length test
f = (du, u, p, t) -> du .= 0
p = zeros(3); u = zeros(50)
prob = ODEProblem(f,u,(0.0,10.0),p)
sol = solve(prob,Tsit5(),abstol=1e-14,reltol=1e-14)
@test_nowarn res = adjoint_sensitivities(sol,Tsit5(),dg,t,abstol=1e-14,
                                 reltol=1e-14,iabstol=1e-14,ireltol=1e-12)

println("Adjoints of u0")

function dg(out,u,p,t,i)
  out .= 1 .- u
end

ū0,adj = adjoint_sensitivities_u0(sol,Tsit5(),dg,t,abstol=1e-14,
                         reltol=1e-14,iabstol=1e-14,ireltol=1e-12)

adj2 = adjoint_sensitivities(sol,Tsit5(),dg,t,abstol=1e-14,
                        reltol=1e-14,iabstol=1e-14,ireltol=1e-12)

ū02,adj3 = adjoint_sensitivities_u0(sol,Tsit5(),dg,t,abstol=1e-14,
                         sensealg=SensitivityAlg(backsolve=true),
                         reltol=1e-14,iabstol=1e-14,ireltol=1e-12)

ū03,adj4 = adjoint_sensitivities_u0(sol,Tsit5(),dg,t,abstol=1e-14,
                        save_everystep=false, save_start=false,
                        sensealg=SensitivityAlg(backsolve=true),
                        reltol=1e-14,iabstol=1e-14,ireltol=1e-12)

ū03,adj4 = adjoint_sensitivities_u0(sol,Tsit5(),dg,t,abstol=1e-14,
                        save_everystep=false, save_start=false,
                        sensealg=SensitivityAlg(backsolve=true),
                        reltol=1e-14,iabstol=1e-14,ireltol=1e-12)

res = ForwardDiff.gradient(prob.u0) do u0
  tmp_prob = remake(prob,u0=u0)
  sol = solve(tmp_prob,Tsit5(),abstol=1e-14,reltol=1e-14,saveat=t)
  A = convert(Array,sol)
  sum(((1 .- A).^2)./2)
end

@test ū0 ≈ res rtol = 1e-10
@test ū02 ≈ res rtol = 1e-10
@test ū03 ≈ res rtol = 1e-10
@test adj ≈ adj2 rtol = 1e-10
@test adj ≈ adj3 rtol = 1e-10
@test adj ≈ adj4 rtol = 1e-10

function ff(du,u,p,t)
  du[1] = dx = p[1]*u[1] - p[2]*u[1]*u[2]
  du[2] = dy = -p[3]*u[2] + u[1]*u[2]
end

p = [1.5,1.0,3.0]
prob = ODEProblem(ff,[1.0;1.0],(0.0,10.0),p)
sol = solve(prob,Tsit5(),abstol=1e-10,reltol=1e-10)

function dg(out,u,p,t,i)
  out .= 1.0 .- u
end
t = collect(0.5:0.5:10.0)

res = adjoint_sensitivities(sol,Tsit5(),dg,t;abstol=1e-14,reltol=1e-14,
                            iabstol=1e-14,ireltol=1e-12)

function GG(p)
    tmp_prob = remake(prob,u0=convert.(eltype(p),prob.u0),p=p)
    sol_i = solve(tmp_prob,Vern9(),abstol=1e-14,reltol=1e-14,saveat=t)
    A = convert(Array,sol_i)
    sum(((1.0 .- A).^2)./2)
end
GG(p)
res2 = ForwardDiff.gradient(GG,p)
res3 = Calculus.gradient(GG,p)
res4 = Flux.Tracker.gradient(GG,p)[1]
res5 = ReverseDiff.gradient(GG,p)

@test norm(res2 .- res') < 1e-8
@test norm(res3 .- res') < 1e-6
@test norm(res4 .- res') < 1e-7
@test norm(res5 .- res') < 1e-8
