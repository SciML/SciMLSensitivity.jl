using DiffEqSensitivity,OrdinaryDiffEq, ParameterizedFunctions,
      RecursiveArrayTools, DiffEqBase, ForwardDiff, Calculus, QuadGK
using Base.Test

f = @ode_def_nohes LotkaVolterra begin
  dx = a*x - b*x*y
  dy = -c*y + d*x*y
end a=>1.5 b=>1.0 c=>3.0 d=1

prob = ODEProblem(f,[1.0;1.0],(0.0,10.0))
sol = solve(prob,Vern9(),abstol=1e-14,reltol=1e-14)

# Do a discrete adjoint problem

t = 0.0:0.5:10.0 # TODO: Add end point handling for callback
# g(t,u,i) = (1-u)^2/2, L2 away from 1
dg(out,u,i) = (out.=1.0.-u)

easy_res = adjoint_sensitivities(sol,Vern9(),dg,t,abstol=1e-14,
                                 reltol=1e-14,iabstol=1e-14,ireltol=1e-12)

adj_prob = ODEAdjointProblem(sol,dg,t)
adj_sol = solve(adj_prob,Vern9(),abstol=1e-14,reltol=1e-14)
integrand = AdjointSensitivityIntegrand(sol,adj_sol)
res,err = quadgk(integrand,0.0,10.0,abstol=1e-14,reltol=1e-12)

@test norm(res - easy_res) < 1e-10

function G(p)
  tmp_prob = problem_new_parameters(prob,p)
  sol = solve(tmp_prob,Vern9(),abstol=1e-14,reltol=1e-14,saveat=t)
  A = convert(Array,sol)
  sum(((1-A).^2)./2)
end
G([1.5,1.0,3.0])
res2 = ForwardDiff.gradient(G,[1.5,1.0,3.0])
res3 = Calculus.gradient(G,[1.5,1.0,3.0])

@test norm(res' - res2) < 1e-8
@test norm(res' - res3) < 1e-6

# Do a continuous adjoint problem

# Energy calculation
g(t,u,p) = (sum(u).^2) ./ 2
# Gradient of (u1 + u2)^2 / 2
function dg(out,t,u,p)
  out[1]= u[1] + u[2]
  out[2]= u[1] + u[2]
end

adj_prob = ODEAdjointProblem(sol,g,nothing,dg)
adj_sol = solve(adj_prob,Vern9(),abstol=1e-14,reltol=1e-10)
integrand = AdjointSensitivityIntegrand(sol,adj_sol)
res,err = quadgk(integrand,0.0,10.0,abstol=1e-14,reltol=1e-10)

easy_res = adjoint_sensitivities(sol,Vern9(),g,nothing,dg,abstol=1e-14,
                                 reltol=1e-14,iabstol=1e-14,ireltol=1e-12)

@test norm(easy_res - res) < 1e-8

function G(p)
  tmp_prob = problem_new_parameters(prob,p)
  sol = solve(tmp_prob,Vern9(),abstol=1e-14,reltol=1e-14)
  res,err = quadgk((t)-> (sum(sol(t)).^2)./2,0.0,10.0,abstol=1e-14,reltol=1e-10)
  res
end
res2 = ForwardDiff.gradient(G,[1.5,1.0,3.0])
res3 = Calculus.gradient(G,[1.5,1.0,3.0])

@test norm(res' - res2) < 1e-8
@test norm(res' - res3) < 1e-6
