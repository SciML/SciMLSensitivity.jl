using DiffEqSensitivity,OrdinaryDiffEq, ParameterizedFunctions,
      RecursiveArrayTools, DiffEqBase, ForwardDiff, Calculus, QuadGK
using Base.Test


f = @ode_def_nohes LotkaVolterra begin
  dx = a*x - b*x*y
  dy = -c*y + d*x*y
end a=>1.5 b=>1.0 c=>3.0 d=1

prob = ODEProblem(f,[1.0;1.0],(0.0,10.0))
sol = solve(prob,Vern9(),abstol=1e-14,reltol=1e-14)

t = 0.0:0.5:10.0 # TODO: Add end point handling for callback
# g(t,u,i) = (2-u)^2/2, L2 away from 2
dg(out,u,i) = (out.=1.-u)

adj_prob = ODEAdjointProblem(sol,dg,t)
adj_sol = solve(adj_prob,Vern9(),abstol=1e-14,reltol=1e-14)
integrand = AdjointSensitivityIntegrand(sol,adj_sol)
res,err = quadgk(integrand,0.0,10.0,abstol=1e-14,reltol=1e-10)

using Plots
gr()
plot(adj_sol,tspan=(0.0,10.0))

t_short = 0.5:0.5:10.0
function G(p)
  tmp_prob = problem_new_parameters(prob,p)
  sol = solve(tmp_prob,Vern9(),abstol=1e-14,reltol=1e-14,saveat=t_short,save_start=false)
  A = convert(Array,sol)
  sum(((1-A).^2)./2)
end
G([1.5,1.0,3.0])
res2 = ForwardDiff.gradient(G,[1.5,1.0,3.0])
res3 = Calculus.gradient(G,[1.5,1.0,3.0])

@test norm(res' - res2) < 1e-8
@test norm(res' - res3) < 1e-6
