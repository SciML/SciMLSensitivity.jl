using DiffEqSensitivity,OrdinaryDiffEq

function f(du,u,p,t)
  du[1] = dx = p[1]*u[1] - p[2]*u[1]*u[2]
  du[2] = dy = -p[3]*u[2] + p[4]*u[1]*u[2]
end

p = [1.5,1.0,3.0,1.0]
prob = ODEProblem(f,[1.0;1.0],(0.0,10.0),p)
sol = solve(prob,Tsit5(),abstol=1e-14,reltol=1e-14)

# g(t,u,i) = (1-u)^2/2, L2 away from 1
function dg(out,u,p,t,i)
  (out.=2.0.-u)
end

easy_res = adjoint_sensitivities(sol,Tsit5(),dg,0.0:0.5:10.0,abstol=1e-14,
                                 reltol=1e-14,iabstol=1e-14,ireltol=1e-12)
