using Test
using OrdinaryDiffEq
using DiffEqSensitivity
using DiffEqBase
using ForwardDiff
using QuadGK

function pendulum_eom(dx, x, p, t)
    dx[1] = x[2]
    dx[2] = -sin(x[1]) + (-p[1]*sin(x[1]) + p[2]*x[2])  # Second term is a simple controller that stabilizes π
end

x0 = [0.1, 0.0]
tspan = (0.0, 10.0)
p = [-24.05, -19.137]
prob = ODEProblem(pendulum_eom, x0, tspan, p)
sol = solve(prob, Vern9(), abstol=1e-8, reltol=1e-8)

g(x, p, t) = 1.0*(x[1] - π)^2 + 1.0*x[2]^2 + 5.0*(-p[1]*sin(x[1]) + p[2]*x[2])^2
dgdu(out, y, p, t) = ForwardDiff.gradient!(out, y -> g(y, p, t), y)
dgdp(out, y, p, t) = ForwardDiff.gradient!(out, p -> g(y, p, t), p)

res_interp = adjoint_sensitivities(sol,Vern9(),g,nothing,(dgdu, dgdp),abstol=1e-8,
                                reltol=1e-8,iabstol=1e-8,ireltol=1e-8, sensealg=InterpolatingAdjoint())
res_quad = adjoint_sensitivities(sol,Vern9(),g,nothing,(dgdu, dgdp),abstol=1e-8,
                                reltol=1e-8,iabstol=1e-8,ireltol=1e-8, sensealg=QuadratureAdjoint())
#res_back = adjoint_sensitivities(sol,Vern9(),g,nothing,(dgdu, dgdp),abstol=1e-8,
#                                reltol=1e-8,iabstol=1e-8,ireltol=1e-8, sensealg=BacksolveAdjoint(checkpointing=true), sol=sol.t) # it's blowing up

function G(p)
    tmp_prob = remake(prob,p=p,u0=convert.(eltype(p), prob.u0))
    sol = solve(tmp_prob,Vern9(),abstol=1e-8,reltol=1e-8)
    res,err = quadgk((t)-> g(sol(t), p, t), 0.0,10.0,atol=1e-8,rtol=1e-8)
    res
end
res2 = ForwardDiff.gradient(G,p)

@test res_interp[2]' ≈ res2 atol=1e-5
@test res_quad[2]' ≈ res2 atol=1e-5
