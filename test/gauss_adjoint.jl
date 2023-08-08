using OrdinaryDiffEq
using SciMLSensitivity
using Test

function f(du, u, p, t)
    du[1] = dx = p[1] * u[1] - p[2] * u[1] * u[2]
    du[2] = dy = -p[3] * u[2] + u[1] * u[2]
end

p = [1.5, 1.0, 3.0]

g(u, p, t) = (sum(u) .^ 2) ./ 2

function dg(out, u, p, t)
    out[1] = u[1] + u[2]
    out[2] = u[1] + u[2]
end

prob = ODEProblem(f, [1.0; 1.0], (0.0, 0.1), p)
sol = solve(prob, DP8())
res_gauss = adjoint_sensitivities(sol, Vern9(), sensealg=GaussAdjoint(), dgdu_continuous = dg, g = g, abstol = 1e-8, reltol = 1e-8)
res_quad = adjoint_sensitivities(sol, Vern9(), sensealg=QuadratureAdjoint(), dgdu_continuous = dg, g = g, abstol = 1e-8, reltol = 1e-8)

@test isapprox(res_gauss[1], res_quad[1], atol = 1e-8, rtol = 1e-8)
@test isapprox(res_gauss[2], res_quad[2], atol = 1e-8, rtol = 1e-8)
