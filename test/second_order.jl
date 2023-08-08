using SciMLSensitivity, OrdinaryDiffEq, DiffEqBase, ForwardDiff
using Test

function fb(du, u, p, t)
    du[1] = dx = p[1] * u[1] - p[2] * u[1] * u[2]
    du[2] = dy = -p[3] * u[2] + p[4] * u[1] * u[2]
end

function jac(J, u, p, t)
    (x, y, a, b, c) = (u[1], u[2], p[1], p[2], p[3])
    J[1, 1] = a + y * b * -1
    J[2, 1] = y
    J[1, 2] = b * x * -1
    J[2, 2] = c * -1 + x
end

f = ODEFunction(fb, jac = jac)
p = [1.5, 1.0, 3.0, 1.0];
u0 = [1.0; 1.0];
prob = ODEProblem(f, u0, (0.0, 10.0), p)
loss(sol) = sum(sol)
v = ones(4)

H = second_order_sensitivities(loss, prob, Vern9(), saveat = 0.1, abstol = 1e-12,
    reltol = 1e-12)
Hv = second_order_sensitivity_product(loss, v, prob, Vern9(), saveat = 0.1, abstol = 1e-12,
    reltol = 1e-12)

function _loss(p)
    loss(solve(prob, Vern9(); u0 = u0, p = p, saveat = 0.1, abstol = 1e-12, reltol = 1e-12))
end
H2 = ForwardDiff.hessian(_loss, p)
H2v = H * v

@test H ≈ H2
@test Hv ≈ H2v
