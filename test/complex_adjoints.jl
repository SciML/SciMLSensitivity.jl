using SciMLSensitivity, OrdinaryDiffEq, Zygote, LinearAlgebra, FiniteDiff, Test
A = [1.0*im 2.0; 3.0 4.0]
u0 = [1.0 0.0*im; 0.0 1.0]
tspan = (0.0, 1.0)

function f(u, p, t)
    (A * u) * (p[1] * t + p[2] * t^2 + p[3] * t^3 + p[4] * t^4)
end

p = [1.5 + im, 1.0, 3.0, 1.0]
prob = ODEProblem{false}(f, u0, tspan, p)

utarget = [0.0*im 1.0; 1.0 0.0]

function loss_adjoint(p)
    ufinal = last(solve(prob, Tsit5(), p = p, abstol = 1e-12, reltol = 1e-12,
        sensealg = InterpolatingAdjoint()))
    loss = 1 - abs(tr(ufinal * utarget') / 2)^2
    return loss
end

grad1 = Zygote.gradient(loss_adjoint, Complex{Float64}[1.5, 1.0, 3.0, 1.0])[1]
grad2 = FiniteDiff.finite_difference_gradient(loss_adjoint,
    Complex{Float64}[1.5, 1.0, 3.0, 1.0])
@test grad1 ≈ grad2

function rhs(u, p, t)
    p .* u
end

function loss_fun(sol)
    final_u = sol[:, end]
    err = sum(abs.(final_u))
    return err
end

function inner_loop(prob, p, loss_fun; sensealg = InterpolatingAdjoint())
    sol = solve(prob, Tsit5(), p = p, saveat = 0.1; sensealg)
    err = loss_fun(sol)
    return err
end

tspan = (0.0, 1.0)
p = [1.0]
u0 = [1.0, 2.0]
prob = ODEProblem(rhs, u0, tspan, p)
grads = Zygote.gradient((p) -> inner_loop(prob, p, loss_fun), p)[1]

u0 = [1.0 + 2.0 * im, 2.0 + 1.0 * im]
prob = ODEProblem(rhs, u0, tspan, p)
dp1 = Zygote.gradient((p) -> inner_loop(prob, p, loss_fun), p)[1]
dp2 = Zygote.gradient((p) -> inner_loop(prob, p, loss_fun; sensealg = QuadratureAdjoint()),
    p)[1]
dp3 = Zygote.gradient((p) -> inner_loop(prob, p, loss_fun; sensealg = BacksolveAdjoint()),
    p)[1]
@test dp1 ≈ dp2 ≈ dp3
@test eltype(dp1) <: Float64

function fiip(du, u, p, t)
    du[1] = dx = p[1] * u[1] - p[2] * u[1] * u[2]
    du[2] = dy = -p[3] * u[2] + p[4] * u[1] * u[2]
end
p = [1.5, 1.0, 3.0, 1.0];
u0 = [1.0; 1.0];
prob = ODEProblem(fiip, complex(u0), (0.0, 10.0), complex(p))

function sum_of_solution(u0, p)
    _prob = remake(prob, u0 = u0, p = p)
    real(sum(solve(_prob, Tsit5(), reltol = 1e-6, abstol = 1e-6, saveat = 0.1)))
end

dx = Zygote.gradient(sum_of_solution, complex(u0), complex(p))
