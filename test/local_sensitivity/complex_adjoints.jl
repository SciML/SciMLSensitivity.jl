using DiffEqSensitivity, OrdinaryDiffEq, Zygote, LinearAlgebra, FiniteDiff, Test
A  = [1.0*im  2.0; 3.0 4.0]
u0 = [1.0 0.0*im; 0.0 1.0]
tspan = (0.0, 1.0)

function f(u,p,t)
    (A*u)*(p[1]*t + p[2]*t^2 + p[3]*t^3 + p[4]*t^4)
end

p = [1.5 + im, 1.0, 3.0, 1.0]
prob = ODEProblem{false}(f,u0,tspan,p)

utarget = [0.0*im 1.0; 1.0 0.0]

function loss_adjoint(p)
    ufinal = last(solve(prob, Tsit5(), p=p, abstol=1e-12, reltol=1e-12))
    loss = 1 - abs(tr(ufinal*utarget')/2)^2
    return loss
end

grad1 = Zygote.gradient(loss_adjoint,Complex{Float64}[1.5, 1.0, 3.0, 1.0])[1]
grad2 = FiniteDiff.finite_difference_gradient(loss_adjoint,Complex{Float64}[1.5, 1.0, 3.0, 1.0])
@test grad1 ≈ grad2
