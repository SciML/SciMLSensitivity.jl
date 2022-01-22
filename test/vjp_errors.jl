using OrdinaryDiffEq, DiffEqSensitivity, Test

function system!(du, u, p, t, controller)

    α, β, γ, δ = 0.5f0, 1.0f0, 1.0f0, 1.0f0

    y1, y2 = u
    c1, c2 = controller(u, p)

    y1_prime = -(c1 + α * c1^2) * y1 + δ * c2
    y2_prime = (β * c1 - γ * c2) * y1

    [y1_prime,y2_prime]
end

function loss(params, prob, tsteps)
    sensealg=QuadratureAdjoint(autojacvec=ZygoteVJP())
    sol = solve(prob, Tsit5(); p=params, saveat=tsteps, sensealg)
    return -Array(sol)[2, end]  # second variable, last value, maximize
end

u0 = [1.0f0, 0.0f0]
tspan = (0.0f0, 1.0f0)
tsteps = 0.0f0:0.01f0:1.0f0

controller = function (x,p)
    σ.(x .* p)
end

θ = randn(2)

dudt!(du, u, p, t) = system!(du, u, p, t, controller)
prob = ODEProblem(dudt!, u0, tspan, θ)

loss(params) = loss(params, prob, tsteps)
@test_throws Any Zygote.gradient(loss, θ)
