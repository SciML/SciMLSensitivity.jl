using OrdinaryDiffEq, SciMLSensitivity, Zygote, ForwardDiff, Test

function lotka_volterra!(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = dx = α * x - β * x * y
    du[2] = dy = -δ * y + γ * x * y
end

# Initial condition
u0 = [1.0, 1.0]

# Simulation interval and intermediary points
tspan = (0.0, 10.0)
tsteps = 0.0:0.1:10.0

# LV equation parameter. p = [α, β, δ, γ]
p = [1.5, 1.0, 3.0, 1.0]

# Setup the ODE problem, then solve
prob = ODEProblem(lotka_volterra!, u0, tspan, p)

function loss(p)
    sol = solve(prob, Tsit5(), p = p, save_idxs = [2], saveat = tsteps, abstol = 1e-14,
        reltol = 1e-14)
    loss = sum(abs2, sol .- 1)
    return loss
end

grad1 = Zygote.gradient(loss, p)[1]
grad2 = ForwardDiff.gradient(loss, p)
@test grad1 ≈ grad2
