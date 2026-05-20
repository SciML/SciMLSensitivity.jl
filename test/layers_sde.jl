using SciMLSensitivity, StochasticDiffEq, Test
using Zygote
function compute_gradient(f, x)
    return Zygote.gradient(f, x)[1]
end

function lotka_volterra(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = dx = α * x - β * x * y
    du[2] = dy = -δ * y + γ * x * y
    return nothing
end
function lotka_volterra(u, p, t)
    x, y = u
    α, β, δ, γ = p
    dx = α * x - β * x * y
    dy = -δ * y + γ * x * y
    return [dx, dy]
end
function lotka_volterra_noise(du, u, p, t)
    du[1] = 0.01u[1]
    du[2] = 0.01u[2]
    return nothing
end
function lotka_volterra_noise(u, p, t)
    return [0.01u[1], 0.01u[2]]
end
p = [2.2, 1.0, 2.0, 0.4]
saveat = 0.0:0.1:0.5
function predict_fd_sde(prob, p)
    return solve(prob, SOSRI(); p, saveat, sensealg = ForwardDiffSensitivity())[1, :]
end

prob = SDEProblem(lotka_volterra, lotka_volterra_noise, [1.0, 1.0], (0.0, 10.0))
loss_fd_sde(p) = sum(abs2, x - 1 for x in predict_fd_sde(prob, p))
loss_fd_sde(p)

prob = SDEProblem{false}(lotka_volterra, lotka_volterra_noise, [1.0, 1.0], (0.0, 10.0))
loss_fd_sde(p) = sum(abs2, x - 1 for x in predict_fd_sde(prob, p))
loss_fd_sde(p)

@test !iszero(compute_gradient(loss_fd_sde, p))

function predict_rd_sde(prob, p)
    return solve(prob, SOSRI(); p, saveat, sensealg = TrackerAdjoint())[1, :]
end
prob = SDEProblem(lotka_volterra, lotka_volterra_noise, [1.0, 1.0], (0.0, 0.5))
loss_rd_sde(p) = sum(abs2, x - 1 for x in predict_rd_sde(prob, p))
prob = SDEProblem{false}(lotka_volterra, lotka_volterra_noise, [1.0, 1.0], (0.0, 0.5))
loss_rd_sde(p) = sum(abs2, x - 1 for x in predict_rd_sde(prob, p))
@test !iszero(compute_gradient(loss_rd_sde, p))
