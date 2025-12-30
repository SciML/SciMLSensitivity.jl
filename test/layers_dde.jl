using SciMLSensitivity, Zygote, DelayDiffEq, Test

## Setup DDE to optimize
function delay_lotka_volterra(du, u, h, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = dx = (α - β * y) * h(p, t - 0.1)[1]
    du[2] = dy = (δ * x - γ) * y
end
h(p, t) = ones(eltype(p), 2)
prob = DDEProblem(delay_lotka_volterra, [1.0, 1.0], h, (0.0, 10.0), constant_lags = [0.1])
p = [2.2, 1.0, 2.0, 0.4]
function predict_fd_dde(p)
    solve(prob, MethodOfSteps(Tsit5()), p = p, saveat = 0.0:0.1:10.0, reltol = 1e-4,
        sensealg = ForwardDiffSensitivity())[1, :]
end
loss_fd_dde(p) = sum(abs2, x - 1 for x in predict_fd_dde(p))
loss_fd_dde(p)
@test !iszero(Zygote.gradient(loss_fd_dde, p)[1])

function predict_rd_dde(p)
    solve(prob, MethodOfSteps(Tsit5()), p = p, saveat = 0.1, reltol = 1e-4,
        sensealg = TrackerAdjoint())[1,
        :]
end
loss_rd_dde(p) = sum(abs2, x - 1 for x in predict_rd_dde(p))
loss_rd_dde(p)
@test !iszero(Zygote.gradient(loss_rd_dde, p)[1])

@test Zygote.gradient(loss_fd_dde, p)[1]≈Zygote.gradient(loss_rd_dde, p)[1] rtol=1e-2

# Test GaussAdjoint with DDEProblem (experimental - see issue #1074)
# Note: GaussAdjoint for DDEs is experimental and may not produce accurate gradients.
# This test verifies that the code runs without crashing.
function predict_gauss_dde(p)
    solve(prob, MethodOfSteps(Tsit5()), p = p, saveat = 0.0:0.1:10.0, reltol = 1e-4,
        sensealg = GaussAdjoint(autojacvec = ZygoteVJP()))[1, :]
end
loss_gauss_dde(p) = sum(abs2, x - 1 for x in predict_gauss_dde(p))
# Verify gradient computation runs without error
gauss_grad = Zygote.gradient(loss_gauss_dde, p)[1]
@test !iszero(gauss_grad)
# Note: Accuracy comparison is @test_broken until DDE adjoint is fully implemented
@test_broken Zygote.gradient(loss_fd_dde, p)[1]≈gauss_grad rtol=1e-1
