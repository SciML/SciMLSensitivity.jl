using SciMLSensitivity, DelayDiffEq, Test

# Use Mooncake on Julia 1.12+ (Zygote has issues), Zygote on older versions
if VERSION >= v"1.12"
    using Mooncake
    function compute_gradient(f, x)
        return Mooncake.value_and_gradient!!(Mooncake.build_rrule(f, x), f, x)[2][2]
    end
else
    using Zygote
    function compute_gradient(f, x)
        return Zygote.gradient(f, x)[1]
    end
end

## Setup DDE to optimize
function delay_lotka_volterra(du, u, h, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = dx = (α - β * y) * h(p, t - 0.1)[1]
    return du[2] = dy = (δ * x - γ) * y
end
h(p, t) = ones(eltype(p), 2)
prob = DDEProblem(delay_lotka_volterra, [1.0, 1.0], h, (0.0, 10.0), constant_lags = [0.1])
p = [2.2, 1.0, 2.0, 0.4]
function predict_fd_dde(p)
    return solve(
        prob, MethodOfSteps(Tsit5()), p = p, saveat = 0.0:0.1:10.0, reltol = 1.0e-4,
        sensealg = ForwardDiffSensitivity()
    )[1, :]
end
loss_fd_dde(p) = sum(abs2, x - 1 for x in predict_fd_dde(p))
loss_fd_dde(p)
@test !iszero(compute_gradient(loss_fd_dde, p))

function predict_rd_dde(p)
    return solve(
        prob, MethodOfSteps(Tsit5()), p = p, saveat = 0.1, reltol = 1.0e-4,
        sensealg = TrackerAdjoint()
    )[
        1,
        :,
    ]
end
loss_rd_dde(p) = sum(abs2, x - 1 for x in predict_rd_dde(p))
loss_rd_dde(p)
@test !iszero(compute_gradient(loss_rd_dde, p))

@test compute_gradient(loss_fd_dde, p) ≈ compute_gradient(loss_rd_dde, p) rtol = 1.0e-2
