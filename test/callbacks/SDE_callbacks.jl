using StochasticDiffEq, Zygote
using SciMLSensitivity, Test, ForwardDiff

abstol = 1e-12
reltol = 1e-12
savingtimes = 0.5

function test_SDE_callbacks()
    function dt!(du, u, p, t)
        x, y = u
        α, β, δ, γ = p
        du[1] = dx = α * x - β * x * y
        du[2] = dy = -δ * y + γ * x * y
    end

    function dW!(du, u, p, t)
        du[1] = 0.1u[1]
        du[2] = 0.1u[2]
    end

    u0 = [1.0, 1.0]
    tspan = (0.0, 10.0)
    p = [2.2, 1.0, 2.0, 0.4]
    prob_sde = SDEProblem(dt!, dW!, u0, tspan, p)

    condition(u, t, integrator) = integrator.t > 9.0 #some condition
    function affect!(integrator)
        #println("Callback")  #some callback
    end
    cb = DiscreteCallback(condition, affect!, save_positions = (false, false))

    function predict_sde(p)
        return Array(solve(prob_sde, EM(), p = p, saveat = savingtimes,
            sensealg = ForwardDiffSensitivity(), dt = 0.001, callback = cb))
    end

    loss_sde(p) = sum(abs2, x - 1 for x in predict_sde(p))

    loss_sde(p)
    @time dp = gradient(p) do p
        loss_sde(p)
    end

    @test !iszero(dp[1])
end

@testset "SDEs" begin
    println("SDEs")
    test_SDE_callbacks()
end
