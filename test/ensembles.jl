using SciMLSensitivity, OrdinaryDiffEq, Optimization, OptimizationOptimisers, Test, Zygote

@testset "$(i): EnsembleAlg = $(alg)" for (i, alg) in enumerate((EnsembleSerial(),
    EnsembleThreads(), EnsembleSerial()))
    function prob_func(prob, i, repeat)
        remake(prob, u0 = 0.5 .+ i / 100 .* prob.u0)
    end
    function model(p)
        prob = ODEProblem((u, p, t) -> 1.01u .* p, p[1:1], (0.0, 1.0), p[2:2])

        ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
        sim = solve(ensemble_prob, Tsit5(), alg, saveat = 0.1, trajectories = 100)
        return i == 3 ? sim.u : sim
    end

    # loss function
    loss = if i == 3
        (p, _) -> sum(abs2, [sum(abs2, 1.0 .- u) for u in model(p)])
    else
        (p, _) -> sum(abs2, 1.0 .- Array(model(p)))
    end

    cb = function (p, l) # callback function to observe training
        @info alg=alg loss=l
        return false
    end

    l1 = loss([1.0, 3.0], nothing)
    @show l1
    res = solve(OptimizationProblem(OptimizationFunction(loss, AutoZygote()),
            [1.0, 3.0]),
        Adam(0.1); callback = cb, maxiters = 10)
    l2 = loss(res.u, nothing)
    @test 10l2 < l1
end
