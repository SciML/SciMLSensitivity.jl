using SciMLSensitivity, OrdinaryDiffEq, Optimization, OptimizationOptimisers, Test
using ADTypes

# These tests differentiate through ensemble solves. Zygote segfaults on Julia
# 1.12+ (#1325), so use Enzyme, which differentiates the ODE `EnsembleProblem`
# adjoint on 1.11 and 1.12. On the LTS (1.10) Enzyme's type analysis rejects the
# reshaped-`ODESolution` indexing in the `sim.u` case (`EnzymeNoTypeError`), so
# keep the working Zygote path there.
if VERSION >= v"1.11"
    using Enzyme
    const AD_BACKEND = AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse))
else
    using Zygote
    const AD_BACKEND = AutoZygote()
end

@testset "$(i): EnsembleAlg = $(alg)" for (i, alg) in enumerate(
        (
            EnsembleSerial(),
            EnsembleThreads(), EnsembleSerial(),
        )
    )
    function prob_func(prob, ctx)
        remake(prob, u0 = 0.5 .+ ctx.sim_id / 100 .* prob.u0)
    end
    function model(p)
        prob = ODEProblem((u, p, t) -> 1.01u .* p, p[1:1], (0.0, 1.0), p[2:2])

        ensemble_prob = EnsembleProblem(prob; prob_func)
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
        @info alg = alg loss = l
        return false
    end

    l1 = loss([1.0, 3.0], nothing)
    @show l1
    res = solve(
        OptimizationProblem(
            OptimizationFunction(loss, AD_BACKEND),
            [1.0, 3.0]
        ),
        Adam(0.1); callback = cb, maxiters = 10
    )
    l2 = loss(res.u, nothing)
    @test 10l2 < l1
end
