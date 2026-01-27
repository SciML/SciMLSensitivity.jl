using Test, Distributed

# These tests require AD differentiation through distributed ensemble solves
# Mooncake doesn't have the necessary rules yet, and Zygote has compatibility
# issues on Julia 1.12+, so we skip these tests on Julia 1.12+
if VERSION >= v"1.12"
    @info "Skipping Distributed tests on Julia 1.12+ due to AD compatibility issues"
    @testset "Distributed Tests (skipped on Julia 1.12+)" begin
        @test_skip false
    end
else
    using Optimization, OptimizationOptimisers
    using ADTypes

    addprocs(2)
    @everywhere begin
        using SciMLSensitivity, OrdinaryDiffEq, Test

        pa = [1.0]
        u0 = [3.0]
    end

    function model_distributed(pu0)
        pa = pu0[1:1]
        u0 = pu0[2:2]
        prob = ODEProblem((u, p, t) -> 1.01u .* p, u0, (0.0, 1.0), pa)

        function prob_func(prob, i, repeat)
            return remake(prob, u0 = 0.5 .+ i / 100 .* prob.u0)
        end

        ensemble_prob = EnsembleProblem(prob; prob_func)
        sim = solve(
            ensemble_prob, Tsit5(), EnsembleDistributed(), saveat = 0.1,
            trajectories = 100
        )
        return sim
    end

    # loss function
    loss = (p, _) -> sum(abs2, 1.0 .- Array(model_distributed(p)))

    cb = function (p, l) # callback function to observe training
        @info loss = l
        return false
    end

    l1 = loss([1.0, 3.0], nothing)
    @show l1
    res = solve(
        OptimizationProblem(
            OptimizationFunction(loss, AutoZygote()),
            [1.0, 3.0]
        ),
        Adam(0.1); callback = cb, maxiters = 100
    )
    l2 = loss(res.u, nothing)
    @test 10l2 < l1

end  # VERSION < v"1.12" else block
