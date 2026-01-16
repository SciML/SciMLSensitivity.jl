using SciMLSensitivity, OrdinaryDiffEq, Test, Optimization, OptimizationOptimisers
using ADTypes

# These tests require AD differentiation through solve
# Mooncake doesn't have the necessary rules yet, and Zygote has compatibility
# issues on Julia 1.12+, so we skip these tests on Julia 1.12+
if VERSION >= v"1.12"
    @info "Skipping Layers tests on Julia 1.12+ due to AD compatibility issues"
    @testset "Layers Tests (skipped on Julia 1.12+)" begin
        @test_skip false
    end
else
    using Zygote
    const AD_BACKEND = AutoZygote()
    function compute_gradient(f, x, args...)
        return Zygote.gradient(f, x, args...)[1]
    end

    function lotka_volterra(du, u, p, t)
        x, y = u
        α, β, δ, γ = p
        du[1] = dx = (α - β * y)x
        return du[2] = dy = (δ * x - γ)y
    end
    p = [2.2, 1.0, 2.0, 0.4]
    u0 = [1.0, 1.0]
    prob = ODEProblem(lotka_volterra, u0, (0.0, 10.0), p)

    @testset "sensealg: $(sensealg)" for sensealg in (
            TrackerAdjoint(),
            ForwardDiffSensitivity(), nothing,
        )
        function predict(pu0)
            p = pu0[1:4]
            u0 = pu0[5:6]
            vec(Array(solve(prob, Tsit5(); p, u0, saveat = 0.1, reltol = 1.0e-4, sensealg)))
        end
        loss(pu0, _) = sum(abs2, x .- 1 for x in predict(pu0))

        grads = compute_gradient(loss, [p; u0], nothing)
        @test !iszero(grads)

        cb = function (p, l)
            @info sensealg loss = l
            return false
        end

        l1 = loss([p; u0], nothing)
        @show l1
        res = solve(
            OptimizationProblem(
                OptimizationFunction(loss, AD_BACKEND),
                [p; u0]
            ),
            Adam(0.1); callback = cb, maxiters = 100
        )
        l2 = loss(res.u, nothing)
        @test 10l2 < l1
    end

end  # VERSION < v"1.12" else block
