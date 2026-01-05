using SciMLSensitivity, DelayDiffEq, Test

# These tests require AD differentiation through DDE solve
# Mooncake doesn't have the necessary rules yet, and Zygote has compatibility
# issues on Julia 1.12+, so we skip these tests on Julia 1.12+
if VERSION >= v"1.12"
    @info "Skipping Layers DDE tests on Julia 1.12+ due to AD compatibility issues"
    @testset "Layers DDE (skipped on Julia 1.12+)" begin
        @test_skip false
    end
else
    using Zygote
    function compute_gradient(f, x)
        return Zygote.gradient(f, x)[1]
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
            prob, MethodOfSteps(Tsit5()); p, saveat = 0.0:0.1:10.0, reltol = 1.0e-4,
            sensealg = ForwardDiffSensitivity()
        )[1, :]
    end
    loss_fd_dde(p) = sum(abs2, x - 1 for x in predict_fd_dde(p))
    loss_fd_dde(p)
    @test !iszero(compute_gradient(loss_fd_dde, p))

    function predict_rd_dde(p)
        return solve(
            prob, MethodOfSteps(Tsit5()); p, saveat = 0.1, reltol = 1.0e-4,
            sensealg = TrackerAdjoint()
        )[1, :]
    end
    loss_rd_dde(p) = sum(abs2, x - 1 for x in predict_rd_dde(p))
    loss_rd_dde(p)
    @test !iszero(compute_gradient(loss_rd_dde, p))

    @test compute_gradient(loss_fd_dde, p) ≈ compute_gradient(loss_rd_dde, p) rtol = 1.0e-2

end  # VERSION < v"1.12" else block
