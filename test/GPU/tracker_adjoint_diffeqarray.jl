using CUDA, OrdinaryDiffEq, SciMLSensitivity, Test, Zygote
using SciMLBase: ODEProblem, solve

CUDA.allowscalar(false)

function tracker_loss(u0, p; kwargs...)
    prob = ODEProblem((u, p, _) -> p .* u, u0, (0.0f0, 1.0f0), p)
    sol = solve(prob, Tsit5(); sensealg = TrackerAdjoint(), kwargs...)
    return sum(last(sol.u))
end

u0 = CUDA.ones(Float32, 2)
p = CUDA.fill(0.1f0, 2)
expected = exp.(Array(p))

@testset "kwargs: $kwargs" for kwargs in (
        (; save_everystep = false, save_start = false),
        (; saveat = 0.1f0),
    )
    du0, dp = Zygote.gradient((u0, p) -> tracker_loss(u0, p; kwargs...), u0, p)
    @test Array(du0) ≈ expected rtol = 1.0f-3
    @test Array(dp) ≈ expected rtol = 1.0f-3
end
