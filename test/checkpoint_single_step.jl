using SciMLSensitivity, OrdinaryDiffEqTsit5, Test

# Regression test for a BoundsError in
# `ODEInterpolatingAdjointSensitivityFunction.split_states` when the
# checkpoint solution that is built between two adjacent saveat / discrete
# observation times only contains a single saved time point.
#
# Previously, the dt for the recomputed forward sub-solve was computed as
# `abs(cpsol_t[end] - cpsol_t[end - 1])`, which throws
# `BoundsError: attempt to access 1-element Vector at index [0]` when
# `length(cpsol_t) == 1`. The fix falls back to the interval width.

function dudt(u, p, t)
    return p .* u
end

u0 = Float32[1.0]
p = Float32[-0.5]
tspan = (0.0f0, 1.5f0)
prob = ODEProblem{false}(dudt, u0, tspan, p)

# saveat times that are tightly spaced enough that an adaptive solver may
# only take a single step inside a checkpoint interval
time_batch = collect(range(tspan[1], tspan[2]; length = 30))

forward_sol = solve(prob, Tsit5(); p, saveat = time_batch)

batch = ones(Float32, length(time_batch))
dgdu_discrete = (out, u, p, t, i) -> (out .= -2 * (batch[i] .- u[1]))

@test_nowarn adjoint_sensitivities(
    forward_sol, Tsit5();
    t = time_batch, p,
    dgdu_discrete,
    sensealg = InterpolatingAdjoint()
)

result = adjoint_sensitivities(
    forward_sol, Tsit5();
    t = time_batch, p,
    dgdu_discrete,
    sensealg = InterpolatingAdjoint()
)
@test result isa Tuple
@test length(result) == 2
@test all(isfinite, result[1])
@test all(isfinite, result[2])
