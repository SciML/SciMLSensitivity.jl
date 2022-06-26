using OrdinaryDiffEq, SciMLSensitivity, ForwardDiff, Zygote, ReverseDiff, Tracker
using Test

prob = ODEProblem((u, p, t) -> u .* p, [2.0], (0.0, 1.0), [3.0])

struct senseloss
    sense::Any
end
function (f::senseloss)(u0p)
    sum(solve(prob, Tsit5(), u0 = u0p[1:1], p = u0p[2:2], abstol = 1e-12,
              reltol = 1e-12, saveat = 0.1, sensealg = f.sense))
end
function loss(u0p)
    sum(solve(prob, Tsit5(), u0 = u0p[1:1], p = u0p[2:2], abstol = 1e-12, reltol = 1e-12,
              saveat = 0.1))
end
u0p = [2.0, 3.0]

dup = Zygote.gradient(senseloss(InterpolatingAdjoint()), u0p)[1]

@test ReverseDiff.gradient(senseloss(InterpolatingAdjoint()), u0p) ≈ dup
@test_broken ReverseDiff.gradient(senseloss(ReverseDiffAdjoint()), u0p) ≈ dup
@test ReverseDiff.gradient(senseloss(TrackerAdjoint()), u0p) ≈ dup
@test ReverseDiff.gradient(senseloss(ForwardDiffSensitivity()), u0p) ≈ dup
@test_throws SciMLSensitivity.ForwardSensitivityOutOfPlaceError ReverseDiff.gradient(senseloss(ForwardSensitivity()),
                                                                                     u0p)≈dup

@test Tracker.gradient(senseloss(InterpolatingAdjoint()), u0p)[1] ≈ dup
@test_broken Tracker.gradient(senseloss(ReverseDiffAdjoint()), u0p)[1] ≈ dup
@test Tracker.gradient(senseloss(TrackerAdjoint()), u0p)[1] ≈ dup
@test Tracker.gradient(senseloss(ForwardDiffSensitivity()), u0p)[1] ≈ dup
@test_throws SciMLSensitivity.ForwardSensitivityOutOfPlaceError Tracker.gradient(senseloss(ForwardSensitivity()),
                                                                                 u0p)[1]≈dup

@test ForwardDiff.gradient(senseloss(InterpolatingAdjoint()), u0p) ≈ dup

struct senseloss2
    sense::Any
end
prob2 = ODEProblem((du, u, p, t) -> du .= u .* p, [2.0], (0.0, 1.0), [3.0])

function (f::senseloss2)(u0p)
    sum(solve(prob2, Tsit5(), u0 = u0p[1:1], p = u0p[2:2], abstol = 1e-12,
              reltol = 1e-12, saveat = 0.1, sensealg = f.sense))
end

u0p = [2.0, 3.0]

dup = Zygote.gradient(senseloss2(InterpolatingAdjoint()), u0p)[1]

@test ReverseDiff.gradient(senseloss2(InterpolatingAdjoint()), u0p) ≈ dup
@test ReverseDiff.gradient(senseloss2(ReverseDiffAdjoint()), u0p) ≈ dup
@test ReverseDiff.gradient(senseloss2(TrackerAdjoint()), u0p) ≈ dup
@test ReverseDiff.gradient(senseloss2(ForwardDiffSensitivity()), u0p) ≈ dup
@test_broken ReverseDiff.gradient(senseloss2(ForwardSensitivity()), u0p) ≈ dup

@test Tracker.gradient(senseloss2(InterpolatingAdjoint()), u0p)[1] ≈ dup
@test Tracker.gradient(senseloss2(ReverseDiffAdjoint()), u0p)[1] ≈ dup
@test Tracker.gradient(senseloss2(TrackerAdjoint()), u0p)[1] ≈ dup
@test Tracker.gradient(senseloss2(ForwardDiffSensitivity()), u0p)[1] ≈ dup
@test_broken Tracker.gradient(senseloss2(ForwardSensitivity()), u0p)[1] ≈ dup

@test ForwardDiff.gradient(senseloss2(InterpolatingAdjoint()), u0p) ≈ dup

struct senseloss3
    sense::Any
end
function (f::senseloss3)(u0p)
    sum(solve(prob2, Tsit5(), p = u0p, abstol = 1e-12,
              reltol = 1e-12, saveat = 0.1, sensealg = f.sense))
end

u0p = [3.0]

dup = Zygote.gradient(senseloss3(InterpolatingAdjoint()), u0p)[1]

@test ReverseDiff.gradient(senseloss3(InterpolatingAdjoint()), u0p) ≈ dup
@test ReverseDiff.gradient(senseloss3(ReverseDiffAdjoint()), u0p) ≈ dup
@test ReverseDiff.gradient(senseloss3(TrackerAdjoint()), u0p) ≈ dup
@test ReverseDiff.gradient(senseloss3(ForwardDiffSensitivity()), u0p) ≈ dup
@test ReverseDiff.gradient(senseloss3(ForwardSensitivity()), u0p) ≈ dup

@test Tracker.gradient(senseloss3(InterpolatingAdjoint()), u0p)[1] ≈ dup
@test Tracker.gradient(senseloss3(ReverseDiffAdjoint()), u0p)[1] ≈ dup
@test Tracker.gradient(senseloss3(TrackerAdjoint()), u0p)[1] ≈ dup
@test Tracker.gradient(senseloss3(ForwardDiffSensitivity()), u0p)[1] ≈ dup
@test Tracker.gradient(senseloss3(ForwardSensitivity()), u0p)[1] ≈ dup

@test ForwardDiff.gradient(senseloss3(InterpolatingAdjoint()), u0p) ≈ dup

struct senseloss4
    sense::Any
end
function (f::senseloss4)(u0p)
    sum(solve(prob, Tsit5(), p = u0p, abstol = 1e-12,
              reltol = 1e-12, saveat = 0.1, sensealg = f.sense))
end

u0p = [3.0]

dup = Zygote.gradient(senseloss4(InterpolatingAdjoint()), u0p)[1]

@test ReverseDiff.gradient(senseloss4(InterpolatingAdjoint()), u0p) ≈ dup
@test_broken ReverseDiff.gradient(senseloss4(ReverseDiffAdjoint()), u0p) ≈ dup
@test ReverseDiff.gradient(senseloss4(TrackerAdjoint()), u0p) ≈ dup
@test ReverseDiff.gradient(senseloss4(ForwardDiffSensitivity()), u0p) ≈ dup
@test_throws SciMLSensitivity.ForwardSensitivityOutOfPlaceError ReverseDiff.gradient(senseloss4(ForwardSensitivity()),
                                                                                     u0p)≈dup

@test Tracker.gradient(senseloss4(InterpolatingAdjoint()), u0p)[1] ≈ dup
@test_broken Tracker.gradient(senseloss4(ReverseDiffAdjoint()), u0p)[1] ≈ dup
@test Tracker.gradient(senseloss4(TrackerAdjoint()), u0p)[1] ≈ dup
@test Tracker.gradient(senseloss4(ForwardDiffSensitivity()), u0p)[1] ≈ dup
@test_throws SciMLSensitivity.ForwardSensitivityOutOfPlaceError Tracker.gradient(senseloss4(ForwardSensitivity()),
                                                                                 u0p)[1]≈dup

@test ForwardDiff.gradient(senseloss4(InterpolatingAdjoint()), u0p) ≈ dup
