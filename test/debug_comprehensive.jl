using OrdinaryDiffEq, SciMLSensitivity, ForwardDiff, Zygote, ReverseDiff, Tracker, Enzyme
using Test
Enzyme.API.typeWarning!(false)

println("=== SECTION 1: Basic setup ===")
flush(stdout)

odef(du, u, p, t) = du .= u .* p
const prob = ODEProblem(odef, [2.0], (0.0, 1.0), [3.0])

struct senseloss{T}
    sense::T
end
function (f::senseloss)(u0p)
    sum(solve(prob, Tsit5(), u0 = u0p[1:1], p = u0p[2:2], abstol = 1e-12,
        reltol = 1e-12, saveat = 0.1, sensealg = f.sense))
end

u0p = [2.0, 3.0]
dup = Zygote.gradient(senseloss(InterpolatingAdjoint()), u0p)[1]

println("=== SECTION 2: All ReverseDiff tests ===")
flush(stdout)

@test ReverseDiff.gradient(senseloss(InterpolatingAdjoint()), u0p) ≈ dup
@test ReverseDiff.gradient(senseloss(ReverseDiffAdjoint()), u0p) ≈ dup
@test ReverseDiff.gradient(senseloss(TrackerAdjoint()), u0p) ≈ dup
@test ReverseDiff.gradient(senseloss(ForwardDiffSensitivity()), u0p) ≈ dup

println("=== SECTION 3: All Tracker tests ===")
flush(stdout)

@test Tracker.gradient(senseloss(InterpolatingAdjoint()), u0p)[1] ≈ dup
@test Tracker.gradient(senseloss(ReverseDiffAdjoint()), u0p)[1] ≈ dup
@test Tracker.gradient(senseloss(TrackerAdjoint()), u0p)[1] ≈ dup
@test Tracker.gradient(senseloss(ForwardDiffSensitivity()), u0p)[1] ≈ dup

println("=== SECTION 4: ForwardDiff test ===")
flush(stdout)

@test ForwardDiff.gradient(senseloss(InterpolatingAdjoint()), u0p) ≈ dup

println("=== SECTION 5: Enzyme tests ===")
flush(stdout)

@test only(Enzyme.gradient(Reverse, senseloss(InterpolatingAdjoint()), u0p)) ≈ dup
@test_throws SciMLSensitivity.EnzymeTrackedRealError only(Enzyme.gradient(
    Reverse, senseloss(ReverseDiffAdjoint()), u0p))≈dup
@test_throws SciMLSensitivity.EnzymeTrackedRealError only(Enzyme.gradient(
    Reverse, senseloss(TrackerAdjoint()), u0p))≈dup
@test only(Enzyme.gradient(Reverse, senseloss(ForwardDiffSensitivity()), u0p)) ≈ dup

println("=== SECTION 6: Starting senseloss2 (prob2) ===")
flush(stdout)

struct senseloss2{T}
    sense::T
end
prob2 = ODEProblem((du, u, p, t) -> du .= u .* p, [2.0], (0.0, 1.0), [3.0])

function (f::senseloss2)(u0p)
    sum(solve(prob2, Tsit5(), u0 = u0p[1:1], p = u0p[2:2], abstol = 1e-12,
        reltol = 1e-12, saveat = 0.1, sensealg = f.sense))
end

u0p = [2.0, 3.0]
dup = Zygote.gradient(senseloss2(InterpolatingAdjoint()), u0p)[1]

println("=== SECTION 7: senseloss2 ReverseDiff tests ===")
flush(stdout)

@test ReverseDiff.gradient(senseloss2(InterpolatingAdjoint()), u0p) ≈ dup
@test ReverseDiff.gradient(senseloss2(ReverseDiffAdjoint()), u0p) ≈ dup
@test ReverseDiff.gradient(senseloss2(TrackerAdjoint()), u0p) ≈ dup
@test ReverseDiff.gradient(senseloss2(ForwardDiffSensitivity()), u0p) ≈ dup

println("=== SECTION 8: senseloss2 Tracker tests ===")
flush(stdout)

@test Tracker.gradient(senseloss2(InterpolatingAdjoint()), u0p)[1] ≈ dup
@test Tracker.gradient(senseloss2(ReverseDiffAdjoint()), u0p)[1] ≈ dup
@test Tracker.gradient(senseloss2(TrackerAdjoint()), u0p)[1] ≈ dup
@test Tracker.gradient(senseloss2(ForwardDiffSensitivity()), u0p)[1] ≈ dup

println("=== SECTION 9: senseloss2 ForwardDiff test ===")
flush(stdout)

@test ForwardDiff.gradient(senseloss2(InterpolatingAdjoint()), u0p) ≈ dup

println("=== Completed all basic sections - issue must be in later optimization/matrix sections ===")
flush(stdout)