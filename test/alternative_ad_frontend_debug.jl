using OrdinaryDiffEq, SciMLSensitivity, ForwardDiff, Zygote, ReverseDiff, Tracker, Enzyme,
      FiniteDiff, Mooncake
using Test
Enzyme.API.typeWarning!(false)

println("DEBUG: Starting alternative_ad_frontend.jl")
flush(stdout)

mooncake_gradient(f, x) = Mooncake.value_and_gradient!!(Mooncake.build_rrule(f, x), f, x)[2][2]

odef(du, u, p, t) = du .= u .* p
const prob = ODEProblem(odef, [2.0], (0.0, 1.0), [3.0])

println("DEBUG: Defined basic structs and functions")
flush(stdout)

struct senseloss0{T}
    sense::T
end
function (f::senseloss0)(u0p)
    prob = ODEProblem{true}(odef, u0p[1:1], (0.0, 1.0), u0p[2:2])
    sum(solve(prob, Tsit5(), abstol = 1e-12, reltol = 1e-12, saveat = 0.1))
end
u0p = [2.0, 3.0]
du0p = zeros(2)

println("DEBUG: About to test senseloss0 with Zygote")
flush(stdout)
dup = Zygote.gradient(senseloss0(InterpolatingAdjoint()), u0p)[1]
println("DEBUG: Completed Zygote gradient")
flush(stdout)

println("DEBUG: About to test senseloss0 with Enzyme")
flush(stdout)
Enzyme.autodiff(Reverse, senseloss0(InterpolatingAdjoint()), Active, Duplicated(u0p, du0p))
println("DEBUG: Completed Enzyme autodiff")
flush(stdout)

println("DEBUG: About to test senseloss0 with Mooncake")
flush(stdout)
dup_mc = mooncake_gradient(senseloss0(InterpolatingAdjoint()), u0p)
println("DEBUG: Completed Mooncake gradient")
flush(stdout)

@test du0p ≈ dup
@test dup_mc ≈ dup

println("DEBUG: Starting senseloss tests")
flush(stdout)

struct senseloss{T}
    sense::T
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

println("DEBUG: About to compute reference gradient with Zygote")
flush(stdout)
dup = Zygote.gradient(senseloss(InterpolatingAdjoint()), u0p)[1]
println("DEBUG: Completed reference gradient")
flush(stdout)

println("DEBUG: Testing ReverseDiff gradients")
flush(stdout)
@test ReverseDiff.gradient(senseloss(InterpolatingAdjoint()), u0p) ≈ dup
println("DEBUG: Completed ReverseDiff InterpolatingAdjoint")
flush(stdout)

@test ReverseDiff.gradient(senseloss(ReverseDiffAdjoint()), u0p) ≈ dup
println("DEBUG: Completed ReverseDiff ReverseDiffAdjoint")
flush(stdout)

@test ReverseDiff.gradient(senseloss(TrackerAdjoint()), u0p) ≈ dup
println("DEBUG: Completed ReverseDiff TrackerAdjoint")
flush(stdout)

@test ReverseDiff.gradient(senseloss(ForwardDiffSensitivity()), u0p) ≈ dup
println("DEBUG: Completed ReverseDiff ForwardDiffSensitivity")
flush(stdout)

@test_broken ReverseDiff.gradient(senseloss(ForwardSensitivity()), u0p) ≈ dup # broken because ForwardSensitivity not compatible with perturbing u0

println("DEBUG: Testing Tracker gradients")
flush(stdout)
@test Tracker.gradient(senseloss(InterpolatingAdjoint()), u0p)[1] ≈ dup
println("DEBUG: Completed Tracker InterpolatingAdjoint")
flush(stdout)

@test Tracker.gradient(senseloss(ReverseDiffAdjoint()), u0p)[1] ≈ dup
println("DEBUG: Completed Tracker ReverseDiffAdjoint")
flush(stdout)

@test Tracker.gradient(senseloss(TrackerAdjoint()), u0p)[1] ≈ dup
println("DEBUG: Completed Tracker TrackerAdjoint")  
flush(stdout)

@test Tracker.gradient(senseloss(ForwardDiffSensitivity()), u0p)[1] ≈ dup
println("DEBUG: Completed Tracker ForwardDiffSensitivity")
flush(stdout)

@test_broken Tracker.gradient(senseloss(ForwardSensitivity()), u0p)[1] ≈ dup # broken because ForwardSensitivity not compatible with perturbing u0

println("DEBUG: Testing ForwardDiff gradient")
flush(stdout)
@test ForwardDiff.gradient(senseloss(InterpolatingAdjoint()), u0p) ≈ dup
println("DEBUG: Completed ForwardDiff gradient")
flush(stdout)

println("DEBUG: Testing Enzyme gradients")
flush(stdout)
@test only(Enzyme.gradient(Reverse, senseloss(InterpolatingAdjoint()), u0p)) ≈ dup
println("DEBUG: Completed Enzyme InterpolatingAdjoint")
flush(stdout)

@test_throws SciMLSensitivity.EnzymeTrackedRealError only(Enzyme.gradient(
    Reverse, senseloss(ReverseDiffAdjoint()), u0p))≈dup
println("DEBUG: Completed Enzyme ReverseDiffAdjoint (expected error)")
flush(stdout)

@test_throws SciMLSensitivity.EnzymeTrackedRealError only(Enzyme.gradient(
    Reverse, senseloss(TrackerAdjoint()), u0p))≈dup
println("DEBUG: Completed Enzyme TrackerAdjoint (expected error)")
flush(stdout)

@test only(Enzyme.gradient(Reverse, senseloss(ForwardDiffSensitivity()), u0p)) ≈ dup
println("DEBUG: Completed Enzyme ForwardDiffSensitivity")
flush(stdout)

@test_broken only(Enzyme.gradient(Reverse, senseloss(ForwardSensitivity()), u0p)) ≈ dup # broken because ForwardSensitivity not compatible with perturbing u0

println("DEBUG: Testing Mooncake gradients")
flush(stdout)
@test mooncake_gradient(senseloss(InterpolatingAdjoint()), u0p) ≈ dup
println("DEBUG: Completed Mooncake InterpolatingAdjoint")
flush(stdout)

@test_throws SciMLSensitivity.MooncakeTrackedRealError mooncake_gradient(senseloss(ReverseDiffAdjoint()), u0p)
println("DEBUG: Completed Mooncake ReverseDiffAdjoint (expected error)")
flush(stdout)

@test_throws SciMLSensitivity.MooncakeTrackedRealError mooncake_gradient(senseloss(TrackerAdjoint()), u0p)
println("DEBUG: Completed Mooncake TrackerAdjoint (expected error)")
flush(stdout)

@test mooncake_gradient(senseloss(ForwardDiffSensitivity()), u0p) ≈ dup
println("DEBUG: Completed Mooncake ForwardDiffSensitivity")
flush(stdout)

@test_broken mooncake_gradient(senseloss(ForwardSensitivity()), u0p) ≈ dup # broken because ForwardSensitivity not compatible with perturbing u0

println("DEBUG: Completed first major section, starting senseloss2")
flush(stdout)

# I'll continue with just a few more sections to identify where it hangs
struct senseloss2{T}
    sense::T
end
prob2 = ODEProblem((du, u, p, t) -> du .= u .* p, [2.0], (0.0, 1.0), [3.0])

function (f::senseloss2)(u0p)
    sum(solve(prob2, Tsit5(), u0 = u0p[1:1], p = u0p[2:2], abstol = 1e-12,
        reltol = 1e-12, saveat = 0.1, sensealg = f.sense))
end

u0p = [2.0, 3.0]

println("DEBUG: About to compute senseloss2 reference with Zygote")
flush(stdout)
dup = Zygote.gradient(senseloss2(InterpolatingAdjoint()), u0p)[1]
println("DEBUG: Completed senseloss2 reference")
flush(stdout)

println("DEBUG: Starting senseloss2 ReverseDiff tests")
flush(stdout)
@test ReverseDiff.gradient(senseloss2(InterpolatingAdjoint()), u0p) ≈ dup
println("DEBUG: senseloss2 ReverseDiff InterpolatingAdjoint done")
flush(stdout)

println("DEBUG: TEST - If we reach here, the issue is later in the file")
flush(stdout)

# Add a early exit to see if we can get to this point
println("DEBUG: EARLY EXIT - Stopping test to see where timeout occurs")
flush(stdout)