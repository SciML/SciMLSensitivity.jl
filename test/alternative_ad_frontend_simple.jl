using OrdinaryDiffEq, SciMLSensitivity, ForwardDiff, Zygote, ReverseDiff, Tracker, Enzyme
using Test
Enzyme.API.typeWarning!(false)

println("DEBUG: Starting alternative_ad_frontend.jl")
flush(stdout)

odef(du, u, p, t) = du .= u .* p
const prob = ODEProblem(odef, [2.0], (0.0, 1.0), [3.0])

println("DEBUG: Defined basic structs and functions")
flush(stdout)

struct senseloss{T}
    sense::T
end
function (f::senseloss)(u0p)
    sum(solve(prob, Tsit5(), u0 = u0p[1:1], p = u0p[2:2], abstol = 1e-12,
        reltol = 1e-12, saveat = 0.1, sensealg = f.sense))
end

u0p = [2.0, 3.0]

println("DEBUG: About to compute reference gradient with Zygote")
flush(stdout)
dup = Zygote.gradient(senseloss(InterpolatingAdjoint()), u0p)[1]
println("DEBUG: Completed reference gradient")
flush(stdout)

println("DEBUG: Testing ReverseDiff.gradient with InterpolatingAdjoint")
flush(stdout)
result1 = ReverseDiff.gradient(senseloss(InterpolatingAdjoint()), u0p)
println("DEBUG: Completed ReverseDiff InterpolatingAdjoint")
flush(stdout)

println("DEBUG: Testing ReverseDiff.gradient with ReverseDiffAdjoint")
flush(stdout)
result2 = ReverseDiff.gradient(senseloss(ReverseDiffAdjoint()), u0p)
println("DEBUG: Completed ReverseDiff ReverseDiffAdjoint")
flush(stdout)

println("DEBUG: Testing ReverseDiff.gradient with TrackerAdjoint")
flush(stdout)
result3 = ReverseDiff.gradient(senseloss(TrackerAdjoint()), u0p)
println("DEBUG: Completed ReverseDiff TrackerAdjoint")
flush(stdout)

println("DEBUG: Testing Tracker.gradient with InterpolatingAdjoint")
flush(stdout)
result4 = Tracker.gradient(senseloss(InterpolatingAdjoint()), u0p)[1]
println("DEBUG: Completed Tracker InterpolatingAdjoint")
flush(stdout)

println("DEBUG: Testing Tracker.gradient with ReverseDiffAdjoint")
flush(stdout)
result5 = Tracker.gradient(senseloss(ReverseDiffAdjoint()), u0p)[1] 
println("DEBUG: Completed Tracker ReverseDiffAdjoint")
flush(stdout)

println("DEBUG: All basic tests completed successfully - no hanging detected in early sections")
flush(stdout)