using OrdinaryDiffEq, SciMLSensitivity, ForwardDiff, Zygote, ReverseDiff, Tracker, Enzyme
using Test
Enzyme.API.typeWarning!(false)

println("=== MATRIX SECTION DEBUG ===")
flush(stdout)

solvealg_test = Tsit5()
sensealg_test = InterpolatingAdjoint()
tspan = (0.0, 1.0)
u0 = rand(4, 8)
p0 = rand(16)
f_aug(u, p, t) = reshape(p, 4, 4) * u

println("Setup complete, u0 size: ", size(u0), ", p0 size: ", size(p0))
flush(stdout)

function loss(p)
    prob = ODEProblem(f_aug, u0, tspan, p; alg = solvealg_test, sensealg = sensealg_test)
    sol = solve(prob)
    sum(sol[:, :, end])
end

function loss2(p)
    prob = ODEProblem(f_aug, u0, tspan, p)
    sol = solve(prob, solvealg_test; sensealg = sensealg_test)
    sum(sol[:, :, end])
end

println("=== TESTING: loss(p0) ===")
flush(stdout)
res1 = loss(p0)
println("res1 = ", res1)
flush(stdout)

println("=== TESTING: ReverseDiff.gradient(loss, p0) ===")
flush(stdout)
res2 = ReverseDiff.gradient(loss, p0)
println("res2 computed, size: ", size(res2))
flush(stdout)

println("=== TESTING: loss2(p0) ===")
flush(stdout)
res3 = loss2(p0)
println("res3 = ", res3)
flush(stdout)

println("=== TESTING: ReverseDiff.gradient(loss2, p0) ===")
flush(stdout)
res4 = ReverseDiff.gradient(loss2, p0)
println("res4 computed, size: ", size(res4))
flush(stdout)

@test res1≈res3 atol=1e-14
@test res2≈res4 atol=1e-14

println("=== TESTING: Enzyme.gradient(Reverse, loss, p0) - this might hang ===")
flush(stdout)
# This is marked @test_broken in the original, so it might be the culprit
try
    enzyme_res = Enzyme.gradient(Reverse, loss, p0)
    println("Enzyme gradient succeeded: ", size(enzyme_res))
    flush(stdout)
catch e
    println("Enzyme gradient failed with error: ", e)
    flush(stdout)
end

println("=== TESTING: Enzyme.gradient(Reverse, loss2, p0) - this might also hang ===")
flush(stdout)
try
    enzyme_res2 = Enzyme.gradient(Reverse, loss2, p0)
    println("Enzyme gradient 2 succeeded: ", size(enzyme_res2))
    flush(stdout)
catch e
    println("Enzyme gradient 2 failed with error: ", e)
    flush(stdout)
end

println("=== MATRIX SECTION COMPLETED SUCCESSFULLY ===")
flush(stdout)