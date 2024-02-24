using OrdinaryDiffEq, SciMLSensitivity, Zygote, RecursiveArrayTools, Test

u0 = Float32[1.0; 2.0]
du0 = Float32[0.0; 2.0]
tspan = (0.0f0, 1.0f0)
t = range(tspan[1], tspan[2], length = 20)
p = Float32[1.01, 0.9]
ff(du, u, p, t) = -p .* u
prob = SecondOrderODEProblem{false}(ff, du0, u0, tspan, p)
ddu01, du01, dp1 = Zygote.gradient(
    (du0, u0, p) -> sum(Array(solve(prob, Tsit5(),
        u0 = ArrayPartition(du0,
            u0),
        p = p, saveat = t,
        sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP())))),
    du0, u0, p)
ddu02, du02, dp2 = Zygote.gradient(
    (du0, u0, p) -> sum(Array(solve(prob, Tsit5(),
        u0 = ArrayPartition(du0,
            u0),
        p = p, saveat = t,
        sensealg = BacksolveAdjoint(autojacvec = ZygoteVJP())))),
    du0, u0, p)
ddu03, du03, dp3 = Zygote.gradient(
    (du0, u0, p) -> sum(Array(solve(prob, Tsit5(),
        u0 = ArrayPartition(du0,
            u0),
        p = p, saveat = t,
        sensealg = QuadratureAdjoint(autojacvec = ZygoteVJP())))),
    du0, u0, p)
ddu04, du04, dp4 = Zygote.gradient(
    (du0, u0, p) -> sum(Array(solve(prob, Tsit5(),
        u0 = ArrayPartition(du0,
            u0),
        p = p, saveat = t,
        sensealg = ForwardDiffSensitivity()))),
    du0, u0, p)
@test ddu01 ≈ ddu02
@test ddu01 ≈ ddu03
@test ddu01 ≈ ddu04
@test du01 ≈ du02
@test du01 ≈ du03
@test du01 ≈ du04
@test dp1 ≈ dp2
@test dp1 ≈ dp3
@test dp1 ≈ dp4
