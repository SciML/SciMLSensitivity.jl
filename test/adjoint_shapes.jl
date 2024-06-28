using OrdinaryDiffEq, SciMLSensitivity, Zygote, ForwardDiff
using SciMLStructures

tspan = (0.0, 1.0)
X = randn(3, 4)
p = randn(3, 4)
f(u, p, t) = u .* p
f(du, u, p, t) = (du .= u .* p)
prob_ube = ODEProblem{false}(f, X, tspan, p)
Zygote.gradient(p -> sum(solve(prob_ube, Midpoint(), u0 = X, p = p)), p)

prob_ube = ODEProblem{true}(f, X, tspan, p)
Zygote.gradient(p -> sum(solve(prob_ube, Midpoint(), u0 = X, p = p)), p)

function aug_dynamics!(dz, z, K, t)
    x = @view z[2:end]
    u = -K * x
    dz[1] = x' * x + u' * u
    dz[2:end] = x + u
end

policy_params = ones(2, 2)
z0 = zeros(3)
fwd_sol = solve(ODEProblem(aug_dynamics!, z0, (0.0, 1.0), policy_params),
    Tsit5(), u0 = z0, p = policy_params)
_, repack, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), policy_params)

sensealg = InterpolatingAdjoint()
sensealg = SciMLSensitivity.setvjp(sensealg,
    SciMLSensitivity.inplace_vjp(
        fwd_sol.prob, fwd_sol.prob.u0, fwd_sol.prob.p, true, repack))

solve(
    ODEAdjointProblem(fwd_sol, sensealg, Tsit5(),
        [1.0], (out, x, p, t, i) -> (out .= 1)),
    Tsit5())

A = ones(2, 2)
B = ones(2, 2)
Q = ones(2, 2)
R = ones(2, 2)

function aug_dynamics!(dz, z, K, t)
    x = @view z[2:end]
    u = -K * x
    dz[1] = x' * Q * x + u' * R * u
    dz[2:end] = A * x + B * u # or just `x + u`
end

policy_params = ones(2, 2)
z0 = zeros(3)
fwd_sol = solve(ODEProblem(aug_dynamics!, z0, (0.0, 1.0), policy_params), u0 = z0,
    p = policy_params, Tsit5())

solve(
    ODEAdjointProblem(fwd_sol, sensealg, Tsit5(), [1.0],
        (out, x, p, t, i) -> (out .= 1)),
    Tsit5())

# https://github.com/SciML/SciMLSensitivity.jl/issues/581

p = rand(1)

function dudt(u, p, t)
    u .* p
end

function loss(p)
    prob = ODEProblem(dudt, [3.0], (0.0, 1.0), p)
    sol = solve(prob, Tsit5(), dt = 0.01, sensealg = ReverseDiffAdjoint())
    sum(abs2, Array(sol))
end
Zygote.gradient(loss, p)[1][1] ≈ ForwardDiff.gradient(loss, p)[1]
