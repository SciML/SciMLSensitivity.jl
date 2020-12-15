using OrdinaryDiffEq, DiffEqSensitivity, Zygote

tspan = (0., 1.)
X = randn(3, 4)
p = randn(3, 4)
f(u,p,t) = u .* p
f(du,u,p,t) = (du .= u .* p)
prob_ube = ODEProblem{false}(f, X, tspan, p)
Zygote.gradient(p->sum(solve(prob_ube, Midpoint(), u0 = X, p = p)),p)

prob_ube = ODEProblem{true}(f, X, tspan, p)
Zygote.gradient(p->sum(solve(prob_ube, Midpoint(), u0 = X, p = p)),p)

function aug_dynamics!(dz, z, K, t)
    x = @view z[2:end]
    u = -K * x
    dz[1] = x' * x + u' * u
    dz[2:end] = x + u
end

policy_params = ones(2, 2)
z0 = zeros(3)
fwd_sol = solve(
    ODEProblem(aug_dynamics!, z0, (0.0, 1.0), policy_params),
    Tsit5(),
    u0 = z0,
    p = policy_params)

solve(
    ODEAdjointProblem(
        fwd_sol,
        InterpolatingAdjoint(),
        (out, x, p, t, i) -> (out .= 1),
        [1.0],
    ),Tsit5()
)

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
fwd_sol = solve(
    ODEProblem(aug_dynamics!, z0, (0.0, 1.0), policy_params),
    u0 = z0,
    p = policy_params,
)

solve(
    ODEAdjointProblem(
        fwd_sol,
        InterpolatingAdjoint(),
        (out, x, p, t, i) -> (out .= 1),
        [1.0],
    ),
)
