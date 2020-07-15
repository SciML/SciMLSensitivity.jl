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
