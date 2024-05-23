# testing `solve`-sensitivities with different AD frameworks on the simplest physical ODE: a mass accelerated by gravity

using OrdinaryDiffEq
using ForwardDiff, ReverseDiff, Zygote, FiniteDiff
using Test, SciMLSensitivity

# A falling mass (without contact, just gravity)
GRAVITY = 9.81
MASS = 1.0
NUM_STATES = 2

t_start = 0.0
t_step = 0.05
t_stop = 2.0
u0 = [1.0, 0.0] # start state: mass position (1.0) and velocity (0.0)
p = [GRAVITY, MASS]

# setup falling mass ODE 
function fx(u, p, t)
    g, m = p
    return [u[2], -g]
end

ff = ODEFunction{false}(fx)
prob = ODEProblem{false}(ff, u0, (t_start, t_stop), p)

function mysolve(p; solver = nothing)
    solution = solve(prob; p = p, alg = solver, saveat = t_start:t_step:t_stop)

    us = solution

    # fix for ReverseDiff not returning an ODESolution
    if !isa(us, ReverseDiff.TrackedArray)
        us = collect(u[1] for u in solution.u)
    else
        us = solution[1, :]
    end

    return us
end

analyt_sol = [-27.675, 0.0]
atol = 1e-2

solvers = [Tsit5(), Rosenbrock23(autodiff = false), Rosenbrock23(autodiff = true)]
for solver in solvers
    loss = (p) -> sum(mysolve(p; solver = solver))
    @test isapprox(FiniteDiff.finite_difference_gradient(loss, p), analyt_sol; atol = atol)
    @test isapprox(ForwardDiff.gradient(loss, p), analyt_sol; atol = atol)
    @test isapprox(Zygote.gradient(loss, p)[1], analyt_sol; atol = atol)
    @test isapprox(ReverseDiff.gradient(loss, p), analyt_sol; atol = atol)
end
