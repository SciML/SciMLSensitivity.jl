using OrdinaryDiffEq, SciMLSensitivity, ForwardDiff, Zygote, ReverseDiff, FiniteDiff
using Test

println("DEBUG: Starting bouncing ball section")
flush(stdout)

GRAVITY = 9.81
MASS = 1.0
NUM_STATES = 2

t_start = 0.0
t_step = 0.05
t_stop = 2.0
tData = t_start:t_step:t_stop
u0 = [1.0, 0.0] # start state: ball position (1.0) and velocity (0.0)
p = [GRAVITY, MASS]

# setup BouncingBallODE
function fx(u, p, t)
    g, m = p
    return [u[2], -g]
end

ff = ODEFunction{false}(fx)
prob3 = ODEProblem{false}(ff, u0, (t_start, t_stop), p)

function loss2(p)
    solution = solve(prob3; p = p, alg = solver, saveat = tData,
        sensealg = sensealg, abstol = 1e-10, reltol = 1e-10)
    # fix for ReverseDiff
    if !isa(solution, ReverseDiff.TrackedArray) && !isa(solution, Array)
        sum(abs.(collect(u[1] for u in solution.u)))
    else
        sum(abs.(solution[1, :]))
    end
end

solver = Rosenbrock23(autodiff = false)
sensealg = ReverseDiffAdjoint()

println("DEBUG: Defined loss function, about to compute gradients")
flush(stdout)

println("DEBUG: Computing FiniteDiff gradient")
flush(stdout)
grad_fi = FiniteDiff.finite_difference_gradient(loss2, p)
println("DEBUG: Completed FiniteDiff gradient: ", grad_fi)
flush(stdout)

println("DEBUG: Computing ForwardDiff gradient")
flush(stdout)
grad_fd = ForwardDiff.gradient(loss2, p)
println("DEBUG: Completed ForwardDiff gradient: ", grad_fd)
flush(stdout)

println("DEBUG: Computing Zygote gradient")
flush(stdout)
grad_zg = Zygote.gradient(loss2, p)[1]
println("DEBUG: Completed Zygote gradient: ", grad_zg)
flush(stdout)

println("DEBUG: Computing ReverseDiff gradient")
flush(stdout)
grad_rd = ReverseDiff.gradient(loss2, p)
println("DEBUG: Completed ReverseDiff gradient: ", grad_rd)
flush(stdout)

println("DEBUG: All gradients computed successfully!")
flush(stdout)