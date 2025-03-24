using OrdinaryDiffEq, SciMLSensitivity, ForwardDiff, Zygote, ReverseDiff, Tracker, Enzyme,
      FiniteDiff, Mooncake
using Test
Enzyme.API.typeWarning!(false)

mooncake_gradient(f, x) = Mooncake.value_and_gradient!!(build_rrule(f, x), f, x)[2][2]

odef(du, u, p, t) = du .= u .* p
const prob = ODEProblem(odef, [2.0], (0.0, 1.0), [3.0])

struct senseloss0{T}
    sense::T
end
function (f::senseloss0)(u0p)
    prob = ODEProblem{true}(odef, u0p[1:1], (0.0, 1.0), u0p[2:2])
    sum(solve(prob, Tsit5(), abstol = 1e-12, reltol = 1e-12, saveat = 0.1))
end
u0p = [2.0, 3.0]
du0p = zeros(2)
dup = Zygote.gradient(senseloss0(InterpolatingAdjoint()), u0p)[1]
Enzyme.autodiff(Reverse, senseloss0(InterpolatingAdjoint()), Active, Duplicated(u0p, du0p))
dup_mc = mooncake_gradient(senseloss0(InterpolatingAdjoint()), u0p)
@test du0p ≈ dup
@test dup_mc ≈ dup

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

dup = Zygote.gradient(senseloss(InterpolatingAdjoint()), u0p)[1]

@test ReverseDiff.gradient(senseloss(InterpolatingAdjoint()), u0p) ≈ dup
@test ReverseDiff.gradient(senseloss(ReverseDiffAdjoint()), u0p) ≈ dup
@test ReverseDiff.gradient(senseloss(TrackerAdjoint()), u0p) ≈ dup
@test ReverseDiff.gradient(senseloss(ForwardDiffSensitivity()), u0p) ≈ dup
@test_broken ReverseDiff.gradient(senseloss(ForwardSensitivity()), u0p) ≈ dup # broken because ForwardSensitivity not compatible with perturbing u0

@test Tracker.gradient(senseloss(InterpolatingAdjoint()), u0p)[1] ≈ dup
@test Tracker.gradient(senseloss(ReverseDiffAdjoint()), u0p)[1] ≈ dup
@test Tracker.gradient(senseloss(TrackerAdjoint()), u0p)[1] ≈ dup
@test Tracker.gradient(senseloss(ForwardDiffSensitivity()), u0p)[1] ≈ dup
@test_broken Tracker.gradient(senseloss(ForwardSensitivity()), u0p)[1] ≈ dup # broken because ForwardSensitivity not compatible with perturbing u0

@test ForwardDiff.gradient(senseloss(InterpolatingAdjoint()), u0p) ≈ dup

@test only(Enzyme.gradient(Reverse, senseloss(InterpolatingAdjoint()), u0p)) ≈ dup
@test_throws SciMLSensitivity.EnzymeTrackedRealError only(Enzyme.gradient(
    Reverse, senseloss(ReverseDiffAdjoint()), u0p))≈dup
@test_throws SciMLSensitivity.EnzymeTrackedRealError only(Enzyme.gradient(
    Reverse, senseloss(TrackerAdjoint()), u0p))≈dup
@test only(Enzyme.gradient(Reverse, senseloss(ForwardDiffSensitivity()), u0p)) ≈ dup
@test_broken only(Enzyme.gradient(Reverse, senseloss(ForwardSensitivity()), u0p)) ≈ dup # broken because ForwardSensitivity not compatible with perturbing u0

@test mooncake_gradient(senseloss(InterpolatingAdjoint()), u0p) ≈ dup
@test_broken mooncake_gradient(senseloss(ReverseDiffAdjoint()), u0p) ≈ dup
@test_broken mooncake_gradient(senseloss(TrackerAdjoint()), u0p) ≈ dup
@test mooncake_gradient(senseloss(ForwardDiffSensitivity()), u0p) ≈ dup
@test_broken mooncake_gradient(senseloss(ForwardSensitivity()), u0p) ≈ dup # broken because ForwardSensitivity not compatible with perturbing u0

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

@test ReverseDiff.gradient(senseloss2(InterpolatingAdjoint()), u0p) ≈ dup
@test ReverseDiff.gradient(senseloss2(ReverseDiffAdjoint()), u0p) ≈ dup
@test ReverseDiff.gradient(senseloss2(TrackerAdjoint()), u0p) ≈ dup
@test ReverseDiff.gradient(senseloss2(ForwardDiffSensitivity()), u0p) ≈ dup
@test_broken ReverseDiff.gradient(senseloss2(ForwardSensitivity()), u0p) ≈ dup # broken because ForwardSensitivity not compatible with perturbing u0

@test Tracker.gradient(senseloss2(InterpolatingAdjoint()), u0p)[1] ≈ dup
@test Tracker.gradient(senseloss2(ReverseDiffAdjoint()), u0p)[1] ≈ dup
@test Tracker.gradient(senseloss2(TrackerAdjoint()), u0p)[1] ≈ dup
@test Tracker.gradient(senseloss2(ForwardDiffSensitivity()), u0p)[1] ≈ dup
@test_broken Tracker.gradient(senseloss2(ForwardSensitivity()), u0p)[1] ≈ dup # broken because ForwardSensitivity not compatible with perturbing u0

@test ForwardDiff.gradient(senseloss2(InterpolatingAdjoint()), u0p) ≈ dup

@test_broken only(Enzyme.gradient(Reverse, senseloss2(InterpolatingAdjoint()), u0p)) ≈ dup
@test_broken only(Enzyme.gradient(Reverse, senseloss2(ReverseDiffAdjoint()), u0p)) ≈ dup
@test_broken only(Enzyme.gradient(Reverse, senseloss2(TrackerAdjoint()), u0p)) ≈ dup
@test_broken only(Enzyme.gradient(Reverse, senseloss2(ForwardDiffSensitivity()), u0p)) ≈ dup
@test_broken only(Enzyme.gradient(Reverse, senseloss2(ForwardSensitivity()), u0p)) ≈ dup # broken because ForwardSensitivity not compatible with perturbing u0

@test mooncake_gradient(senseloss2(InterpolatingAdjoint()), u0p) ≈ dup
@test_broken mooncake_gradient(senseloss2(ReverseDiffAdjoint()), u0p) ≈ dup
@test_broken mooncake_gradient(senseloss2(TrackerAdjoint()), u0p) ≈ dup
@test mooncake_gradient(senseloss2(ForwardDiffSensitivity()), u0p) ≈ dup
@test_broken mooncake_gradient(senseloss2(ForwardSensitivity()), u0p) ≈ dup # broken because ForwardSensitivity not compatible with perturbing u0

struct senseloss3{T}
    sense::T
end
function (f::senseloss3)(u0p)
    sum(solve(prob2, Tsit5(), p = u0p, abstol = 1e-12,
        reltol = 1e-12, saveat = 0.1, sensealg = f.sense))
end

u0p = [3.0]

dup = Zygote.gradient(senseloss3(InterpolatingAdjoint()), u0p)[1]

@test ReverseDiff.gradient(senseloss3(InterpolatingAdjoint()), u0p) ≈ dup
@test ReverseDiff.gradient(senseloss3(ReverseDiffAdjoint()), u0p) ≈ dup
@test ReverseDiff.gradient(senseloss3(TrackerAdjoint()), u0p) ≈ dup
@test ReverseDiff.gradient(senseloss3(ForwardDiffSensitivity()), u0p) ≈ dup
@test ReverseDiff.gradient(senseloss3(ForwardSensitivity()), u0p) ≈ dup

@test Tracker.gradient(senseloss3(InterpolatingAdjoint()), u0p)[1] ≈ dup
@test Tracker.gradient(senseloss3(ReverseDiffAdjoint()), u0p)[1] ≈ dup
@test Tracker.gradient(senseloss3(TrackerAdjoint()), u0p)[1] ≈ dup
@test Tracker.gradient(senseloss3(ForwardDiffSensitivity()), u0p)[1] ≈ dup
@test Tracker.gradient(senseloss3(ForwardSensitivity()), u0p)[1] ≈ dup

@test ForwardDiff.gradient(senseloss3(InterpolatingAdjoint()), u0p) ≈ dup

@test_broken only(Enzyme.gradient(Reverse, senseloss3(InterpolatingAdjoint()), u0p)) ≈ dup
@test_broken only(Enzyme.gradient(Reverse, senseloss3(ReverseDiffAdjoint()), u0p)) ≈ dup
@test_broken only(Enzyme.gradient(Reverse, senseloss3(TrackerAdjoint()), u0p)) ≈ dup
@test_broken only(Enzyme.gradient(Reverse, senseloss3(ForwardDiffSensitivity()), u0p)) ≈ dup
@test_broken only(Enzyme.gradient(Reverse, senseloss3(ForwardSensitivity()), u0p)) ≈ dup

@test mooncake_gradient(senseloss3(InterpolatingAdjoint()), u0p) ≈ dup
@test_broken mooncake_gradient(senseloss3(ReverseDiffAdjoint()), u0p) ≈ dup
@test_broken mooncake_gradient(senseloss3(TrackerAdjoint()), u0p) ≈ dup
@test mooncake_gradient(senseloss3(ForwardDiffSensitivity()), u0p) ≈ dup
@test_broken mooncake_gradient(senseloss3(ForwardSensitivity()), u0p) ≈ dup

struct senseloss4{T}
    sense::T
end
function (f::senseloss4)(u0p)
    sum(solve(prob, Tsit5(), p = u0p, abstol = 1e-12,
        reltol = 1e-12, saveat = 0.1, sensealg = f.sense))
end

u0p = [3.0]

dup = Zygote.gradient(senseloss4(InterpolatingAdjoint()), u0p)[1]

@test ReverseDiff.gradient(senseloss4(InterpolatingAdjoint()), u0p) ≈ dup
@test ReverseDiff.gradient(senseloss4(ReverseDiffAdjoint()), u0p) ≈ dup
@test ReverseDiff.gradient(senseloss4(TrackerAdjoint()), u0p) ≈ dup
@test ReverseDiff.gradient(senseloss4(ForwardDiffSensitivity()), u0p) ≈ dup
@test ReverseDiff.gradient(senseloss4(ForwardSensitivity()), u0p) ≈ dup

@test Tracker.gradient(senseloss4(InterpolatingAdjoint()), u0p)[1] ≈ dup
@test Tracker.gradient(senseloss4(ReverseDiffAdjoint()), u0p)[1] ≈ dup
@test Tracker.gradient(senseloss4(TrackerAdjoint()), u0p)[1] ≈ dup
@test Tracker.gradient(senseloss4(ForwardDiffSensitivity()), u0p)[1] ≈ dup
@test Tracker.gradient(senseloss4(ForwardSensitivity()), u0p)[1] ≈ dup

@test ForwardDiff.gradient(senseloss4(InterpolatingAdjoint()), u0p) ≈ dup

@test only(Enzyme.gradient(Reverse, senseloss4(InterpolatingAdjoint()), u0p)) ≈ dup
@test_throws SciMLSensitivity.EnzymeTrackedRealError only(Enzyme.gradient(
    Reverse, senseloss4(ReverseDiffAdjoint()), u0p))≈dup
@test_throws SciMLSensitivity.EnzymeTrackedRealError only(Enzyme.gradient(
    Reverse, senseloss4(TrackerAdjoint()), u0p))≈dup
@test only(Enzyme.gradient(Reverse, senseloss4(ForwardDiffSensitivity()), u0p)) ≈ dup
@test_broken only(Enzyme.gradient(Reverse, senseloss4(ForwardSensitivity()), u0p)) ≈ dup # broken because ForwardSensitivity not compatible with perturbing u0

@test mooncake_gradient(senseloss4(InterpolatingAdjoint()), u0p) ≈ dup
@test_broken mooncake_gradient(senseloss4(ReverseDiffAdjoint()), u0p) ≈ dup
@test_broken mooncake_gradient(senseloss4(TrackerAdjoint()), u0p) ≈ dup
@test mooncake_gradient(senseloss4(ForwardDiffSensitivity()), u0p) ≈ dup
@test_broken mooncake_gradient(senseloss4(ForwardSensitivity()), u0p) ≈ dup

solvealg_test = Tsit5()
sensealg_test = InterpolatingAdjoint()
tspan = (0.0, 1.0)
u0 = rand(4, 8)
p0 = rand(16)
f_aug(u, p, t) = reshape(p, 4, 4) * u

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

res1 = loss(p0)
res2 = ReverseDiff.gradient(loss, p0)
res3 = loss2(p0)
res4 = ReverseDiff.gradient(loss2, p0)

@test res1≈res3 atol=1e-14
@test res2≈res4 atol=1e-14

@test_broken res2≈Enzyme.gradient(Reverse, loss, p0) atol=1e-14
@test_broken res4≈Enzyme.gradient(Reverse, loss2, p0) atol=1e-14

# I think we're just not successfully hitting the rrule here.
@test_broken res2 ≈ mooncake_gradient(loss, p0)
res4 ≈ mooncake_gradient(loss2, p0)

# Test for recursion https://discourse.julialang.org/t/diffeqsensitivity-jl-issues-with-reversediffadjoint-sensealg/88774
function ode!(derivative, state, parameters, t)
    derivative .= parameters
end

function ode(state, parameters, t)
    return ode!(similar(state), state, parameters, t)
end

function solve_euler(state, times, parameters)
    problem = ODEProblem{true}(ode!, state, times[[1, end]], parameters; saveat = times,
        sensealg = ReverseDiffAdjoint())
    return solve(problem, Euler(); dt = 1e-1)
end

const initial_state = ones(2)
const solution_times = [1.0, 2.0]
ReverseDiff.gradient(p -> sum(sum(solve_euler(initial_state, solution_times, p))), zeros(2))
# Enzyme.gradient(Reverse, p -> sum(sum(solve_euler(initial_state, solution_times, p))), zeros(2))
# mooncake_gradient(p -> sum(sum(solve_euler(initial_state, solution_times, p))), zeros(2))

# https://github.com/SciML/SciMLSensitivity.jl/issues/943

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

grad_fi = FiniteDiff.finite_difference_gradient(loss2, p)
grad_fd = ForwardDiff.gradient(loss2, p)
grad_zg = Zygote.gradient(loss2, p)[1]
grad_rd = ReverseDiff.gradient(loss2, p)
@test grad_fd≈grad_fi atol=1e-2
@test grad_fd ≈ grad_zg atol=1e-4
@test grad_fd ≈ grad_rd atol=1e-4
@test_broken mooncake_gradient(loss2, p) ≈ grad_rd atol=1e-4 # appears to not be hitting the rule
