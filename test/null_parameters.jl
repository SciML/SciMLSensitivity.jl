import OrdinaryDiffEq: ODEProblem, solve, Tsit5
import Zygote
import ForwardDiff
using SciMLSensitivity, Test

dynamics = (x, _p, _t) -> x

function loss(params)
    u0 = zeros(2)
    problem = ODEProblem(dynamics, u0, (0.0, 1.0), params)
    rollout = solve(problem, Tsit5(), u0 = u0, p = params,
        sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP(allow_nothing = true)))
    sum(Array(rollout)[:, end])
end

function loss2(params)
    u0 = zeros(2)
    problem = ODEProblem(dynamics, u0, (0.0, 1.0), params)
    rollout = solve(problem, Tsit5(), u0 = u0, p = params,
        sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP()))
    sum(Array(rollout)[:, end])
end

function loss3(params)
    u0 = zeros(2)
    problem = ODEProblem(dynamics, u0, (0.0, 1.0), params)
    rollout = solve(problem, Tsit5(), u0 = u0, p = params,
        sensealg = InterpolatingAdjoint(autojacvec = TrackerVJP(allow_nothing = true)))
    sum(Array(rollout)[:, end])
end

function loss4(params)
    u0 = zeros(2)
    problem = ODEProblem(dynamics, u0, (0.0, 1.0))
    rollout = solve(problem, Tsit5(), u0 = u0, p = params,
        sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP(allow_nothing = true)))
    sum(Array(rollout)[:, end])
end

function loss5(params)
    u0 = zeros(2)
    problem = ODEProblem(dynamics, u0, (0.0, 1.0))
    rollout = solve(problem, Tsit5(), u0 = u0, p = params,
        sensealg = InterpolatingAdjoint(autojacvec = EnzymeVJP()))
    sum(Array(rollout)[:, end])
end

function loss6(params)
    u0 = zeros(2)
    problem = ODEProblem(dynamics, u0, (0.0, 1.0))
    rollout = solve(problem, Tsit5(), u0 = u0, p = params,
        sensealg = BacksolveAdjoint(autojacvec = ZygoteVJP(allow_nothing = true)))
    sum(Array(rollout)[:, end])
end

function loss7(params)
    u0 = zeros(2)
    problem = ODEProblem(dynamics, u0, (0.0, 1.0))
    rollout = solve(problem, Tsit5(), u0 = u0, p = params,
        sensealg = QuadratureAdjoint(autojacvec = ZygoteVJP(allow_nothing = true)))
    sum(Array(rollout)[:, end])
end

function loss8(params)
    u0 = zeros(2)
    problem = ODEProblem(dynamics, u0, (0.0, 1.0))
    rollout = solve(problem, Tsit5(), u0 = u0, p = params,
        sensealg = QuadratureAdjoint(autojacvec = ReverseDiffVJP()))
    sum(Array(rollout)[:, end])
end

function loss9(params)
    u0 = zeros(2)
    problem = ODEProblem(dynamics, u0, (0.0, 1.0))
    rollout = solve(problem, Tsit5(), u0 = u0, p = params,
        sensealg = QuadratureAdjoint(autojacvec = EnzymeVJP()))
    sum(Array(rollout)[:, end])
end

function loss10(params)
    u0 = zeros(2)
    problem = ODEProblem(dynamics, u0, (0.0, 1.0))
    rollout = solve(problem, Tsit5(), u0 = u0, p = params)
    sum(Array(rollout)[:, end])
end

@test Zygote.gradient(dynamics, 0.0, nothing, nothing) == (1.0, nothing, nothing)

@test Zygote.gradient(loss, nothing)[1] === nothing
@test_broken Zygote.gradient(loss2, nothing)
@test_broken Zygote.gradient(loss3, nothing)
@test Zygote.gradient(loss4, nothing)[1] === nothing
@test Zygote.gradient(loss5, nothing)[1] === nothing
@test Zygote.gradient(loss6, nothing)[1] === nothing
@test Zygote.gradient(loss7, nothing)[1] === nothing
@test Zygote.gradient(loss8, nothing)[1] === nothing
@test Zygote.gradient(loss9, nothing)[1] === nothing
@test Zygote.gradient(loss10, nothing)[1] === nothing

@test Zygote.gradient(loss, zeros(123))[1] == zeros(123)
@test Zygote.gradient(loss2, zeros(123))[1] == zeros(123)
@test Zygote.gradient(loss3, zeros(123))[1] == zeros(123)
@test Zygote.gradient(loss4, zeros(123))[1] == zeros(123)
@test Zygote.gradient(loss5, zeros(123))[1] == zeros(123)
@test Zygote.gradient(loss6, zeros(123))[1] == zeros(123)
@test_broken Zygote.gradient(loss7, zeros(123))[1] == zeros(123)
@test Zygote.gradient(loss8, zeros(123))[1] == zeros(123)
@test Zygote.gradient(loss9, zeros(123))[1] == zeros(123)
@test_throws SciMLSensitivity.ZygoteVJPNothingError Zygote.gradient(loss10,
    zeros(123))[1]==zeros(123)

## OOP tests for initial condition
function loss_oop(u0; sensealg = nothing)
    _prob = ODEProblem(dynamics, u0, (0.0, 1.0))
    _sol = solve(_prob, Tsit5(), u0 = u0, sensealg = sensealg, abstol = 1e-12,
        reltol = 1e-12)
    sum(abs2, Array(_sol)[:, end])
end

u0 = ones(2)
Fdu0 = ForwardDiff.gradient(u0 -> loss_oop(u0), u0)

# BacksolveAdjoint
du0 = Zygote.gradient(
    u0 -> loss_oop(u0,
        sensealg = BacksolveAdjoint(autojacvec = ZygoteVJP(allow_nothing = true))),
    u0)[1]
@test_throws SciMLSensitivity.ZygoteVJPNothingError Zygote.gradient(
    u0 -> loss_oop(u0,
        sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP())),
    u0)[1]
@test Fdu0≈du0 rtol=1e-10
du0 = Zygote.gradient(
    u0 -> loss_oop(u0,
        sensealg = BacksolveAdjoint(autojacvec = ReverseDiffVJP())),
    u0)[1]
@test Fdu0≈du0 rtol=1e-10
du0 = Zygote.gradient(u0 -> loss_oop(u0, sensealg = BacksolveAdjoint(autojacvec = false)),
    u0)[1]
@test Fdu0≈du0 rtol=1e-10

# InterpolatingAdjoint
du0 = Zygote.gradient(
    u0 -> loss_oop(u0,
        sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP(allow_nothing = true))),
    u0)[1]
@test_throws SciMLSensitivity.ZygoteVJPNothingError Zygote.gradient(
    u0 -> loss_oop(u0,
        sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP())),
    u0)[1]
@test Fdu0≈du0 rtol=1e-10
du0 = Zygote.gradient(
    u0 -> loss_oop(u0,
        sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP())),
    u0)[1]
@test Fdu0≈du0 rtol=1e-10
du0 = Zygote.gradient(
    u0 -> loss_oop(u0,
        sensealg = InterpolatingAdjoint(autojacvec = false)),
    u0)[1]
@test Fdu0≈du0 rtol=1e-10

# QuadratureAdjoint
du0 = Zygote.gradient(
    u0 -> loss_oop(u0,
        sensealg = QuadratureAdjoint(autojacvec = ZygoteVJP(allow_nothing = true))),
    u0)[1]
@test_throws SciMLSensitivity.ZygoteVJPNothingError Zygote.gradient(
    u0 -> loss_oop(u0,
        sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP())),
    u0)[1]
@test Fdu0≈du0 rtol=1e-10
du0 = Zygote.gradient(
    u0 -> loss_oop(u0,
        sensealg = QuadratureAdjoint(autojacvec = ReverseDiffVJP())),
    u0)[1]
@test Fdu0≈du0 rtol=1e-10
du0 = Zygote.gradient(u0 -> loss_oop(u0, sensealg = QuadratureAdjoint(autojacvec = false)),
    u0)[1]
@test Fdu0≈du0 rtol=1e-10

# ForwardDiffSensitivity
du0 = Zygote.gradient(u0 -> loss_oop(u0, sensealg = ForwardDiffSensitivity()), u0)[1]
@test Fdu0≈du0 rtol=1e-6
du0 = Zygote.gradient(u0 -> loss_oop(u0, sensealg = ForwardDiffSensitivity()), u0)[1]
@test Fdu0≈du0 rtol=1e-6
du0 = Zygote.gradient(u0 -> loss_oop(u0, sensealg = ForwardDiffSensitivity()), u0)[1]
@test Fdu0≈du0 rtol=1e-6

## iip tests for initial condition
dynamics! = (dx, x, _p, _t) -> dx .= x

function loss_iip(u0; sensealg = nothing)
    _prob = ODEProblem(dynamics!, u0, (0.0, 1.0))
    _sol = solve(_prob, Tsit5(), u0 = u0, sensealg = sensealg, abstol = 1e-12,
        reltol = 1e-12)
    sum(abs2, Array(_sol)[:, end])
end

u0 = ones(2)
Fdu0 = ForwardDiff.gradient(u0 -> loss_iip(u0), u0)

# BacksolveAdjoint
du0 = Zygote.gradient(
    u0 -> loss_iip(u0,
        sensealg = BacksolveAdjoint(autojacvec = ZygoteVJP(allow_nothing = true))),
    u0)[1]
@test_throws SciMLSensitivity.ZygoteVJPNothingError Zygote.gradient(
    u0 -> loss_iip(u0,
        sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP())),
    u0)[1]
@test Fdu0≈du0 rtol=1e-10
du0 = Zygote.gradient(
    u0 -> loss_iip(u0,
        sensealg = BacksolveAdjoint(autojacvec = ReverseDiffVJP())),
    u0)[1]
@test Fdu0≈du0 rtol=1e-10
du0 = Zygote.gradient(u0 -> loss_iip(u0, sensealg = BacksolveAdjoint(autojacvec = false)),
    u0)[1]
@test Fdu0≈du0 rtol=1e-10
du0 = Zygote.gradient(
    u0 -> loss_iip(u0,
        sensealg = BacksolveAdjoint(autojacvec = EnzymeVJP())),
    u0)[1]
@test Fdu0≈du0 rtol=1e-10

# InterpolatingAdjoint
du0 = Zygote.gradient(
    u0 -> loss_iip(u0,
        sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP(allow_nothing = true))),
    u0)[1]
@test_throws SciMLSensitivity.ZygoteVJPNothingError Zygote.gradient(
    u0 -> loss_iip(u0,
        sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP())),
    u0)[1]
@test Fdu0≈du0 rtol=1e-10
du0 = Zygote.gradient(
    u0 -> loss_iip(u0,
        sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP())),
    u0)[1]
@test Fdu0≈du0 rtol=1e-10
du0 = Zygote.gradient(
    u0 -> loss_iip(u0,
        sensealg = InterpolatingAdjoint(autojacvec = false)),
    u0)[1]
@test Fdu0≈du0 rtol=1e-10
du0 = Zygote.gradient(
    u0 -> loss_iip(u0,
        sensealg = InterpolatingAdjoint(autojacvec = EnzymeVJP())),
    u0)[1]
@test Fdu0≈du0 rtol=1e-10

# QuadratureAdjoint
du0 = Zygote.gradient(
    u0 -> loss_iip(u0,
        sensealg = QuadratureAdjoint(autojacvec = ZygoteVJP(allow_nothing = true))),
    u0)[1]
@test_throws SciMLSensitivity.ZygoteVJPNothingError Zygote.gradient(
    u0 -> loss_iip(u0,
        sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP())),
    u0)[1]
@test Fdu0≈du0 rtol=1e-10
du0 = Zygote.gradient(
    u0 -> loss_iip(u0,
        sensealg = QuadratureAdjoint(autojacvec = ReverseDiffVJP())),
    u0)[1]
@test Fdu0≈du0 rtol=1e-10
du0 = Zygote.gradient(u0 -> loss_iip(u0, sensealg = QuadratureAdjoint(autojacvec = false)),
    u0)[1]
@test Fdu0≈du0 rtol=1e-10
du0 = Zygote.gradient(
    u0 -> loss_iip(u0,
        sensealg = QuadratureAdjoint(autojacvec = EnzymeVJP())),
    u0)[1]
@test Fdu0≈du0 rtol=1e-10

# ForwardDiffSensitivity
du0 = Zygote.gradient(u0 -> loss_iip(u0, sensealg = ForwardDiffSensitivity()), u0)[1]
@test Fdu0≈du0 rtol=1e-6
du0 = Zygote.gradient(u0 -> loss_iip(u0, sensealg = ForwardDiffSensitivity()), u0)[1]
@test Fdu0≈du0 rtol=1e-6
du0 = Zygote.gradient(u0 -> loss_iip(u0, sensealg = ForwardDiffSensitivity()), u0)[1]
@test Fdu0≈du0 rtol=1e-6
