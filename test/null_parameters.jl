import OrdinaryDiffEq: ODEProblem, solve, Tsit5
import Zygote
using DiffEqSensitivity, Test

dynamics = (x, _p, _t) -> x

function loss(params)
    u0 = zeros(2)
    problem = ODEProblem(dynamics, u0, (0.0, 1.0), params)
    rollout = solve(problem, Tsit5(), u0 = u0, p = params, sensealg = InterpolatingAdjoint(autojacvec=ZygoteVJP(allow_nothing=true)))
    sum(Array(rollout)[:, end])
end

function loss2(params)
    u0 = zeros(2)
    problem = ODEProblem(dynamics, u0, (0.0, 1.0), params)
    rollout = solve(problem, Tsit5(), u0 = u0, p = params, sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP()))
    sum(Array(rollout)[:, end])
end

function loss3(params)
    u0 = zeros(2)
    problem = ODEProblem(dynamics, u0, (0.0, 1.0), params)
    rollout = solve(problem, Tsit5(), u0 = u0, p = params, sensealg = InterpolatingAdjoint(autojacvec=TrackerVJP(allow_nothing=true)))
    sum(Array(rollout)[:, end])
end

function loss4(params)
    u0 = zeros(2)
    problem = ODEProblem(dynamics, u0, (0.0, 1.0))
    rollout = solve(problem, Tsit5(), u0 = u0, p = params, sensealg = InterpolatingAdjoint(autojacvec=ZygoteVJP(allow_nothing=true)))
    sum(Array(rollout)[:, end])
end

function loss5(params)
    u0 = zeros(2)
    problem = ODEProblem(dynamics, u0, (0.0, 1.0))
    rollout = solve(problem, Tsit5(), u0 = u0, p = params, sensealg = InterpolatingAdjoint(autojacvec=EnzymeVJP()))
    sum(Array(rollout)[:, end])
end

function loss6(params)
    u0 = zeros(2)
    problem = ODEProblem(dynamics, u0, (0.0, 1.0))
    rollout = solve(problem, Tsit5(), u0 = u0, p = params, sensealg = BacksolveAdjoint(autojacvec=ZygoteVJP(allow_nothing=true)))
    sum(Array(rollout)[:, end])
end

function loss7(params)
    u0 = zeros(2)
    problem = ODEProblem(dynamics, u0, (0.0, 1.0))
    rollout = solve(problem, Tsit5(), u0 = u0, p = params, sensealg = QuadratureAdjoint(autojacvec=ZygoteVJP(allow_nothing=true)))
    sum(Array(rollout)[:, end])
end

function loss8(params)
    u0 = zeros(2)
    problem = ODEProblem(dynamics, u0, (0.0, 1.0))
    rollout = solve(problem, Tsit5(), u0 = u0, p = params, sensealg = QuadratureAdjoint(autojacvec=ReverseDiffVJP()))
    sum(Array(rollout)[:, end])
end

function loss9(params)
    u0 = zeros(2)
    problem = ODEProblem(dynamics, u0, (0.0, 1.0))
    rollout = solve(problem, Tsit5(), u0 = u0, p = params, sensealg = QuadratureAdjoint(autojacvec=EnzymeVJP()))
    sum(Array(rollout)[:, end])
end

function loss10(params)
    u0 = zeros(2)
    problem = ODEProblem(dynamics, u0, (0.0, 1.0))
    rollout = solve(problem, Tsit5(), u0 = u0, p = params)
    sum(Array(rollout)[:, end])
end

@test Zygote.gradient(dynamics, 0.0, nothing, nothing) == (1.0,nothing,nothing)

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
@test_throws DiffEqSensitivity.ZygoteVJPNothingError Zygote.gradient(loss10, zeros(123))[1] == zeros(123)
