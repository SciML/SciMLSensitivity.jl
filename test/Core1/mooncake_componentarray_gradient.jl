# Test for issue #1548: Mooncake gradient of an ODE solve crashed when u0 was a
# ComponentArray (`vec(::ChainRulesCore.ZeroTangent)` MethodError).
# https://github.com/SciML/SciMLSensitivity.jl/issues/1548

using ComponentArrays
using OrdinaryDiffEq
using SciMLSensitivity
using Mooncake
using DifferentiationInterface
using Test

function f!(du, u, p, t)
    @. du = p - u
    return nothing
end

const u0 = ComponentArray(a = [0.0, 0.1])
const tspan = (0.0, 1.0)
const backend = AutoMooncake(; config = nothing)

function loss(p)
    prob = ODEProblem(f!, u0, tspan, p)
    sol = solve(prob, Tsit5())
    return sum(sol)
end

p = [0.5, 0.2]

prep = prepare_gradient(loss, backend, p)
grad = DifferentiationInterface.gradient(loss, prep, backend, p)

@test all(isfinite, grad)
@test length(grad) == length(p)
