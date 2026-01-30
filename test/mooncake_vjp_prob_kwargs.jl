# Test for issue #1337: Mooncake differentiation fails when kwargs (like saveat)
# are passed in ODEProblem constructor instead of solve()
# https://github.com/SciML/SciMLSensitivity.jl/issues/1337

using OrdinaryDiffEq
using SciMLSensitivity
using SciMLSensitivity: MooncakeVJP
using Mooncake
using DifferentiationInterface
using Test

function f!(du, u, p, t)
    return du[1] = p[1] * u[1]
end

const u0 = [1.0]
const tspan = (0.0, 1.0)
const backend = AutoMooncake(; config = nothing)

# Case 1: kwargs passed to solve() - this always worked
function test_kwargs_in_solve(p)
    prob = ODEProblem{true}(f!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat = 0.1, sensealg = BacksolveAdjoint(autojacvec = MooncakeVJP()))
    return sol.u[end][1]
end

# Case 2: kwargs baked into ODEProblem - this was failing before the fix
function test_kwargs_in_prob(p)
    prob = ODEProblem{true}(f!, u0, tspan, p; saveat = 0.1)
    sol = solve(prob, Tsit5(), sensealg = BacksolveAdjoint(autojacvec = MooncakeVJP()))
    return sol.u[end][1]
end

p = [0.5]

# Test Case 1: kwargs in solve()
prep1 = prepare_gradient(test_kwargs_in_solve, backend, p)
grad1 = DifferentiationInterface.gradient(test_kwargs_in_solve, prep1, backend, p)

# Test Case 2: kwargs in ODEProblem constructor (was failing with TypeError before fix)
prep2 = prepare_gradient(test_kwargs_in_prob, backend, p)
grad2 = DifferentiationInterface.gradient(test_kwargs_in_prob, prep2, backend, p)

# Both should give the same gradient
@test grad1[1] ≈ grad2[1] rtol = 1.0e-10
