# Handling Exogenous Input Signals

The key to using exogenous input signals is the same as in the rest of the
SciML universe: just use the function in the definition of the differential
equation. For example, if it's a standard differential equation, you can
use the form

```julia
I(t) = t^2

function f(du, u, p, t)
    du[1] = I(t)
    du[2] = u[1]
end
```

so that `I(t)` is an exogenous input signal into `f`. Another form that could be
useful is a closure. For example:

```julia
function f(du, u, p, t, I)
    du[1] = I(t)
    du[2] = u[1]
end

_f(du, u, p, t) = f(du, u, p, t, x -> x^2)
```

which encloses an extra argument into `f` so that `_f` is now the interface-compliant
differential equation definition.

Note that you can also learn what the exogenous equation is from data. For an
example on how to do this, you can use the [Optimal Control Example](@ref optcontrol),
which shows how to parameterize a `u(t)` by a universal function and learn that
from data.

## Example of a Neural ODE with Exogenous Input

In the following example, a discrete exogenous input signal `ex` is defined and
used as an input into the neural network of a neural ODE system.

```@example exogenous
using SciMLSensitivity
using OrdinaryDiffEq, Lux, ComponentArrays, DiffEqFlux, Optimization,
      OptimizationPolyalgorithms, OptimizationOptimisers, Plots, Random

rng = Random.default_rng()
tspan = (0.1f0, Float32(10.0))
tsteps = range(tspan[1], tspan[2], length = 100)
t_vec = collect(tsteps)
ex = vec(ones(Float32, length(tsteps), 1))
f(x) = (atan(8.0 * x - 4.0) + atan(4.0)) / (2.0 * atan(4.0))

function hammerstein_system(u)
    y = zeros(size(u))
    for k in 2:length(u)
        y[k] = 0.2 * f(u[k - 1]) + 0.8 * y[k - 1]
    end
    return y
end

y = Float32.(hammerstein_system(ex))
plot(collect(tsteps), y, ticks = :native)

nn_model = Lux.Chain(Lux.Dense(2, 8, tanh), Lux.Dense(8, 1))
p_model, st = Lux.setup(rng, nn_model)

u0 = Float32.([0.0])

function dudt(u, p, t)
    global st
    #input_val = u_vals[Int(round(t*10)+1)]
    out, st = nn_model(vcat(u[1], ex[Int(round(10 * 0.1))]), p, st)
    return out
end

prob = ODEProblem(dudt, u0, tspan, nothing)

function predict_neuralode(p)
    _prob = remake(prob, p = p)
    Array(solve(_prob, Tsit5(), saveat = tsteps, abstol = 1e-8, reltol = 1e-6))
end

function loss(p)
    sol = predict_neuralode(p)
    N = length(sol)
    return sum(abs2.(y[1:N] .- sol')) / N
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentArray(p_model))

res0 = Optimization.solve(optprob, PolyOpt(), maxiters = 100)

sol = predict_neuralode(res0.u)
plot(tsteps, sol')
N = length(sol)
scatter!(tsteps, y[1:N])
```
