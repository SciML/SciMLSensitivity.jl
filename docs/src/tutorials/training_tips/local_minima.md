# Strategies to Avoid Local Minima

Local minima can be an issue with fitting neural differential equations. However,
there are many strategies to avoid local minima:

 1. Insert stochasticity into the loss function through minibatching
 2. Weigh the loss function to allow for fitting earlier portions first
 3. Iteratively grow the fit
 4. Training both the initial conditions and parameters at first, followed by just the parameters

## Iterative Growing Of Fits to Reduce Probability of Bad Local Minima

In this example, we will show how to use strategy (3) in order to increase the
robustness of the fit. Let's start with the same neural ODE example we've used
before, except with one small twist: we wish to find the neural ODE that fits
on `(0,5.0)`. Naively, we use the same training strategy as before:

```@example iterativefit
using SciMLSensitivity
using OrdinaryDiffEq,
      ComponentArrays, SciMLSensitivity, Optimization, OptimizationOptimisers
using Lux, Plots, Random, Zygote

rng = Random.default_rng()
u0 = Float32[2.0; 0.0]
datasize = 30
tspan = (0.0f0, 5.0f0)
tsteps = range(tspan[1], tspan[2], length = datasize)

function trueODEfunc(du, u, p, t)
    true_A = Float32[-0.1 2.0; -2.0 -0.1]
    du .= ((u .^ 3)'true_A)'
end

prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))

dudt2 = Lux.Chain(x -> x .^ 3,
    Lux.Dense(2, 16, tanh),
    Lux.Dense(16, 2))

pinit, st = Lux.setup(rng, dudt2)
pinit = ComponentArray(pinit)

function neuralode_f(u, p, t)
    dudt2(u, p, st)[1]
end

function predict_neuralode(p)
    prob = ODEProblem(neuralode_f, u0, tspan, p)
    sol = solve(prob, Vern7(), saveat = tsteps, abstol = 1e-6, reltol = 1e-6)
    Array(sol)
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, (ode_data[:, 1:size(pred, 2)] .- pred))
    return loss
end

iter = 0
function callback(state, l; doplot = false)
    global iter
    iter += 1

    println(l)
    if doplot
        # plot current prediction against data
        pred = predict_neuralode(state.u)
        plt = scatter(tsteps[1:size(pred, 2)], ode_data[1, 1:size(pred, 2)], label = "data")
        scatter!(plt, tsteps[1:size(pred, 2)], pred[1, :], label = "prediction")
        display(plot(plt))
    end

    return false
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)

optprob = Optimization.OptimizationProblem(optf, pinit)
result_neuralode = Optimization.solve(optprob,
    Adam(0.05), callback = callback,
    maxiters = 300)

pred = predict_neuralode(result_neuralode.u)
plt = scatter(tsteps[1:size(pred, 2)], ode_data[1, 1:size(pred, 2)], label = "data")
scatter!(plt, tsteps[1:size(pred, 2)], pred[1, :], label = "prediction")
```

However, we've now fallen into a trap of a local minimum. If the optimizer changes
the parameters, so it dips early, it will increase the loss because there will
be more error in the later parts of the time series. Thus it tends to just stay
flat and never fit perfectly. This thus suggests strategies (2) and (3): do not
allow the later parts of the time series to influence the fit until the later
stages. Strategy (3) seems more robust, so this is what will be demonstrated.

Let's start by reducing the timespan to `(0,1.5)`:

```@example iterativefit
function predict_neuralode(p)
    prob = ODEProblem(neuralode_f, u0, (0.0f0, 1.5f0), p)
    sol = solve(prob, Vern7(), saveat = tsteps, abstol = 1e-6, reltol = 1e-6)
    Array(sol)
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)

optprob = Optimization.OptimizationProblem(optf, pinit)
result_neuralode2 = Optimization.solve(optprob,
    Adam(0.05), callback = callback,
    maxiters = 300)

pred = predict_neuralode(result_neuralode2.u)
plt = scatter(tsteps[1:size(pred, 2)], ode_data[1, 1:size(pred, 2)], label = "data")
scatter!(plt, tsteps[1:size(pred, 2)], pred[1, :], label = "prediction")
```

This fits beautifully. Now let's grow the timespan and utilize the parameters
from our `(0,1.5)` fit as the initial condition to our next fit:

```@example iterativefit
function predict_neuralode(p)
    prob = ODEProblem(neuralode_f, u0, (0.0f0, 3.0f0), p)
    sol = solve(prob, Vern7(), saveat = tsteps, abstol = 1e-6, reltol = 1e-6)
    Array(sol)
end

optprob = Optimization.OptimizationProblem(optf, result_neuralode2.u)
result_neuralode3 = Optimization.solve(optprob,
    Adam(0.05), maxiters = 300,
    callback = callback)

pred = predict_neuralode(result_neuralode3.u)
plt = scatter(tsteps[1:size(pred, 2)], ode_data[1, 1:size(pred, 2)], label = "data")
scatter!(plt, tsteps[1:size(pred, 2)], pred[1, :], label = "prediction")
```

Once again, a great fit. Now we utilize these parameters as the initial condition
to the full fit:

```@example iterativefit
function predict_neuralode(p)
    prob = ODEProblem(neuralode_f, u0, (0.0f0, 5.0f0), p)
    sol = solve(prob, Vern7(), saveat = tsteps, abstol = 1e-6, reltol = 1e-6)
    Array(sol)
end

optprob = Optimization.OptimizationProblem(optf, result_neuralode3.u)
result_neuralode4 = Optimization.solve(optprob,
    Adam(0.01), maxiters = 500,
    callback = callback)

pred = predict_neuralode(result_neuralode4.u)
plt = scatter(tsteps[1:size(pred, 2)], ode_data[1, 1:size(pred, 2)], label = "data")
scatter!(plt, tsteps[1:size(pred, 2)], pred[1, :], label = "prediction")
```

## Training both the initial conditions and the parameters to start

In this example, we will show how to use strategy (4) in order to accomplish the
same goal, except rather than growing the trajectory iteratively, we can train on
the whole trajectory. We do this by allowing the neural ODE to learn both the
initial conditions and parameters to start, and then reset the initial conditions
back and train only the parameters. Note: this strategy is demonstrated for the (0, 5)
time span and (0, 10), any longer and more iterations will be required. Alternatively,
one could use a mix of (3) and (4), or breaking up the trajectory into chunks and just (4).

```@example resetic
using SciMLSensitivity
using OrdinaryDiffEq,
      ComponentArrays, SciMLSensitivity, Optimization, OptimizationOptimisers
using Lux, Plots, Random, Zygote

#Starting example with tspan (0, 5)
u0 = Float32[2.0; 0.0]
datasize = 30
tspan = (0.0f0, 5.0f0)
tsteps = range(tspan[1], tspan[2], length = datasize)

function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u .^ 3)'true_A)'
end

prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))

dudt2 = Chain(Dense(2, 16, tanh), Dense(16, 2))

ps, st = Lux.setup(Random.default_rng(), dudt2)
p = ComponentArray(ps)
dudt(u, p, t) = first(dudt2(u, p, st))
prob = ODEProblem(dudt, u0, tspan)

function predict_n_ode(pu0)
    u0 = pu0.u0
    p = pu0.p
    Array(solve(prob, Tsit5(), u0 = u0, p = p, saveat = tsteps))
end

function loss_n_ode(pu0, _)
    pred = predict_n_ode(pu0)
    sqnorm(x) = sum(abs2, x)
    loss = sum(abs2, ode_data .- pred)
    loss
end

function callback(state, l; doplot = true) #callback function to observe training
    pred = predict_n_ode(state.u)
    display(sum(abs2, ode_data .- pred))
    if doplot
        # plot current prediction against data
        pl = plot(tsteps, ode_data[1, :], label = "data")
        plot!(pl, tsteps, pred[1, :], label = "prediction")
        display(plot(pl))
    end
    return false
end

p_init = ComponentArray(; u0 = u0, p = p)

predict_n_ode(p_init)
loss_n_ode(p_init, nothing)

res = solve(OptimizationProblem(OptimizationFunction(loss_n_ode, AutoZygote()), p_init),
    Adam(0.05); callback = callback, maxiters = 1000)

function predict_n_ode2(p)
    Array(solve(prob, Tsit5(), u0 = u0, p = p, saveat = tsteps))
end

function loss_n_ode2(p, _)
    pred = predict_n_ode2(p)
    sqnorm(x) = sum(abs2, x)
    loss = sum(abs2, ode_data .- pred)
    loss
end

function callback2(state, l; doplot = true) #callback function to observe training
    pred = predict_n_ode2(state.u)
    display(sum(abs2, ode_data .- pred))
    if doplot
        # plot current prediction against data
        pl = plot(tsteps, ode_data[1, :], label = "data")
        plot!(pl, tsteps, pred[1, :], label = "prediction")
        display(plot(pl))
    end
    return false
end

#Here we reset the IC back to the original and train only the NODE parameters
u0 = Float32[2.0; 0.0]
res = solve(OptimizationProblem(OptimizationFunction(loss_n_ode2, AutoZygote()), p_init.p),
    Adam(0.05); callback = callback2, maxiters = 1000)

#Now use the same technique for a longer tspan (0, 10)
datasize = 30
tspan = (0.0f0, 10.0f0)
tsteps = range(tspan[1], tspan[2], length = datasize)

prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))

dudt2 = Chain(Dense(2, 16, tanh), Dense(16, 2))

ps, st = Lux.setup(Random.default_rng(), dudt2)
p = ComponentArray(ps)
dudt(u, p, t) = first(dudt2(u, p, st))
prob = ODEProblem(dudt, u0, tspan)

p_init = ComponentArray(; u0 = u0, p = p)
res = solve(OptimizationProblem(OptimizationFunction(loss_n_ode, AutoZygote()), p_init),
    Adam(0.05); callback = callback, maxiters = 1000)

res = solve(OptimizationProblem(OptimizationFunction(loss_n_ode2, AutoZygote()), p_init.p),
    Adam(0.05); callback = callback2, maxiters = 1000)
```

And there we go, a set of robust strategies for fitting an equation that would otherwise
get stuck in local optima.
