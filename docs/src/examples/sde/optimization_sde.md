# Optimization of Stochastic Differential Equations

Here we demonstrate `sensealg = ForwardDiffSensitivity()` (provided by
SciMLSensitivity.jl) for forward-mode automatic differentiation of a small
stochastic differential equation. For large parameter equations, like neural
stochastic differential equations, you should use reverse-mode automatic
differentiation. However, forward-mode can be more efficient for low numbers
of parameters (<100). (Note: the default is reverse-mode AD, which is more suitable
for things like neural SDEs!)

## Example 1: Fitting Data with SDEs via Method of Moments and Parallelism

Let's do the most common scenario: fitting data. Let's say our ecological system
is a stochastic process. Each time we solve this equation we get a different
solution, so we need a sensible data source.

```@example sde
using StochasticDiffEq, SciMLSensitivity, Plots

function lotka_volterra!(du, u, p, t)
    x, y = u
    α, β, γ, δ = p
    du[1] = dx = α * x - β * x * y
    du[2] = dy = δ * x * y - γ * y
end
u0 = [1.0, 1.0]
tspan = (0.0, 10.0)

function multiplicative_noise!(du, u, p, t)
    x, y = u
    du[1] = p[5] * x
    du[2] = p[6] * y
end
p = [1.5, 1.0, 3.0, 1.0, 0.3, 0.3]

prob = SDEProblem(lotka_volterra!, multiplicative_noise!, u0, tspan, p)
sol = solve(prob, SOSRI())
plot(sol)
```

![](https://user-images.githubusercontent.com/1814174/88511873-97bc0a00-cfb3-11ea-8cf5-5930b6575d9d.png)

Let's assume that we are observing the seasonal behavior of this system and have
10,000 years of data, corresponding to 10,000 observations of this timeseries.
We can utilize this to get the seasonal means and variances. To simulate that
scenario, we will generate 10,000 trajectories from the SDE to build our dataset:

```@example sde
using Statistics
ensembleprob = EnsembleProblem(prob)
@time sol = solve(ensembleprob, SOSRI(), saveat = 0.1, trajectories = 10_000)
truemean = mean(sol, dims = 3)[:, :]
truevar = var(sol, dims = 3)[:, :]
```

From here, we wish to utilize the method of moments to fit the SDE's parameters.
Thus our loss function will be to solve the SDE a bunch of times and compute
moment equations and use these as our loss against the original series. We
then plot the evolution of the means and variances to verify the fit. For example:

```@example sde
arrsol = sol
currp = rand(length(p))
function predict(p)
    if p == currp
        return arrsol
    end
    global currp = p
    tmp_prob = remake(prob, p = p)
    ensembleprob = EnsembleProblem(tmp_prob)
    tmp_sol = solve(ensembleprob, SOSRI(), saveat = 0.1, trajectories = 1000)
    global arrsol = Array(tmp_sol)
    return arrsol
end

function loss(p)
    pred = predict(p)
    sum(abs2, truemean - mean(arrsol, dims = 3)) +
    0.1sum(abs2, truevar - var(arrsol, dims = 3))
end

function cb2(st, l)
    @show st.u, l
    arrsol1 = predict(st.u)
    means = mean(arrsol1, dims = 3)[:, :]
    vars = var(arrsol1, dims = 3)[:, :]
    p1 = plot(sol[1].t, means', lw = 5)
    scatter!(p1, sol[1].t, truemean')
    p2 = plot(sol[1].t, vars', lw = 5)
    scatter!(p2, sol[1].t, truevar')
    p = plot(p1, p2, layout = (2, 1))
    display(p)
    false
end
```

We can then use `Optimization.solve` to fit the SDE:

```@example sde
using Optimization, Zygote, OptimizationOptimisers
pinit = [1.2, 0.8, 2.5, 0.8, 0.1, 0.1]
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)
@time res = Optimization.solve(optprob, Adam(0.05), callback = cb2, maxiters = 100)
```

Notice that **both the parameters of the deterministic drift equations and the
stochastic portion (the diffusion equation) are fit through this process!**
Also notice that the final fit of the moment equations is close:

![](https://user-images.githubusercontent.com/1814174/88511872-97bc0a00-cfb3-11ea-9d44-a3ed96a77df9.png)

The time for the full fitting process was:

```
250.654845 seconds (4.69 G allocations: 104.868 GiB, 11.87% gc time)
```

approximately 4 minutes.

## Example 2: Fitting SDEs via Bayesian Quasi-Likelihood Approaches

An inference method which can often be much more efficient is the quasi-likelihood approach.
This approach matches the random likelihood of the SDE output with the random sampling of a Bayesian
inference problem to more efficiently directly estimate the posterior distribution. For more information,
please see [the Turing.jl Bayesian Differential Equations tutorial](https://turinglang.org/v0.29/tutorials/10-bayesian-differential-equations/).

## Example 3: Controlling SDEs to an objective

In this example, we will find the parameters of the SDE that force the
solution to be close to the constant 1.

```@example sde
using StochasticDiffEq, Optimization, OptimizationOptimisers, Plots

function lotka_volterra!(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = dx = α * x - β * x * y
    du[2] = dy = -δ * y + γ * x * y
end

function lotka_volterra_noise!(du, u, p, t)
    du[1] = 0.1u[1]
    du[2] = 0.1u[2]
end

u0 = [1.0, 1.0]
tspan = (0.0, 10.0)
p = [2.2, 1.0, 2.0, 0.4]
prob_sde = SDEProblem(lotka_volterra!, lotka_volterra_noise!, u0, tspan)

function predict_sde(p)
    return Array(solve(prob_sde, SOSRI(), p = p,
        sensealg = ForwardDiffSensitivity(), saveat = 0.1))
end

loss_sde(p) = sum(abs2, x - 1 for x in predict_sde(p))
```

For this training process, because the loss function is stochastic, we will use
the `Adam` optimizer from Flux.jl. The `Optimization.solve` function is the same as
before. However, to speed up the training process, we will use a global counter
so that way we only plot the current results every 10 iterations. This looks
like:

```@example sde
callback = function (state, l)
    display(l)
    remade_solution = solve(remake(prob_sde, p = state.u), SOSRI(), saveat = 0.1)
    plt = plot(remade_solution, ylim = (0, 6))
    display(plt)
    return false
end
```

Let's optimize

```@example sde
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_sde(x), adtype)

optprob = Optimization.OptimizationProblem(optf, p)
result_sde = Optimization.solve(optprob, Adam(0.1), callback = callback, maxiters = 100)
```

![](https://user-images.githubusercontent.com/1814174/51399524-2c6abf80-1b14-11e9-96ae-0192f7debd03.gif)
