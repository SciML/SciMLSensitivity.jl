# [Parameter Estimation of Ordinary Differential Equations](@id odeparamestim)

## Copy-Paste Code

If you want to just get things running, try the following! Explanation will
follow.

```@example optode_cp
using OrdinaryDiffEq,
      Optimization, OptimizationPolyalgorithms, SciMLSensitivity,
      Zygote, Plots

function lotka_volterra!(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = dx = α * x - β * x * y
    du[2] = dy = -δ * y + γ * x * y
end

# Initial condition
u0 = [1.0, 1.0]

# Simulation interval and intermediary points
tspan = (0.0, 10.0)
tsteps = 0.0:0.1:10.0

# LV equation parameter. p = [α, β, δ, γ]
p = [1.5, 1.0, 3.0, 1.0]

# Setup the ODE problem, then solve
prob = ODEProblem(lotka_volterra!, u0, tspan, p)
sol = solve(prob, Tsit5())

# Plot the solution
using Plots
plot(sol)
savefig("LV_ode.png")

function loss(p)
    sol = solve(prob, Tsit5(), p = p, saveat = tsteps)
    loss = sum(abs2, sol .- 1)
    return loss
end

callback = function (state, l)
    display(l)
    pred = solve(prob, Tsit5(), p = state.u, saveat = tsteps)
    plt = plot(pred, ylim = (0, 6))
    display(plt)
    # Tell Optimization.solve to not halt the optimization. If return true, then
    # optimization stops.
    return false
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, p)

result_ode = Optimization.solve(optprob, PolyOpt(),
    callback = callback,
    maxiters = 100)
```

## Explanation

First, let's create a Lotka-Volterra ODE using OrdinaryDiffEq.jl. For
more details, [see the OrdinaryDiffEq.jl documentation](https://docs.sciml.ai/DiffEqDocs/stable/). The Lotka-Volterra equations have the form:

```math
\begin{aligned}
\frac{dx}{dt} &= \alpha x - \beta x y      \\
\frac{dy}{dt} &= -\delta y + \gamma x y    \\
\end{aligned}
```

```@example optode
using OrdinaryDiffEq,
      Optimization, OptimizationPolyalgorithms,
      SciMLSensitivity, Zygote, Plots

function lotka_volterra!(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = dx = α * x - β * x * y
    du[2] = dy = -δ * y + γ * x * y
end

# Initial condition
u0 = [1.0, 1.0]

# Simulation interval and intermediary points
tspan = (0.0, 10.0)
tsteps = 0.0:0.1:10.0

# LV equation parameter. p = [α, β, δ, γ]
p = [1.5, 1.0, 3.0, 1.0]

# Setup the ODE problem, then solve
prob = ODEProblem(lotka_volterra!, u0, tspan, p)
sol = solve(prob, Tsit5())

# Plot the solution
using Plots
plot(sol)
savefig("LV_ode.png")
```

![LV Solution Plot](https://user-images.githubusercontent.com/1814174/51388169-9a07f300-1af6-11e9-8c6c-83c41e81d11c.png)

For this first example, we do not yet include a neural network. We take
[an AD-compatible `solve`
function](https://docs.sciml.ai/SciMLSensitivity/stable/manual/differential_equation_sensitivities/)
that takes the parameters and an initial condition and returns the solution of
the differential equation. Next, we choose a loss function. Our goal will be to
find parameters that make the Lotka-Volterra solution constant `x(t)=1`, so we
define our loss as the squared distance from 1.

```@example optode
function loss(p)
    sol = solve(prob, Tsit5(), p = p, saveat = tsteps)
    loss = sum(abs2, sol .- 1)
    return loss
end
```

Lastly, we use the `Optimization.solve` function to train the parameters using `Adam` to
arrive at parameters which optimize for our goal. `Optimization.solve` allows defining
a callback that will be called at each step of our training loop. It takes in
the current parameter vector and the returns of the last call to the loss
function. We will display the current loss and make a plot of the current
situation:

```@example optode
callback = function (state, l)
    display(l)
    pred = solve(prob, Tsit5(), p = state.u, saveat = tsteps)
    plt = plot(pred, ylim = (0, 6))
    display(plt)
    # Tell Optimization.solve to not halt the optimization. If return true, then
    # optimization stops.
    return false
end
```

Let's optimize the model.

```@example optode
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, p)

result_ode = Optimization.solve(optprob, PolyOpt(),
    callback = callback,
    maxiters = 100)
```

In just seconds we found parameters which give a relative loss of `1e-16`! We can
get the final loss with `result_ode.minimum`, and get the optimal parameters
with `result_ode.u`. For example, we can plot the final outcome and show
that we solved the control problem and successfully found parameters to make the
ODE solution constant:

```@example optode
remade_solution = solve(remake(prob, p = result_ode.u), Tsit5(),
    saveat = tsteps)
plot(remade_solution, ylim = (0, 6))
```

![Final plot](https://user-images.githubusercontent.com/1814174/51399500-1f4dd080-1b14-11e9-8c9d-144f93b6eac2.gif)
