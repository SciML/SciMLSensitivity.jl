# Delay Differential Equations

Other differential equation problem types from DifferentialEquations.jl are
supported. For example, we can build a layer with a delay differential equation
like:

```@example dde
import OrdinaryDiffEq as ODE
import Optimization as OPT
import SciMLSensitivity as SMS
import OptimizationPolyalgorithms as OPA
import DelayDiffEq as DDE

# Define the same LV equation, but including a delay parameter
function delay_lotka_volterra!(du, u, h, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = dx = (α - β * y) * h(p, t - 0.1)[1]
    du[2] = dy = (δ * x - γ) * y
end

# Initial parameters
p = [2.2, 1.0, 2.0, 0.4]

# Define a vector containing delays for each variable (although only the first
# one is used)
h(p, t) = ones(eltype(p), 2)

# Initial conditions
u0 = [1.0, 1.0]

# Define the problem as a delay differential equation
prob_dde = DDE.DDEProblem(delay_lotka_volterra!, u0, h, (0.0, 10.0),
    constant_lags = [0.1])

function predict_dde(p)
    return Array(ODE.solve(prob_dde, DDE.MethodOfSteps(ODE.Tsit5());
        u0, p, saveat = 0.1, sensealg = SMS.ReverseDiffAdjoint()))
end

loss_dde(p) = sum(abs2, x - 1 for x in predict_dde(p))

import Plots
callback = function (state, l; doplot = false)
    display(loss_dde(state.u))
    doplot &&
        display(Plots.plot(
            ODE.solve(ODE.remake(prob_dde, p = state.u), DDE.MethodOfSteps(ODE.Tsit5()), saveat = 0.1),
            ylim = (0, 6)))
    return false
end

adtype = OPT.AutoZygote()
optf = OPT.OptimizationFunction((x, p) -> loss_dde(x), adtype)
optprob = OPT.OptimizationProblem(optf, p)
result_dde = OPT.solve(optprob, OPA.PolyOpt(); maxiters = 300, callback)
```

Notice that we chose `sensealg = ReverseDiffAdjoint()` to utilize the ReverseDiff.jl
reverse-mode to handle the delay differential equation.

We define a callback to display the solution at the current parameters for each step of the training:

```@example dde
import Plots
callback = function (state, l; doplot = false)
    display(loss_dde(state.u))
    doplot &&
        display(Plots.plot(
            ODE.solve(ODE.remake(prob_dde, p = state.u), DDE.MethodOfSteps(ODE.Tsit5()), saveat = 0.1),
            ylim = (0, 6)))
    return false
end
```

We use `Optimization.solve` to optimize the parameters for our loss function:

```@example dde
adtype = OPT.AutoZygote()
optf = OPT.OptimizationFunction((x, p) -> loss_dde(x), adtype)
optprob = OPT.OptimizationProblem(optf, p)
result_dde = OPT.solve(optprob, OPA.PolyOpt(); callback)
```
