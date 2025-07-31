# [Solving Optimal Control Problems with Universal Differential Equations](@id optcontrol)

Here we will solve a classic optimal control problem with a universal differential
equation. Let

```math
x^{′′} = u^3(t)
```

where we want to optimize our controller `u(t)` such that the following is
minimized:

```math
L(\theta) = \sum_i \Vert 4 - x(t_i) \Vert + 2 \Vert x^\prime(t_i) \Vert + \Vert u(t_i) \Vert
```

where ``i`` is measured on (0,8) at 0.01 intervals. To do this, we rewrite the
ODE in first order form:

```math
\begin{aligned}
x^\prime &= v \\
v^′ &= u^3(t) \\
\end{aligned}
```

and thus

```math
L(\theta) = \sum_i \Vert 4 - x(t_i) \Vert + 2 \Vert v(t_i) \Vert + \Vert u(t_i) \Vert
```

is our loss function on the first order system. We thus choose a neural network
form for ``u`` and optimize the equation with respect to this loss. Note that we
will first reduce control cost (the last term) by 10x in order to bump the network out
of a local minimum. This looks like:

```@example neuraloptimalcontrol
import Lux
import ComponentArrays as CA
import OrdinaryDiffEq as ODE
import Optimization as OPT
import OptimizationOptimJL as OOJ
import OptimizationOptimisers as OPO
import SciMLSensitivity as SMS
import Zygote
import Plots
import Statistics
import Random
import ForwardDiff as FD

rng = Random.default_rng()
tspan = (0.0f0, 8.0f0)

ann = Lux.Chain(Lux.Dense(1, 32, tanh), Lux.Dense(32, 32, tanh), Lux.Dense(32, 1))
ps, st = Lux.setup(rng, ann)
p = CA.ComponentArray(ps)

θ, _ax = CA.getdata(p), CA.getaxes(p)
const ax = _ax

function dxdt_(dx, x, p, t)
    ps = CA.ComponentArray(p, ax)
    x1, x2 = x
    dx[1] = x[2]
    dx[2] = first(ann([t], ps, st))[1]^3
end
x0 = [-4.0f0, 0.0f0]
ts = Float32.(collect(0.0:0.01:tspan[2]))
prob = ODE.ODEProblem(dxdt_, x0, tspan, θ)
ODE.solve(prob, ODE.Vern9(), abstol = 1e-10, reltol = 1e-10)

function predict_adjoint(θ)
    Array(ODE.solve(prob, ODE.Vern9(), p = θ, saveat = ts))
end
function loss_adjoint(θ)
    x = predict_adjoint(θ)
    ps = CA.ComponentArray(θ, ax)
    Statistics.mean(abs2, 4.0f0 .- x[1, :]) + 2Statistics.mean(abs2, x[2, :]) +
    Statistics.mean(abs2, [first(first(ann([t], ps, st))) for t in ts]) / 10
end

l = loss_adjoint(θ)
cb = function (state, l; doplot = true)
    println(l)

    ps = CA.ComponentArray(state.u, ax)

    if doplot
        p = Plots.plot(
            ODE.solve(ODE.remake(prob, p = state.u), ODE.Tsit5(), saveat = 0.01),
            ylim = (-6, 6), lw = 3)
        Plots.plot!(
            p, ts, [first(first(ann([t], ps, st))) for t in ts], label = "u(t)", lw = 3)
        display(p)
    end

    return false
end

# Setup and run the optimization

loss1 = loss_adjoint(θ)
adtype = OPT.AutoForwardDiff()
optf = OPT.OptimizationFunction((x, p) -> loss_adjoint(x), adtype)

optprob = OPT.OptimizationProblem(optf, θ)
res1 = OPT.solve(
    optprob, OPO.Adam(0.01), callback = cb, maxiters = 100)

optprob2 = OPT.OptimizationProblem(optf, res1.u)
res2 = OPT.solve(
    optprob2, OOJ.BFGS(), callback = cb, maxiters = 100)
```

Now that the system is in a better behaved part of parameter space, we return to
the original loss function to finish the optimization:

```@example neuraloptimalcontrol
function loss_adjoint(θ)
    x = predict_adjoint(θ)
    ps = CA.ComponentArray(θ, ax)
    Statistics.mean(abs2, 4.0 .- x[1, :]) + 2Statistics.mean(abs2, x[2, :]) +
    Statistics.mean(abs2, [first(first(ann([t], ps, st))) for t in ts])
end
optf3 = OPT.OptimizationFunction((x, p) -> loss_adjoint(x), adtype)

optprob3 = OPT.OptimizationProblem(optf3, res2.u)
res3 = OPT.solve(optprob3, OOJ.BFGS(), maxiters = 100)
```

Now let's see what we received:

```@example neuraloptimalcontrol
l = loss_adjoint(res3.u)
cb(res3, l)
p = Plots.plot(ODE.solve(ODE.remake(prob, p = res3.u), ODE.Tsit5(), saveat = 0.01), ylim = (
    -6, 6), lw = 3)
Plots.plot!(p, ts, [first(first(ann([t], CA.ComponentArray(res3.u, ax), st))) for t in ts],
    label = "u(t)", lw = 3)
```
