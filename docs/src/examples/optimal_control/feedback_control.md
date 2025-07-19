# Universal Differential Equations for Neural Feedback Control

You can also mix a known differential equation and a neural differential
equation, so that the parameters and the neural network are estimated
simultaneously!

We will assume that we know the dynamics of the second equation
(linear dynamics), and our goal is to find a neural network that is dependent
on the current state of the dynamical system that will control the second
equation to stay close to 1.

```@example udeneuralcontrol
import Lux
import Optimization as OPT
import OptimizationPolyalgorithms as OPA
import ComponentArrays as CA
import SciMLSensitivity as SMS
import Zygote
import OrdinaryDiffEq as ODE
import Plots
import Random

rng = Random.default_rng()
u0 = [1.1]
tspan = (0.0, 25.0)
tsteps = 0.0:1.0:25.0

model_univ = Lux.Chain(Lux.Dense(2, 16, tanh), Lux.Dense(16, 16, tanh), Lux.Dense(16, 1))
ps, st = Lux.setup(Random.default_rng(), model_univ)
p_model = CA.ComponentArray(ps)

# Parameters of the second equation (linear dynamics)
p_system = [0.5, -0.5]
p_all = CA.ComponentArray(; p_model, p_system)
θ = CA.ComponentArray(; u0, p_all)

function dudt_univ!(du, u, p, t)
    # Destructure the parameters
    model_weights = p.p_model
    α, β = p.p_system

    # The neural network outputs a control taken by the system
    # The system then produces an output
    model_control, system_output = u

    # Dynamics of the control and system
    dmodel_control = first(model_univ(u, model_weights, st))[1]
    dsystem_output = α * system_output + β * model_control

    # Update in place
    du[1] = dmodel_control
    du[2] = dsystem_output
end

prob_univ = ODE.ODEProblem(dudt_univ!, [0.0, u0[1]], tspan, p_all)
sol_univ = ODE.solve(prob_univ, ODE.Tsit5(), abstol = 1e-8, reltol = 1e-6)

function predict_univ(θ)
    return Array(ODE.solve(prob_univ, ODE.Tsit5(), u0 = [0.0, θ.u0[1]], p = θ.p_all,
        sensealg = SMS.InterpolatingAdjoint(autojacvec = SMS.ReverseDiffVJP(true)),
        saveat = tsteps))
end

loss_univ(θ) = sum(abs2, predict_univ(θ)[2, :] .- 1)
l = loss_univ(θ)
```

```@example udeneuralcontrol
list_plots = []
iter = 0
cb = function (state, l; makeplot = false)
    global list_plots, iter

    if iter == 0
        list_plots = []
    end
    iter += 1

    println(l)

    if makeplot
        plt = Plots.plot(predict_univ(state.u)', ylim = (0, 6))
        push!(list_plots, plt)
        display(plt)
    end
    return false
end
```

```@example udeneuralcontrol
adtype = OPT.AutoZygote()
optf = OPT.OptimizationFunction((x, p) -> loss_univ(x), adtype)
optprob = OPT.OptimizationProblem(optf, θ)
result_univ = OPT.solve(optprob, OPA.PolyOpt(), callback = cb)
```

```@example udeneuralcontrol
cb(result_univ, result_univ.minimum; makeplot = true)
```
