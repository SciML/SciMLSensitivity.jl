# Neural Second Order Ordinary Differential Equation

The neural ODE focuses and finding a neural network such that:

```math
u^\prime = NN(u)
```

However, often in physics-based modeling, the key object is not the
velocity but the acceleration: knowing the acceleration tells you the force
field and thus the generating process for the dynamical system. Thus what we want
to do is find the force, i.e.:

```math
u^{\prime\prime} = NN(u)
```

(Note that in order to be the acceleration, we should divide the output of the
neural network by the mass!)

An example of training a neural network on a second order ODE is as follows:

```@example secondorderneural
import SciMLSensitivity as SMS
import OrdinaryDiffEq as ODE
import Lux
import Optimization as OPT
import OptimizationOptimisers as OPO
import RecursiveArrayTools
import Random
import ComponentArrays as CA

u0 = Float32[0.0; 2.0]
du0 = Float32[0.0; 0.0]
tspan = (0.0f0, 1.0f0)
t = range(tspan[1], tspan[2], length = 20)

model = Lux.Chain(Lux.Dense(2, 50, tanh), Lux.Dense(50, 2))
ps, st = Lux.setup(Random.default_rng(), model)
ps = CA.ComponentArray(ps)
model = Lux.StatefulLuxLayer{true}(model, ps, st)

ff(du, u, p, t) = model(u, p)
prob = ODE.SecondOrderODEProblem{false}(ff, du0, u0, tspan, ps)

function predict(p)
    Array(ODE.solve(prob, ODE.Tsit5(); p, saveat = t))
end

correct_pos = Float32.(transpose(hcat(collect(0:0.05:1)[2:end], collect(2:-0.05:1)[2:end])))

function loss_n_ode(p)
    pred = predict(p)
    sum(abs2, correct_pos .- pred[1:2, :])
end

l1 = loss_n_ode(ps)

callback = function (state, l)
    println(l)
    l < 0.01
end

adtype = OPT.AutoZygote()
optf = OPT.OptimizationFunction((x, p) -> loss_n_ode(x), adtype)
optprob = OPT.OptimizationProblem(optf, ps)

res = OPT.solve(optprob, OPO.Adam(0.01); callback, maxiters = 1000)
```
