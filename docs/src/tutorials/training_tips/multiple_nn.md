# Simultaneous Fitting of Multiple Neural Networks

In many cases users are interested in fitting multiple neural networks
or parameters simultaneously. This tutorial addresses how to perform
this kind of study.

The following is a fully working demo on the Fitzhugh-Nagumo ODE:

```@example
import SciMLSensitivity as SMS
import Lux, ComponentArrays as CA, Optimization as OPT, OptimizationOptimJL as OOJ,
      OptimizationOptimisers as OPO, OrdinaryDiffEq as ODE, Random

rng = Random.default_rng()
Random.seed!(rng, 1)

function fitz(du, u, p, t)
    v, w = u
    a, b, τinv, l = p
    du[1] = v - v^3 / 3 - w + l
    du[2] = τinv * (v + a - b * w)
end

p_ = Float32[0.7, 0.8, 1 / 12.5, 0.5]
u0 = [1.0f0; 1.0f0]
tspan = (0.0f0, 10.0f0)
prob = ODE.ODEProblem(fitz, u0, tspan, p_)
sol = ODE.solve(prob, ODE.Tsit5(), saveat = 0.5)

# Ideal data
X = Array(sol)
Xₙ = X + Float32(1e-3) * randn(eltype(X), size(X))  #noisy data

# For xz term
NN_1 = Lux.Chain(Lux.Dense(2, 16, tanh), Lux.Dense(16, 1))
p1, st1 = Lux.setup(rng, NN_1)

# for xy term
NN_2 = Lux.Chain(Lux.Dense(3, 16, tanh), Lux.Dense(16, 1))
p2, st2 = Lux.setup(rng, NN_2)
scaling_factor = 1.0f0

p1 = CA.ComponentArray(p1)
p2 = CA.ComponentArray(p2)

p = CA.ComponentArray{eltype(p1)}()
p = CA.ComponentArray(p; p1)
p = CA.ComponentArray(p; p2)
p = CA.ComponentArray(p; scaling_factor)

function dudt_(u, p, t)
    v, w = u
    z1 = NN_1([v, w], p.p1, st1)[1]
    z2 = NN_2([v, w, t], p.p2, st2)[1]
    [z1[1], p.scaling_factor * z2[1]]
end
prob_nn = ODE.ODEProblem(dudt_, u0, tspan, p)
sol_nn = ODE.solve(prob_nn, ODE.Tsit5(), saveat = sol.t)

function predict(θ)
    Array(ODE.solve(prob_nn, ODE.Vern7(), p = θ, saveat = sol.t,
        abstol = 1e-6, reltol = 1e-6,
        sensealg = SMS.InterpolatingAdjoint(autojacvec = SMS.ReverseDiffVJP(true))))
end

# No regularisation right now
function loss(θ)
    pred = predict(θ)
    sum(abs2, Xₙ .- pred)
end
loss(p)
const losses = []
callback(θ, l) = begin
    push!(losses, l)
    if length(losses) % 50 == 0
        println(losses[end])
    end
    false
end
adtype = OPT.AutoZygote()
optf = OPT.OptimizationFunction((x, p) -> loss(x), adtype)

optprob = OPT.OptimizationProblem(optf, p)
res1_uode = OPT.solve(optprob, OPO.Adam(0.01),
    callback = callback, maxiters = 500)

optprob2 = OPT.OptimizationProblem(optf, res1_uode.u)
res2_uode = OPT.solve(optprob2, OOJ.BFGS(), maxiters = 10000,
    callback = callback)
```

The key is that `OPT.solve` acts on a single parameter vector `p`.
Thus what we do here is concatenate all the parameters into a single
ComponentVector `p` and then train on this parameter
vector. Whenever we need to evaluate the neural networks, we dereference the
vector and grab the key that corresponds to the neural network.
For example, the `p1` portion is `p.p1`, which is why the
first neural network's evolution is written like `NN_1([v,w], p.p1)`.

This method is flexible to use with many optimizers and in fairly
optimized ways.
We can also see with the `scaling_factor` that we can grab parameters
directly out of the vector and use them as needed.
