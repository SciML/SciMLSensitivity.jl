# Benchmarks

## Note on benchmarking and getting the best performance out of the SciML stack's adjoints

From our [recent papers](https://arxiv.org/abs/1812.01892), it's clear that `EnzymeVJP` is the fastest,
especially when the program is set up to be fully non-allocating mutating functions. Thus for all benchmarking,
especially with PDEs, this should be done. Neural network libraries don't make use of mutation effectively
[except for SimpleChains.jl](https://julialang.org/blog/2022/04/simple-chains/), so we recommend creating a
neural ODE / universal ODE with `ZygoteVJP` and Lux first, but then check the correctness by moving the
implementation over to SimpleChains and if possible `EnzymeVJP`. This can be an order of magnitude improvement
(or more) in many situations over all the previous benchmarks using Zygote and Lux, and thus it's
highly recommended in scenarios that require performance.

## Vs Torchdiffeq 1 million and less ODEs

A raw ODE solver benchmark showcases [>30x performance advantage for DifferentialEquations.jl](https://gist.github.com/ChrisRackauckas/cc6ac746e2dfd285c28e0584a2bfd320)
for ODEs ranging in size from 3 to nearly 1 million.

## Vs Torchdiffeq on neural ODE training

A training benchmark using the spiral ODE from the original neural ODE paper
[demonstrates a 100x performance advantage for DiffEqFlux in training neural ODEs](https://gist.github.com/ChrisRackauckas/4a4d526c15cc4170ce37da837bfc32c4).

## Vs torchsde on small SDEs

Using the code from torchsde's README, we demonstrated a [>70,000x performance
advantage over torchsde](https://gist.github.com/ChrisRackauckas/6a03e7b151c86b32d74b41af54d495c6).
Further benchmarking is planned, but was found to be computationally infeasible
at this time.

## A bunch of adjoint choices on neural ODEs

Quick summary:

  - `BacksolveAdjoint` can be the fastest (but use with caution!); about 25% faster
  - Using `ZygoteVJP` is faster than other vjp choices for larger neural networks
  - `ReverseDiffVJP(compile = true)` works well for small Lux neural networks

```julia
using OrdinaryDiffEq, Lux, SciMLSensitivity, Zygote, BenchmarkTools, Random, ComponentArrays

u0 = Float32[2.0; 0.0]
datasize = 30
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2], length = datasize)

function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u .^ 3)'true_A)'
end

prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))

dudt2 = Chain(x -> x .^ 3, Dense(2, 50, tanh), Dense(50, 2))
Random.seed!(100)

for sensealg in (InterpolatingAdjoint(autojacvec = ZygoteVJP()),
    InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)),
    BacksolveAdjoint(autojacvec = ReverseDiffVJP(true)),
    BacksolveAdjoint(autojacvec = ZygoteVJP()),
    BacksolveAdjoint(autojacvec = ReverseDiffVJP(false)),
    BacksolveAdjoint(autojacvec = TrackerVJP()),
    QuadratureAdjoint(autojacvec = ReverseDiffVJP(true)),
    TrackerAdjoint())
    prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(); saveat = tsteps,
        sensealg = sensealg)
    ps, st = Lux.setup(Random.default_rng(), prob_neuralode)
    ps = ComponentArray(ps)

    loss_neuralode = function (u0, p, st)
        pred = Array(first(prob_neuralode(u0, p, st)))
        loss = sum(abs2, ode_data .- pred)
        return loss
    end

    t = @belapsed Zygote.gradient($loss_neuralode, $u0, $ps, $st)
    println("$(sensealg) took $(t)s")
end

# InterpolatingAdjoint{0, true, Val{:central}, ZygoteVJP}(ZygoteVJP(false), false, false) took 0.029134224s
# InterpolatingAdjoint{0, true, Val{:central}, ReverseDiffVJP{true}}(ReverseDiffVJP{true}(), false, false) took 0.001657377s
# BacksolveAdjoint{0, true, Val{:central}, ReverseDiffVJP{true}}(ReverseDiffVJP{true}(), true, false) took 0.002477057s
# BacksolveAdjoint{0, true, Val{:central}, ZygoteVJP}(ZygoteVJP(false), true, false) took 0.031533335s
# BacksolveAdjoint{0, true, Val{:central}, ReverseDiffVJP{false}}(ReverseDiffVJP{false}(), true, false) took 0.004605386s
# BacksolveAdjoint{0, true, Val{:central}, TrackerVJP}(TrackerVJP(false), true, false) took 0.044568018s
# QuadratureAdjoint{0, true, Val{:central}, ReverseDiffVJP{true}}(ReverseDiffVJP{true}(), 1.0e-6, 0.001) took 0.002489559s
# TrackerAdjoint() took 0.003759097s
```
