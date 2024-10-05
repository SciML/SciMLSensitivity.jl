# Newton and Hessian-Free Newton-Krylov with Second Order Adjoint Sensitivity Analysis

In many cases it may be more optimal or more stable to fit using second order
Newton-based optimization techniques. Since SciMLSensitivity.jl provides
second order sensitivity analysis for fast Hessians and Hessian-vector
products (via forward-over-reverse), we can utilize these in our neural/universal
differential equation training processes.

`sciml_train` is set up to automatically use second order sensitivity analysis
methods if a second order optimizer is requested via Optim.jl. Thus `Newton`
and `NewtonTrustRegion` optimizers will use a second order Hessian-based
optimization, while `KrylovTrustRegion` will utilize a Krylov-based method
with Hessian-vector products (never forming the Hessian) for large parameter
optimizations.

```@example secondorderadjoints
using SciMLSensitivity
using Lux, ComponentArrays, Optimization, OptimizationOptimisers,
      OrdinaryDiffEq, Plots, Random, OptimizationOptimJL

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
ps, st = Lux.setup(Random.default_rng(), dudt2)
function neuralodefunc(u, p, t)
    dudt2(u, p, st)[1]
end
function prob_neuralode(u0, p)
    prob = ODEProblem(neuralodefunc, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat = tsteps)
end
ps = ComponentArray(ps)

function predict_neuralode(p)
    Array(prob_neuralode(u0, p))
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss
end

# Callback function to observe training
list_plots = []
iter = 0
callback = function (state, l; doplot = false)
    global list_plots, iter

    if iter == 0
        list_plots = []
    end
    iter += 1

    display(l)

    # plot current prediction against data
    pred = predict_neuralode(state.u)
    plt = scatter(tsteps, ode_data[1, :], label = "data")
    scatter!(plt, tsteps, pred[1, :], label = "prediction")
    push!(list_plots, plt)
    if doplot
        display(plot(plt))
    end

    return l < 0.01
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)

optprob1 = Optimization.OptimizationProblem(optf, ps)
pstart = Optimization.solve(
    optprob1, Optimisers.Adam(0.01), callback = callback, maxiters = 100).u

optprob2 = Optimization.OptimizationProblem(optf, pstart)
pmin = Optimization.solve(optprob2, NewtonTrustRegion(), callback = callback,
    maxiters = 200)
```

Note that we do not demonstrate `Newton()` because we have not found a single
case where it is competitive with the other two methods. `KrylovTrustRegion()`
is generally the fastest due to its use of Hessian-vector products.
