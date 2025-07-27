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
import SciMLSensitivity as SMS
import Lux
import ComponentArrays as CA
import Optimization as OPT
import OptimizationOptimisers as OPO
import OrdinaryDiffEq as ODE
import Plots
import Random
import OptimizationOptimJL as OOJ

u0 = Float32[2.0; 0.0]
datasize = 30
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2], length = datasize)

function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u .^ 3)'true_A)'
end

prob_trueode = ODE.ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(ODE.solve(prob_trueode, ODE.Tsit5(), saveat = tsteps))

dudt2 = Lux.Chain(x -> x .^ 3, Lux.Dense(2, 50, tanh), Lux.Dense(50, 2))
ps, st = Lux.setup(Random.default_rng(), dudt2)
function neuralodefunc(u, p, t)
    dudt2(u, p, st)[1]
end
function prob_neuralode(u0, p)
    prob = ODE.ODEProblem(neuralodefunc, u0, tspan, p)
    sol = ODE.solve(prob, ODE.Tsit5(), saveat = tsteps)
end
ps = CA.ComponentArray(ps)

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
    plt = Plots.scatter(tsteps, ode_data[1, :], label = "data")
    Plots.scatter!(plt, tsteps, pred[1, :], label = "prediction")
    push!(list_plots, plt)
    if doplot
        display(Plots.plot(plt))
    end

    return l < 0.01
end

adtype = OPT.AutoZygote()
optf = OPT.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)

optprob1 = OPT.OptimizationProblem(optf, ps)
pstart = OPT.solve(
    optprob1, OPO.Adam(0.01), callback = callback, maxiters = 100).u

optprob2 = OPT.OptimizationProblem(optf, pstart)
pmin = OPT.solve(optprob2, OOJ.NewtonTrustRegion(), callback = callback,
    maxiters = 200)
```

Note that we do not demonstrate `Newton()` because we have not found a single
case where it is competitive with the other two methods. `KrylovTrustRegion()`
is generally the fastest due to its use of Hessian-vector products.
