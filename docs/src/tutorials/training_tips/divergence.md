# Handling Divergent and Unstable Trajectories

It is not uncommon for a set of parameters in an ODE model to simply give a
divergent trajectory. If the rate of growth compounds and outpaces the rate
of decay, you will end up at infinity in finite time. This it is not uncommon
to see divergent trajectories in the optimization of parameters, as many times
an optimizer can take an excursion into a parameter regime which simply gives
a model with an infinite solution.

This can be addressed by using the retcode system. In DifferentialEquations.jl,
[ReturnCodes](https://docs.sciml.ai/DiffEqDocs/stable/basics/solution/#retcodes) detail
the status of the returned solution. Thus if the retcode corresponds to a
failure, we can use this to give an infinite loss and effectively discard the
parameters. This is shown in the loss function:

```julia
function loss(p)
    tmp_prob = ODE.remake(prob, p = p)
    tmp_sol = ODE.solve(tmp_prob, ODE.Tsit5(), saveat = 0.1)
    if tmp_sol.retcode == SciMLBase.ReturnCode.Success
        return sum(abs2, Array(tmp_sol) - dataset)
    else
        return Inf
    end
end
```

A full example making use of this trick is:

```@example divergence
import OrdinaryDiffEq as ODE, SciMLSensitivity as SMS, SciMLBase, Optimization as OPT, OptimizationOptimisers as OPO, Plots

function lotka_volterra!(du, u, p, t)
    rab, wol = u
    α, β, γ, δ = p
    du[1] = drab = α * rab - β * rab * wol
    du[2] = dwol = γ * rab * wol - δ * wol
    nothing
end

u0 = [1.0, 1.0]
tspan = (0.0, 10.0)
p = [1.5, 1.0, 3.0, 1.0]
prob = ODE.ODEProblem(lotka_volterra!, u0, tspan, p)
sol = ODE.solve(prob, ODE.Tsit5(); saveat = 0.1)
Plots.plot(sol)

dataset = Array(sol)
Plots.scatter!(sol.t, dataset')

tmp_prob = ODE.remake(prob, p = [1.2, 0.8, 2.5, 0.8])
tmp_sol = ODE.solve(tmp_prob, ODE.Tsit5())
Plots.plot(tmp_sol)
Plots.scatter!(sol.t, dataset')

function loss(p)
    tmp_prob = ODE.remake(prob, p = p)
    tmp_sol = ODE.solve(tmp_prob, ODE.Tsit5(), saveat = 0.1)
    if tmp_sol.retcode == SciMLBase.ReturnCode.Success
        return sum(abs2, Array(tmp_sol) - dataset)
    else
        return Inf
    end
end

pinit = [1.2, 0.8, 2.5, 0.8]
adtype = OPT.AutoZygote()
optf = OPT.OptimizationFunction((x, p) -> loss(x), adtype)

optprob = OPT.OptimizationProblem(optf, pinit)
res = OPT.solve(optprob, OPO.Adam(), maxiters = 1000)

# res = OPT.solve(optprob,NLopt.LD_LBFGS(), maxiters = 1000) ### errors!
```

You might notice that `AutoZygote` (default) fails for the above `OPT.solve` call
with Optim's optimizers, which happens because of Zygote's behavior for zero gradients, in
which case it returns `nothing`. To avoid such issues, you can just use a different version
of the same check which compares the size of the obtained solution and the data we have,
shown below, which is easier to AD.

```julia
function loss(p)
    tmp_prob = ODE.remake(prob, p = p)
    tmp_sol = ODE.solve(tmp_prob, ODE.Tsit5(), saveat = 0.1)
    if size(tmp_sol) == size(dataset)
        return sum(abs2, Array(tmp_sol) .- dataset)
    else
        return Inf
    end
end
```
