# Bouncing Ball Hybrid ODE Optimization

The bouncing ball is a classic hybrid ODE which can be represented in
the [DifferentialEquations.jl event handling system](https://diffeq.sciml.ai/stable/features/callback_functions/). This can be applied to ODEs, SDEs, DAEs, DDEs,
and more. Let's now add the DiffEqFlux machinery to this
problem in order to optimize the friction that's required to match
data. Assume we have data for the ball's height after 15 seconds. Let's
first start by implementing the ODE:

```@example bouncing_ball
using Optimization, OptimizationPolyalgorithms, DiffEqSensitivity, DifferentialEquations

function f(du,u,p,t)
  du[1] = u[2]
  du[2] = -p[1]
end

function condition(u,t,integrator) # Event when event_f(u,t) == 0
  u[1]
end

function affect!(integrator)
  integrator.u[2] = -integrator.p[2]*integrator.u[2]
end

callback = ContinuousCallback(condition,affect!)
u0 = [50.0,0.0]
tspan = (0.0,15.0)
p = [9.8, 0.8]
prob = ODEProblem(f,u0,tspan,p)
sol = solve(prob,Tsit5(),callback=callback)
```

Here we have a friction coefficient of `0.8`. We want to refine this
coefficient to find the value so that the predicted height of the ball
at the endpoint is 20. We do this by minimizing a loss function against
the value 20:

```@example bouncing_ball
function loss(θ)
  sol = solve(prob,Tsit5(),p=[9.8,θ[1]],callback=callback)
  target = 20.0
  abs2(sol[end][1] - target)
end

loss([0.8])
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p)->loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, [0.8])
@time res = Optimization.solve(optprob, PolyOpt(), maxiters = 300)
@show res.u # [0.866554105436901]
```

This runs in about `0.091215 seconds (533.45 k allocations: 80.717 MiB)` and finds
an optimal drag coefficient.

## Note on Sensitivity Methods

The continuous adjoint sensitivities `BacksolveAdjoint`, `InterpolatingAdjoint`,
and `QuadratureAdjoint` are compatible with events for ODEs. `BacksolveAdjoint` and
`InterpolatingAdjoint` can also handle events for SDEs. Use `BacksolveAdjoint` if
the event terminates the time evolution and several states are saved. Currently,
the continuous adjoint sensitivities do not support multiple events per time point.

All methods based on discrete sensitivity analysis via automatic differentiation,
like `ReverseDiffAdjoint`, `TrackerAdjoint`, or `ForwardDiffSensitivity` are the methods
to use (and `ReverseDiffAdjoint` is demonstrated above), are compatible with events.
This applies to SDEs, DAEs, and DDEs as well.
