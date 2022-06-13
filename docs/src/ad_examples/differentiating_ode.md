# [Differentiating an ODE Solution with Automatic Differentiation](@id auto_diff)

!!! note

      This tutorial assumes familiarity with DifferentialEquations.jl
      If you are not familiar with DifferentialEquations.jl, please consult
      [the DifferentialEquations.jl documentation](https://diffeq.sciml.ai/stable/)

In this tutorial we will introduce how to use local sensitivity analysis via
automatic differentiation. The automatic differentiation interfaces are the
most common ways that local sensitivity analysis is done. It's fairly fast
and flexible, but most notably, it's a very small natural extension to the 
normal differential equation solving code and is thus the easiest way to
do most things.

## Setup

Let's first define a differential equation we wish to solve. We will choose the
Lotka-Volterra equation. This is done via DifferentialEquations.jl using:

```@example diffode
using DifferentialEquations

function lotka_volterra!(du,u,p,t)
  du[1] = dx = p[1]*u[1] - p[2]*u[1]*u[2]
  du[2] = dy = -p[3]*u[2] + p[4]*u[1]*u[2]
end
p = [1.5,1.0,3.0,1.0]; u0 = [1.0;1.0]
prob = ODEProblem(lotka_volterra!,u0,(0.0,10.0),p)
sol = solve(prob,Tsit5(),reltol=1e-6,abstol=1e-6)
```

Now let's differentiate the solution to this ODE using a few different automatic
differentiation methods.

## Forward-Mode Automatic Differentiation with ForwardDiff.jl

Let's say we need the derivative of the solution with respect to the initial condition
`u0` and its parameters `p`. One of the simplest ways to do this is via ForwardDiff.jl.
To do this, all that one needs to do is use 
[the ForwardDiff.jl library](https://github.com/JuliaDiff/ForwardDiff.jl) to differentiate
some function `f` which uses a differential equation `solve` inside of it. For example,
let's say we want the derivative of the first component of ODE solution with respect to 
these quantities at evenly spaced time points of `dt = 1`. We can compute this via:

```@example diffode
using ForwardDiff

function f(x)
    _prob = remake(prob,u0=x[1:2],p=x[3:end])
    solve(_prob,Tsit5(),reltol=1e-6,abstol=1e-6,saveat=1)[1,:]
end
x = [u0;p]
dx = ForwardDiff.jacobian(f,x)
```

Let's dig into what this is saying a bit. `x` is a vector which concatenates the initial condition
and parameters, meaning that the first 2 values are the initial conditions and the last 4 are the
parameters. We use the `remake` function to build a function `f(x)` which uses these new initial
conditions and parameters to solve the differential equation and return the time series of the first
component. 

Then `ForwardDiff.jacobian(f,x)` computes the Jacobian of `f` with respect to `x`. The
output `dx[i,j]` corresponds to the derivative of the solution of the first component at time `t=j-1`
with respect to `x[i]`. For example, `dx[3,2]` is the derivative of the first component of the
solution at time `t=1` with respect to `p[1]`.

!!! note

      Since [the global error is 1-2 orders of magnitude higher than the local error](https://diffeq.sciml.ai/stable/basics/faq/#What-does-tolerance-mean-and-how-much-error-should-I-expect), we use accuracies of 1e-6 (instead of the default 1e-3) to get reasonable sensitivities

## Reverse-Mode Automatic Differentiation

[The `solve` function is automatically compatible with AD systems like Zygote.jl](https://diffeq.sciml.ai/latest/analysis/sensitivity/)
and thus there is no machinery that is necessary to use other than to put `solve` inside of
a function that is differentiated by Zygote. For example, the following computes the solution 
to an ODE and computes the gradient of a loss function (the sum of the ODE's output at each 
timepoint with dt=0.1) via the adjoint method:

```@example diffode
function sum_of_solution(u0,p)
  _prob = remake(prob,u0=u0,p=p)
  sum(solve(_prob,Tsit5(),reltol=1e-6,abstol=1e-6,saveat=0.1))
end
du01,dp1 = Zygote.gradient(sum_of_solution,u0,p)
```

Zygote.jl's automatic differentiation system is overloaded to allow SciMLSensitivity.jl
to redefine the way the derivatives are computed, allowing trade-offs between numerical
stability, memory, and compute performance, similar to how ODE solver algorithms are
chosen. The algorithms for differentiation calculation are called `AbstractSensitivityAlgorithms`,
or `sensealg`s for short. These are choosen by passing the `sensealg` keyword argument into solve.

Let's demonstrate this by choosing the `QuadratureAdjoint` `sensealg` for the differentiation of
this system:

```@example diffode
function sum_of_solution(u0,p)
  _prob = remake(prob,u0=u0,p=p)
  sum(solve(_prob,Tsit5(),reltol=1e-6,abstol=1e-6,saveat=0.1,sensealg=QuadratureAdjoint()))
end
du01,dp1 = Zygote.gradient(sum_of_solution,u0,p)
```

Here this computes the derivative of the output with respect to the initial
condition and the the derivative with respect to the parameters respectively
using the `QuadratureAdjoint()`. For more information on the choices of sensitivity
algorithms, see the [reference documentation in choosing sensitivity algorithms](@ref sensitivity_diffeq)

## When Should You Use Forward or Reverse Mode?

Good question! The simple answer is, if you are differentiating a system of
100 equations or less, use forward-mode, otherwise reverse-mode. But it can
be a lot more complicated than that! For more information, see the 
[reference documentation in choosing sensitivity algorithms](@ref sensitivity_diffeq)