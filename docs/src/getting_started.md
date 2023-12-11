# [Getting Started with SciMLSensitivity: Differentiating ODE Solutions](@id auto_diff)

!!! warn
    
    This tutorial assumes familiarity with DifferentialEquations.jl.
    If you are not familiar with DifferentialEquations.jl, please consult
    [the DifferentialEquations.jl documentation](https://docs.sciml.ai/DiffEqDocs/stable/).

SciMLSensitivity.jl is a tool for obtaining derivatives of equation solvers,
such as differential equation solvers. These can be used in many ways, such as
for analyzing the local sensitivities of a system or to compute the gradients
of cost functions for model calibration and parameter estimation. In this
tutorial, we will show how to make use of the tooling in SciMLSensitivity.jl
to differentiate the ODE solvers.

!!! note
    
    SciMLSensitivity.jl applies to all equation solvers of the SciML ecosystem,
    such as linear solvers, nonlinear solvers, nonlinear optimization,
    and more. This tutorial focuses on differential equations, so please see
    the other tutorials focused on these other SciMLProblem types as necessary.
    While the interface works similarly for all problem types, these tutorials
    will showcase the aspects that are special to a given problem.

## Setup

Let's first define a differential equation we wish to solve. We will choose the
Lotka-Volterra equation. This is done via DifferentialEquations.jl using:

```@example diffode
using OrdinaryDiffEq

function lotka_volterra!(du, u, p, t)
    du[1] = dx = p[1] * u[1] - p[2] * u[1] * u[2]
    du[2] = dy = -p[3] * u[2] + p[4] * u[1] * u[2]
end
p = [1.5, 1.0, 3.0, 1.0];
u0 = [1.0; 1.0];
prob = ODEProblem(lotka_volterra!, u0, (0.0, 10.0), p)
sol = solve(prob, Tsit5(), reltol = 1e-6, abstol = 1e-6)
```

Now let's differentiate the solution to this ODE using a few different automatic
differentiation methods.

## Forward-Mode Automatic Differentiation with ForwardDiff.jl

Let's say we need the derivative of the solution with respect to the initial condition
`u0` and its parameters `p`. One of the simplest ways to do this is via ForwardDiff.jl.
All one needs to do is to use
[the ForwardDiff.jl library](https://juliadiff.org/ForwardDiff.jl/stable/) to differentiate
some function `f` which uses a differential equation `solve` inside of it. For example,
let's say we want the derivative of the first component of the ODE solution with respect to
these quantities at evenly spaced time points of `dt = 1`. We can compute this via:

```@example diffode
using ForwardDiff

function f(x)
    _prob = remake(prob, u0 = x[1:2], p = x[3:end])
    solve(_prob, Tsit5(), reltol = 1e-6, abstol = 1e-6, saveat = 1)[1, :]
end
x = [u0; p]
dx = ForwardDiff.jacobian(f, x)
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
    
    Since [the global error is 1-2 orders of magnitude higher than the local error](https://docs.sciml.ai/DiffEqDocs/stable/basics/faq/#What-does-tolerance-mean-and-how-much-error-should-I-expect), we use accuracies of 1e-6 (instead of the default 1e-3) to get reasonable sensitivities

## Reverse-Mode Automatic Differentiation

[The `solve` function is automatically compatible with AD systems like Zygote.jl](https://docs.sciml.ai/SciMLSensitivity/stable/)
and thus there is no machinery that is necessary to use other than to put `solve` inside
a function that is differentiated by Zygote. For example, the following computes the solution
to an ODE and computes the gradient of a loss function (the sum of the ODE's output at each
timepoint with dt=0.1) via the adjoint method:

```@example diffode
using Zygote, SciMLSensitivity

function sum_of_solution(u0, p)
    _prob = remake(prob, u0 = u0, p = p)
    sum(solve(_prob, Tsit5(), reltol = 1e-6, abstol = 1e-6, saveat = 0.1))
end
du01, dp1 = Zygote.gradient(sum_of_solution, u0, p)
```

Zygote.jl's automatic differentiation system is overloaded to allow SciMLSensitivity.jl
to redefine the way the derivatives are computed, allowing trade-offs between numerical
stability, memory, and compute performance, similar to how ODE solver algorithms are
chosen.

### Choosing Sensitivity Algorithms

The algorithms for differentiation calculation are called `AbstractSensitivityAlgorithms`,
or `sensealg`s for short. These are chosen by passing the `sensealg` keyword argument into solve.
Let's demonstrate this by choosing the `QuadratureAdjoint` `sensealg` for the differentiation of
this system:

```@example diffode
function sum_of_solution(u0, p)
    _prob = remake(prob, u0 = u0, p = p)
    sum(solve(_prob, Tsit5(), reltol = 1e-6, abstol = 1e-6, saveat = 0.1,
        sensealg = GaussAdjoint()))
end
du01, dp1 = Zygote.gradient(sum_of_solution, u0, p)
```

Here this computes the derivative of the output with respect to the initial
condition and the derivative with respect to the parameters respectively
using the `GaussAdjoint()`. For more information on the choices of sensitivity
algorithms, see the [reference documentation in choosing sensitivity algorithms](@ref sensitivity_diffeq).

!!! note
    
    ForwardDiff.jl's automatic differentiation system ignores the sensitivity algorithms.

## When Should You Use Forward or Reverse Mode?

Good question! The simple answer is, if you are differentiating a system of
fewer than 100 equations, use forward-mode, otherwise reverse-mode. But it can
be a lot more complicated than that! For more information, see the
[reference documentation in choosing sensitivity algorithms](@ref sensitivity_diffeq).

## And that is it! Where should you go from here?

That's all there is to the basics of differentiating the ODE solvers with SciMLSensitivity.jl.
That said, check out the following tutorials to dig into more detail:

  - See the [ODE parameter estimation tutorial](@ref odeparamestim) to learn how to fit the parameters of ODE systems
  - See the [direct sensitivity tutorial](@ref direct_sensitivity) to dig into the lower level API for more performance
