# Differentiating an ODE Solution

```julia
using DiffEqSensitivity, OrdinaryDiffEq, ForwardDiff, Zygote

function fiip(du,u,p,t)
  du[1] = dx = p[1]*u[1] - p[2]*u[1]*u[2]
  du[2] = dy = -p[3]*u[2] + p[4]*u[1]*u[2]
end
p = [1.5,1.0,3.0,1.0]; u0 = [1.0;1.0]
prob = ODEProblem(fiip,u0,(0.0,10.0),p)
sol = solve(prob,Tsit5(),reltol=1e-6,abstol=1e-6)
```

!!! note

      Since [the global error is 1-2 orders of magnitude higher than the local error](https://diffeq.sciml.ai/stable/basics/faq/#What-does-tolerance-mean-and-how-much-error-should-I-expect), we use accuracies of 1e-6 (instead of the default 1e-3) to get reasonable sensitivities

## Forward-Mode Automatic Differentiation

If we want to perturb `u0` and `p` in a gradient calculation then we can do forward-mode
automatic differentiation:

```julia
function sum_of_solution(x)
    _prob = remake(prob,u0=x[1:2],p=x[3:end])
    sum(solve(_prob,Tsit5(),reltol=1e-6,abstol=1e-6,saveat=0.1))
end
dx = ForwardDiff.gradient(sum_of_solution,[u0;p])
```

## Reverse-Mode Automatic Differentiation

Similarly, we can use reverse-mode automatic differentiation:

```julia
function sum_of_solution(u0,p)
  _prob = remake(prob,u0=u0,p=p)
  sum(solve(_prob,Tsit5(),reltol=1e-6,abstol=1e-6,saveat=0.1,sensealg=QuadratureAdjoint()))
end
du01,dp1 = Zygote.gradient(sum_of_solution,u0,p)
```

Or we can use the `u0` and `p` keyword argument short hands to tell
it to replace `u0` and `p` by the inputs:

```julia
du01,dp1 = Zygote.gradient((u0,p)->sum(solve(prob,Tsit5(),u0=u0,p=p,
                            saveat=0.1,sensealg=QuadratureAdjoint())),u0,p)
```

Here this computes the derivative of the output with respect to the initial
condition and the the derivative with respect to the parameters respectively
using the `QuadratureAdjoint()`.
