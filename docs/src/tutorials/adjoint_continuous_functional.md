# [Adjoint Sensitivity Analysis of Continuous Functionals](@id continuous_loss)

[The automatic differentiation tutorial](@ref auto_diff) demonstrated
how to use AD packages like ForwardDiff.jl and Zygote.jl to compute derivatives
of differential equation solutions with respect to initial conditions and
parameters. The subsequent [direct sensitivity analysis tutorial](@ref direct_sensitivity)
showed how to directly use the SciMLSensitivity.jl internals to define and solve
the augmented differential equation systems which are used in the automatic
differentiation process.

While these internal functions give more flexibility, the previous demonstration
focused on a case which was possible via automatic differentiation: discrete cost functionals.
What is meant by discrete cost functionals is differentiation of a cost which uses a finite
number of time points. In the automatic differentiation case, these finite time points are
the points returned by `solve`, i.e. those chosen by the `saveat` option in the solve call.
In the direct adjoint sensitivity tooling, these were the time points chosen by the `ts`
vector.

However, there is an expanded set of cost functionals supported by SciMLSensitivity.jl,
continuous cost functionals, which are not possible through automatic differentiation
interfaces. In an abstract sense, a continuous cost functional is a total cost ``G``
defined as the integral of the instantaneous cost ``g`` at all time points. In other words,
the total cost is defined as:

```math
G(u,p)=G(u(\cdot,p))=\int_{t_{0}}^{T}g(u(t,p),p)dt
```

Notice that this cost function cannot accurately be computed using only estimates of `u`
at discrete time points. The purpose of this tutorial is to demonstrate how such cost
functionals can be easily evaluated using the direct sensitivity analysis interfaces.

## Example: Continuous Functionals with Forward Sensitivity Analysis via Interpolation

Evaluating continuous cost functionals with forward sensitivity analysis is rather
straightforward, since one can simply use the fact that the solution from
`ODEForwardSensitivityProblem` is continuous when `dense=true`. For example,

```@example continuousadjoint
using OrdinaryDiffEq, SciMLSensitivity

function f(du, u, p, t)
    du[1] = dx = p[1] * u[1] - p[2] * u[1] * u[2]
    du[2] = dy = -p[3] * u[2] + u[1] * u[2]
end

p = [1.5, 1.0, 3.0]
prob = ODEForwardSensitivityProblem(f, [1.0; 1.0], (0.0, 10.0), p)
sol = solve(prob, DP8())
```

gives a continuous solution `sol(t)` with the derivative at each time point. This
can then be used to define a continuous cost function via
[Integrals.jl](https://docs.sciml.ai/Integrals/stable/), though the derivative would
need to be manually defined using the extra sensitivity terms.

## Example: Continuous Adjoints on an Energy Functional

Continuous adjoints on a continuous functional are more automatic than forward mode.
In this case, we'd like to calculate the adjoint sensitivity of the scalar energy
functional:

```math
G(u,p)=\int_{0}^{T}\frac{\sum_{i=1}^{n}u_{i}^{2}(t)}{2}dt
```

which is:

```@example continuousadjoint
g(u, p, t) = sum(u .^ 2) ./ 2
```

Notice that the gradient of this function with respect to the state `u` is:

```@example continuousadjoint
function dg(out, u, p, t)
    out[1] = u[1]
    out[2] = u[2]
end
```

To get the adjoint sensitivities, we call:

```@example continuousadjoint
prob = ODEProblem(f, [1.0; 1.0], (0.0, 10.0), p)
sol = solve(prob, DP8())
res = adjoint_sensitivities(sol, Vern9(), dgdu_continuous = dg, g = g, abstol = 1e-8,
    reltol = 1e-8)
```

Notice that we can check this against autodifferentiation and numerical
differentiation as follows:

```@example continuousadjoint
using QuadGK, ForwardDiff, Calculus
function G(p)
    tmp_prob = remake(prob, p = p)
    sol = solve(tmp_prob, Vern9(), abstol = 1e-14, reltol = 1e-14)
    res, err = quadgk((t) -> sum(sol(t) .^ 2) ./ 2, 0.0, 10.0, atol = 1e-14, rtol = 1e-10)
    res
end
res2 = ForwardDiff.gradient(G, [1.5, 1.0, 3.0])
res3 = Calculus.gradient(G, [1.5, 1.0, 3.0])
```
