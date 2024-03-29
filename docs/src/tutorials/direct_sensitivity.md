# [Direct Sensitivity Analysis Functionality](@id direct_sensitivity)

While sensitivity analysis tooling can be used implicitly via integration with
automatic differentiation libraries, one can often times obtain more speed
and flexibility with the direct sensitivity analysis interfaces. This tutorial
demonstrates some of those functions.

## Example using an ODEForwardSensitivityProblem

Forward sensitivity analysis is performed by defining and solving an augmented
ODE. To define this augmented ODE, use the `ODEForwardSensitivityProblem` type
instead of an ODE type. For example, we generate an ODE with the sensitivity
equations attached to the Lotka-Volterra equations by:

```@example directsense
using OrdinaryDiffEq, SciMLSensitivity

function f(du, u, p, t)
    du[1] = dx = p[1] * u[1] - p[2] * u[1] * u[2]
    du[2] = dy = -p[3] * u[2] + u[1] * u[2]
end

p = [1.5, 1.0, 3.0]
prob = ODEForwardSensitivityProblem(f, [1.0; 1.0], (0.0, 10.0), p)
```

This generates a problem which the ODE solvers can solve:

```@example directsense
sol = solve(prob, DP8())
```

Note that the solution is the standard ODE system and the sensitivity system combined.
We can use the following helper functions to extract the sensitivity information:

```julia
x, dp = extract_local_sensitivities(sol)
x, dp = extract_local_sensitivities(sol, i)
x, dp = extract_local_sensitivities(sol, t)
```

In each case, `x` is the ODE values and `dp` is the matrix of sensitivities
The first gives the full timeseries of values and `dp[i]` contains the time series of the
sensitivities of all components of the ODE with respect to `i`th parameter.
The second returns the `i`th time step, while the third
interpolates to calculate the sensitivities at time `t`. For example, if we do:

```@example directsense
x, dp = extract_local_sensitivities(sol)
da = dp[1]
```

then `da` is the timeseries for ``\frac{\partial u(t)}{\partial p}``. We can
plot this

```@example directsense
using Plots
plot(sol.t, da', lw = 3)
```

transposing so that the rows (the timeseries) is plotted.

![Local Sensitivity Solution](https://user-images.githubusercontent.com/1814174/170916167-11d1b5c6-3c3c-439a-92af-d3899e24d2ad.png)

For more information on the internal representation of the `ODEForwardSensitivityProblem`
solution, see the [direct forward sensitivity analysis manual page](@ref forward_sense).

## Example using `adjoint_sensitivities` for discrete adjoints

In this example, we will show solving for the adjoint sensitivities of a discrete
cost functional. First, let's solve the ODE and get a high quality continuous
solution:

```@example directsense
function f(du, u, p, t)
    du[1] = dx = p[1] * u[1] - p[2] * u[1] * u[2]
    du[2] = dy = -p[3] * u[2] + u[1] * u[2]
end

p = [1.5, 1.0, 3.0]
prob = ODEProblem(f, [1.0; 1.0], (0.0, 10.0), p)
sol = solve(prob, Vern9(), abstol = 1e-10, reltol = 1e-10)
```

Now let's calculate the sensitivity of the ``\ell_2`` error against 1 at evenly spaced
points in time, that is:

```math
L(u,p,t)=\sum_{i=1}^{n}\frac{\Vert1-u(t_{i},p)\Vert^{2}}{2}
```

for ``t_i = 0.5i``. This is the assumption that the data is `data[i]=1.0`.
For this function, notice we have that:

```math
\begin{aligned}
dg_{1}&=u_{1}-1 \\
dg_{2}&=u_{2}-1 \\
& \quad \vdots
\end{aligned}
```

and thus:

```@example directsense
dg(out, u, p, t, i) = (out .= u .- 1.0)
```

Also, we can omit `dgdp`, because the cost function doesn't dependent on `p`.
If we had data, we'd just replace `1.0` with `data[i]`. To get the adjoint
sensitivities, call:

```@example directsense
ts = 0:0.5:10
res = adjoint_sensitivities(sol, Vern9(), t = ts, dgdu_discrete = dg, abstol = 1e-14,
    reltol = 1e-14)
```

This is super high accuracy. As always, there's a tradeoff between accuracy
and computation time. We can check this almost exactly matches the
autodifferentiation and numerical differentiation results:

```@example directsense
using ForwardDiff, Calculus, ReverseDiff, Tracker
function G(p)
    tmp_prob = remake(prob, u0 = convert.(eltype(p), prob.u0), p = p)
    sol = solve(tmp_prob, Vern9(), abstol = 1e-14, reltol = 1e-14, saveat = ts,
        sensealg = SensitivityADPassThrough())
    A = convert(Array, sol)
    sum(((1 .- A) .^ 2) ./ 2)
end
res2 = ForwardDiff.gradient(G, [1.5, 1.0, 3.0])
```

and see this gives the same values.
