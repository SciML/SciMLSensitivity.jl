# [Sensitivity Algorithms for Differential Equations with Automatic Differentiation (AD)](@id sensitivity_diffeq)

SciMLSensitivity.jl's high-level interface allows for specifying a
sensitivity algorithm (`sensealg`) to control the method by which
`solve` is differentiated in an automatic differentiation (AD)
context by a compatible AD library. The underlying algorithms then
use the direct interface methods, like `ODEForwardSensitivityProblem`
and `adjoint_sensitivities`, to compute the derivatives without
requiring the user to do any of the setup.

Current AD libraries whose calls are captured by the sensitivity
system are:

  - [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl)
  - [Zygote.jl](https://fluxml.ai/Zygote.jl/stable/)
  - [Diffractor.jl](https://github.com/JuliaDiff/Diffractor.jl)
  - [ReverseDiff.jl](https://github.com/JuliaDiff/ReverseDiff.jl)

## Using and Controlling Sensitivity Algorithms within AD

Take for example this simple differential equation solve on Lotka-Volterra:

```julia
using SciMLSensitivity, OrdinaryDiffEq, Zygote

function fiip(du, u, p, t)
    du[1] = dx = p[1] * u[1] - p[2] * u[1] * u[2]
    du[2] = dy = -p[3] * u[2] + p[4] * u[1] * u[2]
end
p = [1.5, 1.0, 3.0, 1.0];
u0 = [1.0; 1.0];
prob = ODEProblem(fiip, u0, (0.0, 10.0), p)
sol = solve(prob, Tsit5())
loss(u0, p) = sum(solve(prob, Tsit5(), u0 = u0, p = p, saveat = 0.1))
du0, dp = Zygote.gradient(loss, u0, p)
```

This will compute the gradient of the loss function "sum of the values of the
solution to the ODE at timepoints dt=0.1" using an adjoint method, where `du0`
is the derivative of the loss function with respect to the initial condition
and `dp` is the derivative of the loss function with respect to the parameters.

Because the gradient is calculated by `Zygote.gradient` and Zygote.jl is one of
the compatible AD libraries, this derivative calculation will be captured
by the `sensealg` system, and one of SciMLSensitivity.jl's adjoint overloads
will be used to compute the derivative. By default, if the `sensealg` keyword
argument is not defined, then a smart polyalgorithm is used to automatically
determine the most appropriate method for a given equation.

Likewise, the `sensealg` argument can be given to directly control the method
by which the derivative is computed. For example:

```julia
function loss(u0, p)
    sum(solve(prob, Tsit5(), u0 = u0, p = p, saveat = 0.1, sensealg = ForwardSensitivity()))
end
du0, dp = Zygote.gradient(loss, u0, p)
```

would do reverse-mode automatic differentiation of the loss function, but when reversing
over the ODE solve, it would do forward sensitivity analysis to compute the required
pullbacks, effectively creating an algorithm that mixes forward and reverse differentiation.

## Choosing a Sensitivity Algorithm

There are two classes of algorithms: the continuous sensitivity analysis
methods, and the discrete sensitivity analysis methods (direct automatic
differentiation). Generally:

  - [Continuous sensitivity analysis are more efficient while the discrete
    sensitivity analysis is more stable](https://arxiv.org/abs/2001.04385)
    (full discussion is in the appendix of that paper)

  - Continuous sensitivity analysis methods only support a subset of
    equations, which currently includes:
    
      + ODEProblem (with mass matrices for differential-algebraic equations (DAEs)
      + SDEProblem
      + SteadyStateProblem / NonlinearProblem
  - Discrete sensitivity analysis methods only support a subset of algorithms,
    namely, the pure Julia solvers which are written generically.

For an analysis of which methods will be most efficient for computing the
solution derivatives for a given problem, consult our analysis
[in this arXiv paper](https://arxiv.org/abs/1812.01892). A general rule of thumb
is:

  - `ForwardDiffSensitivity` is the fastest for differential equations with small
    numbers of parameters (<100) and can be used on any differential equation
    solver that is native Julia. If the chosen ODE solver is incompatible
    with direct automatic differentiation, `ForwardSensitivty` may be used instead.
  - Adjoint sensitivity analysis is the fastest when the number of parameters is
    sufficiently large. `GaussAdjoint` should be generally preferred. `BacksolveAdjoint`
    uses the least memory but on very stiff problems it may be unstable and
    requires many checkpoints, while `InterpolatingAdjoint` is more compute intensive
    than `GaussAdjoint` but allows for checkpointing which can reduce the
    total memory requirement (`GaussAdjoint` in the future will support checkpointing
    in which case `QuadratureAdjoint` and `InterpolatingAdjoint` would only be
    recommending in rare benchmarking scenarios).
  - The methods which use direct automatic differentiation (`ReverseDiffAdjoint`,
    `TrackerAdjoint`, `ForwardDiffSensitivity`, and `ZygoteAdjoint`) support
    the full range of DifferentialEquations.jl features (SDEs, DDEs, events, etc.),
    but only work on native Julia solvers.
  - For non-ODEs with large numbers of parameters, `TrackerAdjoint` in out-of-place
    form may be the best performer on GPUs, and `ReverseDiffAdjoint`
  - `TrackerAdjoint` is able to use a `TrackedArray` form with out-of-place
    functions `du = f(u,p,t)` but requires an `Array{TrackedReal}` form for
    `f(du,u,p,t)` mutating `du`. The latter has much more overhead, and should be
    avoided if possible. When solving non-ODEs with lots of parameters, using
    `TrackerAdjoint` with an out-of-place definition may currently be the best option.

!!! note
    
    Compatibility with direct automatic differentiation algorithms (`ForwardDiffSensitivity`,
    `ReverseDiffAdjoint`, etc.) can be queried using the
    `SciMLBase.isautodifferentiable(::SciMLAlgorithm)` trait function.

If the chosen algorithm is a continuous sensitivity analysis algorithm, then an `autojacvec`
argument can be given for choosing how the Jacobian-vector product (`J*v`) or vector-Jacobian
product (`J'*v`) calculation is computed. For the forward sensitivity methods, `autojacvec=true`
is the most efficient, though `autojacvec=false` is slightly less accurate but very close in
efficiency. For adjoint methods, it's more complicated and dependent on the way that the user's
`f` function is implemented:

  - `EnzymeVJP()` is the most efficient if it's applicable on your equation.
  - If your function has no branching (no if statements) but uses mutation, `ReverseDiffVJP(true)`
    will be the most efficient after Enzyme. Otherwise, `ReverseDiffVJP()`, but you may wish to
    proceed with eliminating mutation as without compilation enabled this can be slow.
  - If you are on the CPU or GPU and your function is very vectorized and has no mutation, choose `ZygoteVJP()`.
  - Else fallback to `TrackerVJP()` if Zygote does not support the function.

## Special Notes on Non-ODE Differential Equation Problems

While all of the choices are compatible with ordinary differential
equations, specific notices apply to other forms:

### Differential-Algebraic Equations

We note that while all continuous adjoints are compatible with index-1 DAEs via the
[derivation in the universal differential equations paper](https://arxiv.org/abs/2001.04385)
(note the reinitialization), we do not recommend `BacksolveAdjoint`
on DAEs because the stiffness inherent in these problems tends to
cause major difficulties with the accuracy of the backwards solution
due to reinitialization of the algebraic variables.

### Stochastic Differential Equations

We note that all of the adjoints except `QuadratureAdjoint` are applicable
to stochastic differential equations.

### Delay Differential Equations

We note that only the discretize-then-optimize methods are applicable
to delay differential equations. Constant lag and variable lag
delay differential equation parameters can be estimated, but the lag
times themselves are unable to be estimated through these automatic
differentiation techniques.

### Hybrid Equations (Equations with events/callbacks) and Jump Equations

`ForwardDiffSensitivity` can differentiate code with callbacks when `convert_tspan=true`.
`ForwardSensitivity` is incompatible with hybrid equations. The shadowing methods are
incompatible with callbacks. All methods based on discrete adjoint sensitivity analysis
via automatic differentiation, like `ReverseDiffAdjoint`, `TrackerAdjoint`, or
`QuadratureAdjoint` are fully compatible with events. This applies to ODEs, SDEs, DAEs,
and DDEs. The continuous adjoint sensitivities `BacksolveAdjoint`, `InterpolatingAdjoint`,
`GaussAdjoint`, and `QuadratureAdjoint` are compatible with events for ODEs. `BacksolveAdjoint` and
`InterpolatingAdjoint` can also handle events for SDEs. Use `BacksolveAdjoint` if
the event terminates the time evolution and several states are saved. Currently,
the continuous adjoint sensitivities do not support multiple events per time point.

## Manual VJPs

Note that when defining your differential equation, the vjp can be
manually overwritten by providing the `AbstractSciMLFunction` definition
with  a `vjp(u,p,t)` that returns a tuple `f(u,p,t),v->J*v` in the form of
[ChainRules.jl](https://www.juliadiff.org/ChainRulesCore.jl/stable/).
When this is done, the choice of `ZygoteVJP` will utilize your VJP
function during the internal steps of the adjoint. This is useful for
models where automatic differentiation may have trouble producing
optimal code. This can be paired with
[ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/stable/)
for producing hyper-optimized, sparse, and parallel VJP functions utilizing
the automated symbolic conversions.

## Sensitivity Algorithms

The following algorithm choices exist for `sensealg`. See
[the sensitivity mathematics page](@ref sensitivity_math) for more details on
the definition of the methods.

```@docs
ForwardSensitivity
ForwardDiffSensitivity
BacksolveAdjoint
GaussAdjoint
InterpolatingAdjoint
QuadratureAdjoint
ReverseDiffAdjoint
TrackerAdjoint
ZygoteAdjoint
ForwardLSS
AdjointLSS
NILSS
NILSAS
```

## Vector-Jacobian Product (VJP) Choices

```@docs
ZygoteVJP
EnzymeVJP
TrackerVJP
ReverseDiffVJP
```

## More Details on Sensitivity Algorithm Choices

The following section describes a bit more details to consider when choosing
a sensitivity algorithm.

### Optimize-then-Discretize

[The original neural ODE paper](https://arxiv.org/abs/1806.07366)
popularized optimize-then-discretize with O(1) adjoints via backsolve.
This is the methodology `BacksolveAdjoint`
When training non-stiff neural ODEs, `BacksolveAdjoint` with `ZygoteVJP`
is generally the fastest method. Additionally, this method does not
require storing the values of any intermediate points and is thus the
most memory efficient. However, `BacksolveAdjoint` is prone
to instabilities whenever the Lipschitz constant is sufficiently large,
like in stiff equations, PDE discretizations, and many other contexts,
so it is not used by default. When training a neural ODE for machine
learning applications, the user should try `BacksolveAdjoint` and see
if it is sufficiently accurate on their problem. More details on this
topic can be found in
[Stiff Neural Ordinary Differential Equations](https://aip.scitation.org/doi/10.1063/5.0060697)

Note that DiffEqFlux's implementation of `BacksolveAdjoint` includes
an extra feature `BacksolveAdjoint(checkpointing=true)` which mixes
checkpointing with `BacksolveAdjoint`. What this method does is that,
at `saveat` points, values from the forward pass are saved. Since the
reverse solve should numerically be the same as the forward pass, issues
with divergence of the reverse pass are mitigated by restarting the
reverse pass at the `saveat` value from the forward pass. This reduces
the divergence and can lead to better gradients at the cost of higher
memory usage due to having to save some values of the forward pass.
This can stabilize the adjoint in some applications, but for highly
stiff applications the divergence can be too fast for this to work in
practice.

To avoid the issues of backwards solving the ODE, `InterpolatingAdjoint`, `QuadratureAdjoint`, and `GaussAdjoint` utilize information from the forward pass.
By default, these methods utilize the [continuous solution](https://docs.sciml.ai/DiffEqDocs/stable/basics/solution/#Interpolations-1)
provided by DifferentialEquations.jl in the calculations of the
adjoint pass. `QuadratureAdjoint` uses this to build a continuous
function for the solution of the adjoint equation and then performs an
adaptive quadrature via [Integrals.jl](https://docs.sciml.ai/Integrals/stable/);
`GaussAdjoint` computes the integrand with a callback that performs
adaptive quadrature via [Integrals.jl](https://docs.sciml.ai/Integrals/stable/)
during the adjoint equation solve;
`InterpolatingAdjoint` appends the integrand to the ODE, so it's
computed simultaneously to the Lagrange multiplier. When memory is
not an issue, we find that the `QuadratureAdjoint` approach tends to
be the most efficient as it has a significantly smaller adjoint
differential equation and the quadrature converges very fast, but this
form requires holding the full continuous solution of the adjoint which
can be a significant burden for large parameter problems. The
`InterpolatingAdjoint` is thus a compromise between memory efficiency
and compute efficiency, and is in the same spirit as [CVODES](https://computing.llnl.gov/projects/sundials).
`GaussAdjoint` combines the advantages of both of these approaaches,
having a small adjoint differential equation while not requiring 
saving the full continuous solution of the adjoint problem.

However, if the memory cost of the `InterpolatingAdjoint` is too high,
checkpointing can be used via `InterpolatingAdjoint(checkpointing=true)`.
When this is used, the checkpoints default to `sol.t` of the forward
pass (i.e. the saved timepoints usually set by `saveat`). Then in the
adjoint, intervals of `sol.t[i-1]` to `sol.t[i]` are re-solved in order
to obtain a short interpolation which can be utilized in the adjoints.
This at most results in two full solves of the forward pass, but
dramatically reduces the computational cost while being a low-memory
format. This is the preferred method for highly stiff equations
when memory is an issue, i.e. stiff PDEs or large neural DAEs.

For forward-mode, the `ForwardSensitivty` is the version that performs
the optimize-then-discretize approach. In this case, `autojacvec` corresponds
to the method for computing `J*v` within the forward sensitivity equations,
which is either `true` or `false` for whether to use Jacobian-free
forward-mode AD (via ForwardDiff.jl) or Jacobian-free numerical
differentiation.

### Discretize-then-Optimize

In this approach, the discretization is done first and then optimization
is done on the discretized system. While traditionally this can be
done discrete sensitivity analysis, this can equivalently be done
by automatic differentiation on the solver itself. `ReverseDiffAdjoint`
performs reverse-mode automatic differentiation on the solver via
[ReverseDiff.jl](https://juliadiff.org/ReverseDiff.jl/),
`ZygoteAdjoint` performs reverse-mode automatic
differentiation on the solver via
[Zygote.jl](https://fluxml.ai/Zygote.jl/latest/), and `TrackerAdjoint`
performs reverse-mode automatic differentiation on the solver via
[Tracker.jl](https://github.com/FluxML/Tracker.jl). In addition,
`ForwardDiffSensitivty` performs forward-mode automatic differentiation
on the solver via [ForwardDiff.jl](https://juliadiff.org/ForwardDiff.jl/stable/).

We note that many studies have suggested that [this approach produces
more accurate gradients than the optimize-than-discretize approach](https://arxiv.org/abs/2005.13420)
