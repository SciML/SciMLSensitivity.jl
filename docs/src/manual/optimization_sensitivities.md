# [Sensitivity Algorithms for Optimization Problems](@id sensitivity_optimization)

SciMLSensitivity provides adjoint algorithms for differentiating through the optimum
`u*(p)` of a parameterized [`OptimizationProblem`](https://docs.sciml.ai/Optimization/stable/),
giving `dG/dp` for any downstream loss `G(u*(p))` via implicit differentiation rather
than by differentiating through the iterations of the optimizer.

  - `UnconstrainedOptimizationAdjoint` handles unconstrained problems by treating the
    stationarity condition `∇f(u*, p) = 0` as a steady-state nonlinear system and
    reusing the `SteadyStateAdjoint` machinery.
  - `OptimizationAdjoint` handles problems with equality, two-sided inequality, and
    variable-bound constraints by implicit differentiation of the KKT first-order
    optimality conditions. It detects the active inequality set at `u*`, recovers
    multipliers from the stationarity equation, and solves a single symmetric KKT
    linear system to produce the adjoint.

```@docs
UnconstrainedOptimizationAdjoint
OptimizationAdjoint
```
