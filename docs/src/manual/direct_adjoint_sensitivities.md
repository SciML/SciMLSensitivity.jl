# [Direct Adjoint Sensitivities of Differential Equations](@id adjoint_sense)

## First Order Adjoint Sensitivities

The `SensitivityFunction` interface is developer API for packages that build
adjoint, callback, or solver integrations. Application code should select a
documented sensitivity algorithm through `solve` or use the documented problem
wrappers instead of subtyping `SensitivityFunction` directly.

```@docs
adjoint_sensitivities
ODEAdjointProblem
SDEAdjointProblem
RODEAdjointProblem
DAEAdjointProblem
AdjointSensitivityIntegrand
SensitivityFunction
SciMLSensitivity.getprob
SciMLSensitivity.inplace_sensitivity
StochasticTransformedFunction
```

## Second Order Adjoint Sensitivities

```@docs
second_order_sensitivities
second_order_sensitivity_product
```
