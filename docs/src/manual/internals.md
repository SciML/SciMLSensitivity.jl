# [Developer Internals](@id sensitivity_internals)

These names are **internal, not public API, and may change without notice**.
They are used by SciMLSensitivity adjoint, callback, shadowing, and optimization
integrations. Application code should select a documented sensitivity algorithm
through `solve` or use the documented problem wrappers rather than calling these
helpers directly.

## Adjoint and Optimization Internals

```@docs
SciMLSensitivity.adjointdiffcache
SciMLSensitivity.OptimizationAdjointProblem
SciMLSensitivity.build_adjoint_jac
SciMLSensitivity.enzyme_rhs
SciMLSensitivity._init_originator_gradient
```

## Callback Tracking

```@docs
SciMLSensitivity.track_callbacks
SciMLSensitivity.setup_reverse_callbacks
SciMLSensitivity._has_effective_callback
```

## Derivative Wrappers and VJP Traits

```@docs
SciMLSensitivity.jacobianvec!
SciMLSensitivity.supports_callback_vjp
SciMLSensitivity.supports_structured_vjp
SciMLSensitivity.sensealg_autodiff_as_bool
SciMLSensitivity._get_sensitivity_vjp_verbose
```

## Parameter and Cotangent Helpers

```@docs
SciMLSensitivity.recursive_copyto!
SciMLSensitivity.recursive_add!
SciMLSensitivity.recursive_sub!
SciMLSensitivity.recursive_neg!
SciMLSensitivity.recursive_adjoint
SciMLSensitivity.allocate_zeros
SciMLSensitivity.allocate_vjp
SciMLSensitivity.mutable_zeros
SciMLSensitivity.tunable_cotangent
SciMLSensitivity.solution_parameter_cotangent
```

## Transformed Functions and Shadowing Regularizers

```@docs
SciMLSensitivity.TransformedFunction
SciMLSensitivity.TimeDilation
```
