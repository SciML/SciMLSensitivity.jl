# This file will be removed, and put in an extension in DiffEqBase before merging.
Mooncake.@from_rrule(
    Mooncake.MinimalCtx,
    Tuple{
        typeof(DiffEqBase.solve_up),
        DiffEqBase.AbstractDEProblem,
        Union{Nothing, DiffEqBase.AbstractSensitivityAlgorithm},
        Any,
        Any,
        Any,
    },
    true,
)

Mooncake.@zero_adjoint Mooncake.MinimalCtx Tuple{typeof(DiffEqBase.numargs), Any}