module SciMLSensitivity

using ADTypes: ADTypes, AutoEnzyme, AutoFiniteDiff, AutoForwardDiff,
    AutoMooncake, AutoReverseDiff, AutoTracker, AutoZygote


using Accessors: @reset
using Adapt: Adapt, adapt
using ArrayInterface: ArrayInterface
using DiffEqBase: DiffEqBase, SensitivityADPassThrough
using DiffEqCallbacks: DiffEqCallbacks, IntegrandValuesSum, IntegratingSumCallback,
    IntegratingGKSumCallback, PresetTimeCallback
using DiffEqNoiseProcess: DiffEqNoiseProcess
using FastBroadcast: @..
using Functors: Functors, fmap
using FunctionProperties: hasbranching
using FunctionWrappersWrappers: FunctionWrappersWrappers
using GPUArraysCore: GPUArraysCore
using IntervalSets: IntervalSets, var".."
using LinearSolve: LinearSolve
using PreallocationTools: PreallocationTools, get_tmp, DiffCache,
    LazyBufferCache
using RandomNumbers: Xorshifts
using RecursiveArrayTools: RecursiveArrayTools, AbstractDiffEqArray,
    AbstractVectorOfArray, ArrayPartition, DiffEqArray,
    VectorOfArray
using SciMLJacobianOperators: VecJacOperator, StatefulJacobianOperator
using SciMLLogging: SciMLLogging, verbosity_to_bool, @SciMLMessage
using SciMLStructures: SciMLStructures, canonicalize, Tunable, isscimlstructure
using SymbolicIndexingInterface: SymbolicIndexingInterface, current_time, getu,
    parameter_values, state_values
using QuadGK: quadgk
using SciMLBase: SciMLBase, AbstractOverloadingSensitivityAlgorithm,
    AbstractForwardSensitivityAlgorithm, AbstractAdjointSensitivityAlgorithm,
    AbstractSecondOrderSensitivityAlgorithm,
    AbstractShadowingSensitivityAlgorithm,
    AbstractNonlinearProblem, AbstractSensitivityAlgorithm,
    AbstractDiffEqFunction, AbstractODEFunction, unwrapped_f, CallbackSet,
    ContinuousCallback, AbstractTimeseriesSolution, NonlinearFunction, NonlinearProblem,
    DiscreteCallback, LinearProblem, ODEFunction, ODEProblem, DAEFunction, DAEProblem,
    RODEFunction, RODEProblem, ReturnCode, SDEFunction,
    SDEProblem, VectorContinuousCallback,
    get_tmp_cache, isinplace, reinit!, remake,
    solve, derivative_discontinuity!, LinearAliasSpecifier, OverrideInit, AbstractOptimizationProblem
using SciMLOperators: has_adjoint

using OrdinaryDiffEqCore: OrdinaryDiffEqCore, BrownFullBasicInit, DefaultInit,
    default_nlsolve, has_autodiff

# AD Backends
using ChainRulesCore: unthunk, @thunk, NoTangent, @not_implemented, Tangent, ZeroTangent,
    AbstractThunk, AbstractTangent
using Enzyme: Enzyme
using FiniteDiff: FiniteDiff
using ForwardDiff: ForwardDiff
using Tracker: Tracker, TrackedArray
using ReverseDiff: ReverseDiff
using Zygote: Zygote
using ConstructionBase: ConstructionBase, setproperties

# Std Libs
using LinearAlgebra: LinearAlgebra, Diagonal, I, UniformScaling, adjoint, axpy!,
    convert, copyto!, dot, issuccess, ldiv!, lmul!, lu, lu!, mul!,
    norm, normalize!, qr, transpose, transpose!
using Markdown: Markdown, @doc_str
using Random: Random, rand!
using SparseArrays: SparseArrays
using Statistics: Statistics, mean

"""
    SensitivityFunction

Developer interface for the right-hand-side functions used by
SciMLSensitivity adjoint and callback sensitivity problems.

`SensitivityFunction` declares no fields. Concrete subtypes normally retain the
original differential-equation function in `f`, a forward solution in `sol`,
the originating problem in `prob` or `sol.prob`, and the algorithm and
derivative-cache state needed by the sensitivity calculation. The constructors
in SciMLSensitivity create these objects; users should select a `sensealg`
through `solve` or use a documented sensitivity problem wrapper instead.

# Interface

The subtype is passed as the differential-equation function of a generated
problem. It must implement the call convention selected by
[`inplace_sensitivity`](@ref):

  - in-place ODE or SDE drift: `(S)(du, u, p, t) -> nothing`;
  - out-of-place ODE or SDE drift: `(S)(u, p, t) -> du`;
  - reverse RODE/SDE noise path: `(S)(du, u, p, t, W) -> nothing` when the
    generated problem supplies `W`;
  - fully implicit DAE residual: `(S)(res, du, u, p, t) -> nothing` for a
    subtype used as a DAE residual.

The call must fill or return the complete sensitivity state derivative required
by the generated problem, including adjoint, parameter-gradient, quadrature,
or residual components. The original model function is available as `S.f` to
the derivative-wrapper machinery and must use the same in-place convention as
the sensitivity call unless the subtype supplies the corresponding wrapper
methods.

# Fields

Concrete subtypes commonly define the following fields:

  - `f`: Original differential-equation function used for state and parameter
    derivatives.
  - `prob` or `sol.prob`: Originating SciML problem, exposed through
    [`getprob`](@ref).
  - `sensealg`: Sensitivity algorithm controlling derivative evaluation.
  - `diffcache`: Derivative workspaces owned by the sensitivity algorithm.
  - Additional state such as the forward solution, checkpoint solution, cost
    derivatives, and callback data required by the subtype.

The concrete field types and the remaining field set are implementation
details. They should be kept concrete so that the generated sensitivity
problem remains type stable.

# Extension Rules

Use the package constructors when possible. An extension that defines a new
subtype must provide a callable method with the appropriate signature, retain
the originating problem, and implement
`SciMLSensitivity.getprob(S::MySensitivityFunction)` when the problem is not
available as `S.sol.prob`. The default `getprob` method supports the built-in
backsolve layout and the `sol.prob` layout used by the other built-in
subtypes.

The default [`inplace_sensitivity`](@ref) method delegates to
`SciMLBase.isinplace(getprob(S))`. Override it only when the callable method
deliberately uses a different convention. A subtype that supports reverse
noise must also implement the five-argument call; a subtype that supports
fully implicit DAE residuals must follow the DAE residual convention above.
Do not rely on the fields of another concrete sensitivity-function subtype.

# Examples

The following minimal subtype demonstrates the generic in-place contract. Real
implementations should use the package constructors so that derivative caches,
callbacks, and solver-specific state are initialized consistently.

```julia
struct ExampleSensitivityFunction{S, F} <: SensitivityFunction
    sol::S
    f::F
end

function (S::ExampleSensitivityFunction)(du, u, p, t)
    du .= -u
    return nothing
end
```

This is developer API for SciMLSensitivity and SciML solver integrations. It is
versioned and tested for extension authors, but it is not a user-facing
modeling interface. Users should not subtype it merely to differentiate an
ODE; use the documented sensitivity algorithms and problem wrappers instead.
"""
abstract type SensitivityFunction end

"""
    TransformedFunction

Abstract supertype for transformed differential-equation functions used inside
SciMLSensitivity.

Concrete subtypes adapt a user-supplied model into the drift or sensitivity
form required by a sensitivity algorithm. This is developer API for
SciMLSensitivity internals and extensions; it is not a user-facing modeling
interface.
"""
abstract type TransformedFunction end

"""
    ODEAdjointProblem(sol, sensealg, alg, t=nothing,
        dgdu_discrete=nothing, dgdp_discrete=nothing,
        dgdu_continuous=nothing, dgdp_continuous=nothing, g=nothing; kwargs...)

Construct the reverse-time `ODEProblem` used by continuous adjoint sensitivity
algorithms.

## Arguments

  - `sol`: forward solution whose problem, trajectory, and parameters define the
    adjoint system.
  - `sensealg`: adjoint sensitivity algorithm, such as `BacksolveAdjoint`,
    `InterpolatingAdjoint`, `QuadratureAdjoint`, or `GaussAdjoint`.
  - `alg`: differential-equation solver algorithm used for the adjoint solve.
  - `t`: saved time points for discrete costs. Use `nothing` for a continuous
    cost.
  - `dgdu_discrete`, `dgdp_discrete`: derivatives of a discrete cost at the
    saved points.
  - `dgdu_continuous`, `dgdp_continuous`: derivatives of a continuous cost
    integrand.
  - `g`: optional scalar cost function used when derivative callbacks are not
    supplied.

## Returns

An `ODEProblem` whose state contains adjoint variables and parameter-gradient
accumulators. Some internal methods can also return callback bookkeeping when
requested by SciMLSensitivity internals.
"""
function ODEAdjointProblem end

"""
    SDEAdjointProblem(sol, sensealg, alg, t=nothing,
        dgdu_discrete=nothing, dgdp_discrete=nothing,
        dgdu_continuous=nothing, dgdp_continuous=nothing, g=nothing; kwargs...)

Construct the reverse-time `SDEProblem` used by continuous adjoint sensitivity
algorithms for stochastic differential equations.

The arguments mirror `ODEAdjointProblem`. For Ito problems the drift is
internally transformed to the adjoint-compatible form; Stratonovich problems use
the original drift interpretation.
"""
function SDEAdjointProblem end

"""
    RODEAdjointProblem(sol, sensealg, alg, t=nothing,
        dgdu_discrete=nothing, dgdp_discrete=nothing,
        dgdu_continuous=nothing, dgdp_continuous=nothing, g=nothing; kwargs...)

Construct the reverse-time `RODEProblem` used by continuous adjoint sensitivity
algorithms for random ordinary differential equations.

The arguments mirror `ODEAdjointProblem`. The returned problem reuses the
forward solution noise process in reverse order and augments the state with
adjoint and parameter-gradient variables.
"""
function RODEAdjointProblem end

include("utils.jl")
include("parameters_handling.jl")
include("sensitivity_algorithms.jl")
include("derivative_wrappers.jl")
include("sensitivity_interface.jl")
include("forward_sensitivity.jl")
include("adjoint_common.jl")
include("lss.jl")
include("nilss.jl")
include("nilsas.jl")
include("backsolve_adjoint.jl")
include("interpolating_adjoint.jl")
include("quadrature_adjoint.jl")
include("gauss_adjoint.jl")
include("dae_adjoint.jl")
include("callback_tracking.jl")
include("concrete_solve.jl")
include("second_order.jl")
include("steadystate_adjoint.jl")
include("sde_tools.jl")
include("enzyme_rules.jl")

export extract_local_sensitivities

export ODEForwardSensitivityFunction, ODEForwardSensitivityProblem, SensitivityFunction,
    ODEAdjointProblem, AdjointSensitivityIntegrand,
    SDEAdjointProblem, RODEAdjointProblem, DAEAdjointProblem, SensitivityAlg,
    adjoint_sensitivities,
    ForwardLSSProblem, AdjointLSSProblem,
    NILSSProblem, NILSASProblem,
    shadow_forward, shadow_adjoint

export BacksolveAdjoint, QuadratureAdjoint, GaussAdjoint, GaussKronrodAdjoint,
    SundialsAdjoint,
    InterpolatingAdjoint,
    TrackerAdjoint, ZygoteAdjoint, ReverseDiffAdjoint, MooncakeAdjoint,
    EnzymeAdjoint, ForwardSensitivity, ForwardDiffSensitivity,
    ForwardDiffOverAdjoint,
    SteadyStateAdjoint, UnconstrainedOptimizationAdjoint,
    ForwardLSS, AdjointLSS, NILSS, NILSAS

export second_order_sensitivities, second_order_sensitivity_product

export TrackerVJP, ZygoteVJP, EnzymeVJP, ReverseDiffVJP, ReactantVJP,
    ReactantVJPConfig, ReactantDualTag

export supports_functor_params

export StochasticTransformedFunction

end # module
