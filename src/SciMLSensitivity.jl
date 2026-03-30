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
using PrecompileTools: @compile_workload
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
SciMLSensitivity adjoint and callback sensitivity problems. This is a
versioned extension interface for sensitivity and solver packages, not a
user-facing modeling interface.

`SensitivityFunction` has no fields or public constructor. The package
constructors create concrete subtypes with the forward solution, originating
problem, sensitivity algorithm, and derivative workspaces needed by a specific
algorithm. Application code should select a `sensealg` through `solve` or use a
documented sensitivity problem wrapper instead.

# Fields

  - `SensitivityFunction`: no fields; this is an abstract interface only.
  - Concrete subtypes choose their own fields. The originating problem must be
    available through `prob`, `sol.prob`, or a `getprob` method. Fields such as
    `f`, `sensealg`, and `diffcache` are required only by derivative-wrapper
    integrations that access them.

# Arguments

The required positional arguments are determined by the generated problem and
the callable convention below. In-place methods receive a destination first;
out-of-place methods return the complete derivative. The arguments `u`, `p`,
and `t` are the generated state, parameters, and time. The optional `W` is the
noise process supplied by a RODE or SDE problem.

# Keywords

No keyword arguments are required by the `SensitivityFunction` interface. A
solver or extension may pass additional keywords to a wrapper it owns, but a
new sensitivity function must support the positional convention selected by
its generated problem.

# Callable Interface

The subtype is passed as the differential-equation function of a generated
problem. A sensitivity right-hand-side subtype must implement one of the
following call conventions:

  - in-place ODE or SDE drift: `(S)(du, u, p, t) -> nothing`;
  - out-of-place ODE or SDE drift: `(S)(u, p, t) -> du`;
  - in-place RODE or SDE noise path: `(S)(du, u, p, t, W) -> nothing`;
  - augmented DAE sensitivity state: `(S)(dz, z, p, t) -> nothing`.

`DAEAdjointResidual` is a built-in wrapper for a fully implicit DAE residual
with the convention `(R)(res, dz, z, p, t) -> nothing`; a custom DAE residual
wrapper must preserve that convention.

The callable must fill or return the complete sensitivity state derivative
required by the generated problem, including adjoint, parameter-gradient,
quadrature, and residual components when present. The five-argument form is
required only when the generated problem supplies a noise process.

# Required Properties

A concrete subtype must retain the originating SciML problem. Store it in
`prob` or `sol.prob` when the default [`getprob`](@ref) method applies; otherwise
implement `SciMLSensitivity.getprob(S::MySensitivityFunction)` and return the
exact problem used to create the sensitivity function.

The default [`inplace_sensitivity`](@ref) method obtains the calling convention
from `SciMLBase.isinplace(getprob(S))`. Override it when the callable method
intentionally uses a different convention, and ensure the reported value and
call signature agree. For example, a subtype whose originating problem is
out-of-place but whose sensitivity callable mutates `du` must return `true`.

For use with SciMLSensitivity's derivative-wrapper machinery, also expose:

  - `f`: the original differential-equation function used for state and
    parameter derivatives;
  - `sensealg`: the sensitivity algorithm, including the selected derivative
    backend;
  - `diffcache`: the backend-specific derivative workspaces and caches;
  - any additional state required by the selected algorithm, such as `sol`,
    checkpoint data, cost derivatives, or callback data.

The `f`, `sensealg`, and `diffcache` properties are requirements for using the
derivative wrappers, not for the minimal callable/trait contract alone. Keep
their types concrete and do not depend on the fields of another concrete
sensitivity-function subtype.

# Extension Rules

Use the package constructors when possible. Add methods to the generic
`getprob` and `inplace_sensitivity` functions rather than relying on a concrete
SciMLSensitivity subtype's fields. A new subtype may use any storage layout if
it satisfies the callable convention, retains the originating problem, and
provides the derivative-wrapper properties when it calls those wrappers.

# Examples

The following minimal subtype demonstrates the generic in-place contract and a
custom problem-storage method. Real implementations should use the package
constructors so that derivative caches, callbacks, and solver-specific state
are initialized consistently.

```julia
using SciMLBase: ODEProblem

struct ExampleSensitivityFunction{P, F} <: SensitivityFunction
    prob::P
    f::F
end

SciMLSensitivity.getprob(S::ExampleSensitivityFunction) = S.prob

function (S::ExampleSensitivityFunction)(du, u, p, t)
    du .= -u
    return nothing
end

prob = ODEProblem((du, u, p, t) -> du .= u, [1.0], (0.0, 1.0))
sense = ExampleSensitivityFunction(prob, prob.f)
du = similar(prob.u0)
sense(du, prob.u0, prob.p, first(prob.tspan))
```

This is developer API for SciMLSensitivity and SciML solver integrations. It is
versioned and tested for extension authors. Users should not subtype it merely
to differentiate an ODE; use the documented sensitivity algorithms and problem
wrappers instead.
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
include("optimization_adjoint.jl")
include("concrete_solve.jl")
include("second_order.jl")
include("steadystate_adjoint.jl")
include("sde_tools.jl")
include("enzyme_rules.jl")
include("precompilation.jl")

@static if VERSION >= v"1.11.0-DEV.469"
    eval(Expr(:public, :getprob, :inplace_sensitivity))
end

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
    SteadyStateAdjoint, UnconstrainedOptimizationAdjoint, OptimizationAdjoint,
    ForwardLSS, AdjointLSS, NILSS, NILSAS

export second_order_sensitivities, second_order_sensitivity_product

export TrackerVJP, ZygoteVJP, EnzymeVJP, ReverseDiffVJP, ReactantVJP,
    ReactantVJPConfig, ReactantDualTag

export supports_functor_params

export StochasticTransformedFunction

end # module
