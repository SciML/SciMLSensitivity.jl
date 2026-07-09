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
    SDEProblem, VectorContinuousCallback, deleteat!,
    get_tmp_cache, has_adjoint, isinplace, reinit!, remake,
    solve, derivative_discontinuity!, LinearAliasSpecifier, OverrideInit, AbstractOptimizationProblem

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

Abstract supertype for the internal right-hand-side functions used by
SciMLSensitivity adjoint and callback sensitivity problems.

Concrete subtypes carry the forward solution, cost-function derivatives,
algorithm choice, and derivative caches needed by the generated sensitivity
`ODEProblem`, `SDEProblem`, or `RODEProblem`. This is developer API for
SciMLSensitivity and SciML solver integrations; users normally select a
`sensealg` through `solve` or construct one of the documented sensitivity
problem wrappers instead.
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
