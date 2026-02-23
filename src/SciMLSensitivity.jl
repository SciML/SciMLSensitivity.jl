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
using LinearSolve: LinearSolve
using PreallocationTools: PreallocationTools, dualcache, get_tmp, DiffCache,
    LazyBufferCache
using RandomNumbers: Xorshifts
using RecursiveArrayTools: RecursiveArrayTools, AbstractDiffEqArray,
    AbstractVectorOfArray, ArrayPartition, DiffEqArray,
    VectorOfArray
using SciMLJacobianOperators: VecJacOperator, StatefulJacobianOperator
using SciMLLogging: SciMLLogging, verbosity_to_bool
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
    ContinuousCallback, DESolution, NonlinearFunction, NonlinearProblem,
    DiscreteCallback, LinearProblem, ODEFunction, ODEProblem, DAEProblem,
    RODEFunction, RODEProblem, ReturnCode, SDEFunction,
    SDEProblem, VectorContinuousCallback, deleteat!,
    get_tmp_cache, has_adjoint, isinplace, reinit!, remake,
    solve, u_modified!, LinearAliasSpecifier, OverrideInit

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
using SciMLBase.ConstructionBase: setproperties

# Std Libs
using LinearAlgebra: LinearAlgebra, Diagonal, I, UniformScaling, adjoint, axpy!,
    convert, copyto!, dot, issuccess, ldiv!, lu, lu!, mul!,
    norm, normalize!, qr, transpose
using Markdown: Markdown, @doc_str
using Random: Random, rand!
using Statistics: Statistics, mean

abstract type SensitivityFunction end
abstract type TransformedFunction end

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
include("callback_tracking.jl")
include("concrete_solve.jl")
include("second_order.jl")
include("steadystate_adjoint.jl")
include("sde_tools.jl")
include("enzyme_rules.jl")

export extract_local_sensitivities

export ODEForwardSensitivityFunction, ODEForwardSensitivityProblem, SensitivityFunction,
    ODEAdjointProblem, AdjointSensitivityIntegrand,
    SDEAdjointProblem, RODEAdjointProblem, SensitivityAlg,
    adjoint_sensitivities,
    ForwardLSSProblem, AdjointLSSProblem,
    NILSSProblem, NILSASProblem,
    shadow_forward, shadow_adjoint

export BacksolveAdjoint, QuadratureAdjoint, GaussAdjoint, GaussKronrodAdjoint,
    InterpolatingAdjoint,
    TrackerAdjoint, ZygoteAdjoint, ReverseDiffAdjoint, MooncakeAdjoint,
    EnzymeAdjoint, ForwardSensitivity, ForwardDiffSensitivity,
    ForwardDiffOverAdjoint,
    SteadyStateAdjoint,
    ForwardLSS, AdjointLSS, NILSS, NILSAS

export second_order_sensitivities, second_order_sensitivity_product

export TrackerVJP, ZygoteVJP, EnzymeVJP, ReverseDiffVJP, ReactantVJP

export supports_functor_params

export StochasticTransformedFunction

end # module
