module SciMLSensitivity

using DiffEqBase, ForwardDiff, Tracker, FiniteDiff, Statistics
using DiffEqCallbacks, QuadGK, RecursiveArrayTools, LinearAlgebra
using Adapt
using LinearSolve
using Parameters: @unpack
using StochasticDiffEq
import DiffEqNoiseProcess
import RandomNumbers: Xorshifts
using Random
import Zygote, ReverseDiff
import ArrayInterface
import Enzyme
import GPUArraysCore
using ADTypes
using SparseDiffTools
using SciMLOperators
using Functors
import TruncatedStacktraces

import PreallocationTools: dualcache, get_tmp, DiffCache, LazyBufferCache,
                           FixedSizeDiffCache
import FunctionWrappersWrappers
using EllipsisNotation
using FunctionProperties: hasbranching

using SymbolicIndexingInterface
using SciMLStructures: canonicalize, Tunable, isscimlstructure

using Markdown

using Reexport
import ChainRulesCore: unthunk, @thunk, NoTangent, @not_implemented
abstract type SensitivityFunction end
abstract type TransformedFunction end

import SciMLBase: unwrapped_f, _unwrap_val

import SciMLBase: AbstractOverloadingSensitivityAlgorithm, AbstractSensitivityAlgorithm,
                  AbstractForwardSensitivityAlgorithm, AbstractAdjointSensitivityAlgorithm,
                  AbstractSecondOrderSensitivityAlgorithm,
                  AbstractShadowingSensitivityAlgorithm,
                  AbstractTimeseriesSolution

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

export extract_local_sensitivities

export ODEForwardSensitivityFunction, ODEForwardSensitivityProblem, SensitivityFunction,
       ODEAdjointProblem, AdjointSensitivityIntegrand,
       SDEAdjointProblem, RODEAdjointProblem, SensitivityAlg,
       adjoint_sensitivities,
       ForwardLSSProblem, AdjointLSSProblem,
       NILSSProblem, NILSASProblem,
       shadow_forward, shadow_adjoint

export BacksolveAdjoint, QuadratureAdjoint, GaussAdjoint, InterpolatingAdjoint,
       TrackerAdjoint, ZygoteAdjoint, ReverseDiffAdjoint,
       ForwardSensitivity, ForwardDiffSensitivity,
       ForwardDiffOverAdjoint,
       SteadyStateAdjoint,
       ForwardLSS, AdjointLSS, NILSS, NILSAS

export second_order_sensitivities, second_order_sensitivity_product

export TrackerVJP, ZygoteVJP, EnzymeVJP, ReverseDiffVJP

export StochasticTransformedFunction

end # module
