module DiffEqSensitivity

using DiffEqBase, ForwardDiff, Tracker, FiniteDiff, Statistics
using DiffEqCallbacks, QuadGK, RecursiveArrayTools, LinearAlgebra
using Adapt
using Parameters: @unpack
using Requires
using StochasticDiffEq
using SharedArrays
import DiffEqNoiseProcess
import ZygoteRules, Zygote, ReverseDiff
using Reexport
@reexport using GlobalSensitivity
abstract type SensitivityFunction end

include("require.jl")
include("sensitivity_algorithms.jl")
include("derivative_wrappers.jl")
include("sensitivity_interface.jl")
include("forward_sensitivity.jl")
include("adjoint_common.jl")
include("backsolve_adjoint.jl")
include("interpolating_adjoint.jl")
include("quadrature_adjoint.jl")
include("concrete_solve.jl")
include("second_order.jl")
include("steadystate_adjoint.jl")
include("sde_tools.jl")

export extract_local_sensitivities

export ODEForwardSensitivityFunction, ODEForwardSensitivityProblem, SensitivityFunction,
       ODEAdjointSensitivityProblem, ODEAdjointProblem, AdjointSensitivityIntegrand,
       SDEAdjointProblem, SensitivityAlg,
       adjoint_sensitivities, adjoint_sensitivities_u0 

export BacksolveAdjoint, QuadratureAdjoint, InterpolatingAdjoint,
       TrackerAdjoint, ZygoteAdjoint, ReverseDiffAdjoint,
       ForwardSensitivity, ForwardDiffSensitivity,
       ForwardDiffOverAdjoint,
       SteadyStateAdjoint

export second_order_sensitivities, second_order_sensitivity_product

export TrackerVJP, ZygoteVJP, ReverseDiffVJP

export StochasticTransformedFunction

function gsa(f, method::GlobalSensitivity.GSAMethod, args...; kwargs...)
    @warn "Global Sensitivity Analysis has been moved to separate package GlobalSensitivity.jl (https://github.com/SciML/GlobalSensitivity.jl)"
    GlobalSensitivity.gsa(f, method, args...; kwargs...)
end
end # module
