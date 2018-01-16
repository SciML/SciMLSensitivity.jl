__precompile__()

module DiffEqSensitivity

using DiffEqBase, Compat, ForwardDiff, DiffEqDiffTools

abstract type SensitivityFunction end

include("derivative_wrappers.jl")
include("local_sensitivity.jl")

export extract_local_sensitivities

export ODELocalSensitvityFunction, ODELocalSensitivityProblem, SensitivityFunction
end # module
