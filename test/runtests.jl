using DiffEqSensitivity
using Base.Test

@testset "Local Sensitivity" begin include("local.jl") end
@testset "Adjoint Sensitivity" begin include("adjoint.jl") end
