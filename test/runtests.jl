using DiffEqSensitivity
using Base.Test

@testset "Local Sensitivity" begin include("local.jl") end
@testset "Adjoint Sensitivity" begin include("adjoint.jl") end
@testset "Morris Method" begin include("morris_method.jl") end
@testset "Sobol Method" begin include("sobol_method.jl") end