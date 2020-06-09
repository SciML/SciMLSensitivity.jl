using DiffEqSensitivity, SafeTestsets
using Test

const GROUP = get(ENV, "GROUP", "All")
const is_APPVEYOR = Sys.iswindows() && haskey(ENV,"APPVEYOR")
const is_TRAVIS = haskey(ENV,"TRAVIS")

@time begin
if GROUP == "All" || GROUP == "Core1" || GROUP == "Downstream"
    @time @safetestset "Forward Sensitivity" begin include("local_sensitivity/forward.jl") end
    @time @safetestset "Adjoint Sensitivity" begin include("local_sensitivity/adjoint.jl") end
    @time @safetestset "Second Order Sensitivity" begin include("local_sensitivity/second_order.jl") end
    @time @safetestset "Concrete Solve Derivatives" begin include("local_sensitivity/concrete_solve_derivatives.jl") end
    @time @safetestset "Branching Derivatives" begin include("local_sensitivity/branching_derivatives.jl") end
end

if GROUP == "All" || GROUP == "Core2"
    @time @safetestset "Steady State Adjoint" begin include("local_sensitivity/steady_state.jl") end
    @time @safetestset "SDE Adjoint" begin include("local_sensitivity/sde.jl") end
    @time @safetestset "Concrete Solve Derivatives of Second Order ODEs" begin include("local_sensitivity/second_order_odes.jl") end
end

if GROUP == "All" || GROUP == "Core3"
    @time @safetestset "Stiff Adjoints" begin include("local_sensitivity/stiff_adjoints.jl") end
end

if GROUP == "All" || GROUP == "GSA"
    @time @safetestset "Morris Method" begin include("global_sensitivity/morris_method.jl") end
    @time @safetestset "Sobol Method" begin include("global_sensitivity/sobol_method.jl") end
    @time @safetestset "DGSM Method" begin include("global_sensitivity/DGSM.jl") end
    @time @safetestset "eFAST Method" begin include("global_sensitivity/eFAST_method.jl") end
    @time @safetestset "RegressionGSA Method" begin include("global_sensitivity/regression_sensitivity.jl") end
end

if GROUP == "DiffEqFlux"
    using Pkg
    if is_TRAVIS
      using Pkg
      Pkg.add("DiffEqFlux")
    end
    @time Pkg.test("DiffEqFlux")
end
end
