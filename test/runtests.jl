using DiffEqSensitivity, SafeTestsets
using Test

const GROUP = get(ENV, "GROUP", "All")
const is_APPVEYOR = Sys.iswindows() && haskey(ENV,"APPVEYOR")
const is_TRAVIS = haskey(ENV,"TRAVIS")

@time begin
if GROUP == "All" || GROUP == "Core" || GROUP == "Downstream"
    @time @safetestset "Forward Sensitivity" begin include("forward.jl") end
    @time @safetestset "Adjoint Sensitivity" begin include("adjoint.jl") end
    @time @safetestset "Morris Method" begin include("morris_method.jl") end
    @time @safetestset "Sobol Method" begin include("sobol_method.jl") end
    @time @safetestset "DGSM Method" begin include("DGSM.jl") end
    @time @safetestset "eFAST Method" begin include("eFAST_method.jl") end
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
