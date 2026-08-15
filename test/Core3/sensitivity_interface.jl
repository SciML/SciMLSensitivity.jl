using SciMLSensitivity
using SciMLBase: ODEProblem
using Test

struct GenericSensitivitySolution
    prob::ODEProblem
end

struct GenericSensitivityFunction <: SensitivityFunction
    sol::GenericSensitivitySolution
end

function (sense::GenericSensitivityFunction)(du, u, p, t)
    du .= -u
    return nothing
end

function f!(du, u, p, t)
    du .= -u
    return nothing
end

prob = ODEProblem(f!, [1.0], (0.0, 1.0))
sense = GenericSensitivityFunction(GenericSensitivitySolution(prob))

@testset "SensitivityFunction developer interface" begin
    du = similar(prob.u0)
    sense(du, prob.u0, prob.p, first(prob.tspan))

    @test sense isa SensitivityFunction
    @test du == [-1.0]
    @test SciMLSensitivity.getprob(sense) === prob
    @test SciMLSensitivity.inplace_sensitivity(sense)
end
