using SciMLSensitivity
using SciMLBase: ODEProblem
using Test

function fiip(du, u, p, t)
    du .= -u
    return nothing
end

function foop(u, p, t)
    return -u
end

struct GenericSensitivitySolution{P}
    prob::P
end

struct GenericSensitivityFunction{S, F} <: SensitivityFunction
    sol::S
    f::F
end

function (sense::GenericSensitivityFunction)(du, u, p, t)
    du .= -u
    return nothing
end

struct GenericProblemSensitivityFunction{P, F} <: SensitivityFunction
    prob::P
    f::F
end

SciMLSensitivity.getprob(sense::GenericProblemSensitivityFunction) = sense.prob

function (sense::GenericProblemSensitivityFunction)(du, u, p, t)
    du .= -u
    return nothing
end

struct GenericOutOfPlaceSensitivityFunction{S, F} <: SensitivityFunction
    sol::S
    f::F
end

function (sense::GenericOutOfPlaceSensitivityFunction)(u, p, t)
    return -u
end

prob = ODEProblem(fiip, [1.0], (0.0, 1.0))
proboop = ODEProblem(foop, [1.0], (0.0, 1.0))
sense = GenericSensitivityFunction(GenericSensitivitySolution(prob), fiip)
custom_sense = GenericProblemSensitivityFunction(prob, fiip)
oop_sense = GenericOutOfPlaceSensitivityFunction(GenericSensitivitySolution(proboop), foop)

@testset "SensitivityFunction developer interface" begin
    du = similar(prob.u0)
    sense(du, prob.u0, prob.p, first(prob.tspan))
    custom_sense(du, prob.u0, prob.p, first(prob.tspan))

    @test sense isa SensitivityFunction
    @test du == [-1.0]
    @test SciMLSensitivity.getprob(sense) === prob
    @test SciMLSensitivity.inplace_sensitivity(sense)
    @test SciMLSensitivity.getprob(custom_sense) === prob
    @test SciMLSensitivity.inplace_sensitivity(custom_sense)
    @test oop_sense(prob.u0, proboop.p, first(proboop.tspan)) == [-1.0]
    @test SciMLSensitivity.getprob(oop_sense) === proboop
    @test !SciMLSensitivity.inplace_sensitivity(oop_sense)
end
