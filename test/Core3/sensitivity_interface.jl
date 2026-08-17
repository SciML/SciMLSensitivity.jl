using SciMLSensitivity
using SciMLBase: ODEProblem, isinplace
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

struct GenericOverriddenSensitivityFunction{P, F} <: SensitivityFunction
    prob::P
    f::F
end

SciMLSensitivity.getprob(sense::GenericOverriddenSensitivityFunction) = sense.prob
SciMLSensitivity.inplace_sensitivity(::GenericOverriddenSensitivityFunction) = true

function (sense::GenericOverriddenSensitivityFunction)(du, u, p, t)
    du .= -u
    return nothing
end

struct GenericNoiseSensitivityFunction{S, F} <: SensitivityFunction
    sol::S
    f::F
end

function (sense::GenericNoiseSensitivityFunction)(du, u, p, t, W)
    du .= -u .+ W
    return nothing
end

prob = ODEProblem(fiip, [1.0], (0.0, 1.0))
proboop = ODEProblem(foop, [1.0], (0.0, 1.0))
sense = GenericSensitivityFunction(GenericSensitivitySolution(prob), fiip)
custom_sense = GenericProblemSensitivityFunction(prob, fiip)
oop_sense = GenericOutOfPlaceSensitivityFunction(GenericSensitivitySolution(proboop), foop)
overridden_sense = GenericOverriddenSensitivityFunction(proboop, foop)
noise_sense = GenericNoiseSensitivityFunction(GenericSensitivitySolution(prob), fiip)

@testset "SensitivityFunction developer interface" begin
    du = similar(prob.u0)
    sense(du, prob.u0, prob.p, first(prob.tspan))
    custom_sense(du, prob.u0, prob.p, first(prob.tspan))

    generated_prob = ODEProblem(sense, prob.u0, prob.tspan, prob.p)
    generated_prob.f(du, prob.u0, prob.p, first(prob.tspan))

    @test sense isa SensitivityFunction
    @test du == [-1.0]
    @test isinplace(generated_prob.f)
    @test SciMLSensitivity.getprob(sense) === prob
    @test SciMLSensitivity.inplace_sensitivity(sense)
    @test SciMLSensitivity.getprob(custom_sense) === prob
    @test SciMLSensitivity.inplace_sensitivity(custom_sense)
    @test oop_sense(prob.u0, proboop.p, first(proboop.tspan)) == [-1.0]
    @test SciMLSensitivity.getprob(oop_sense) === proboop
    @test !SciMLSensitivity.inplace_sensitivity(oop_sense)

    overridden_du = similar(proboop.u0)
    overridden_sense(overridden_du, proboop.u0, proboop.p, first(proboop.tspan))
    @test overridden_du == [-1.0]
    @test SciMLSensitivity.getprob(overridden_sense) === proboop
    @test SciMLSensitivity.inplace_sensitivity(overridden_sense)

    noise_du = similar(prob.u0)
    noise_sense(noise_du, prob.u0, prob.p, first(prob.tspan), [2.0])
    @test noise_du == [1.0]

    @static if isdefined(Base, :ispublic)
        @test Base.ispublic(SciMLSensitivity, :SensitivityFunction)
        @test Base.ispublic(SciMLSensitivity, :getprob)
        @test Base.ispublic(SciMLSensitivity, :inplace_sensitivity)
    end
end
