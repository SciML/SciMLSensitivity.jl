using SciMLSensitivity, SciMLLogging, Test, Logging

@testset "_get_sensitivity_vjp_verbose" begin
    @test SciMLSensitivity._get_sensitivity_vjp_verbose(true) === true
    @test SciMLSensitivity._get_sensitivity_vjp_verbose(false) === false

    # Presets are not Bool and don't have :sensitivity_vjp_choice; backward-compat default is true.
    @test SciMLSensitivity._get_sensitivity_vjp_verbose(SciMLLogging.Standard()) === true
    @test SciMLSensitivity._get_sensitivity_vjp_verbose(SciMLLogging.None()) === true

    # A struct exposing :sensitivity_vjp_choice is honored.
    struct _DummyVerb
        sensitivity_vjp_choice::Any
    end
    @test SciMLSensitivity._get_sensitivity_vjp_verbose(_DummyVerb(SciMLLogging.Silent())) === false
    @test SciMLSensitivity._get_sensitivity_vjp_verbose(_DummyVerb(SciMLLogging.WarnLevel())) === true
end

@testset "@SciMLMessage routing for sensitivity_vjp_choice" begin
    # Bool true → emits at WarnLevel through SciMLLogging
    io = IOBuffer()
    with_logger(ConsoleLogger(io, Logging.Warn)) do
        @SciMLMessage("vjp warn", true, :sensitivity_vjp_choice)
    end
    @test occursin("vjp warn", String(take!(io)))

    # Bool false → silent
    io2 = IOBuffer()
    with_logger(ConsoleLogger(io2, Logging.Warn)) do
        @SciMLMessage("vjp silent", false, :sensitivity_vjp_choice)
    end
    @test isempty(String(take!(io2)))
end
