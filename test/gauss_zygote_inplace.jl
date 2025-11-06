using SciMLSensitivity, DifferentialEquations, Zygote
using Test

# Test for issue #1282: GaussAdjoint with ZygoteVJP should handle in-place ODE functions
@testset "GaussAdjoint with ZygoteVJP and in-place ODE" begin
    function fiip(du, u, p, t)
        du[1] = p[1] * u[1] - p[2] * u[1] * u[2]
        du[2] = -p[3] * u[2] + p[4] * u[1] * u[2]
        return nothing
    end

    p = [1.5, 1.0, 3.0, 1.0]
    u0 = [1.0; 1.0]
    prob = ODEProblem{true, SciMLBase.FullSpecialize}(
        ODEFunction{true, SciMLBase.FullSpecialize}(fiip),
        u0, (0.0, 10.0), p
    )

    # Test that basic solve works
    sol = solve(prob, KenCarp4(), sensealg = GaussAdjoint())
    @test sol.retcode == ReturnCode.Success

    # Test that gradient computation with ZygoteVJP works
    loss(u0,
        p) = sum(solve(
        prob, KenCarp4(), u0 = u0, p = p, saveat = 0.1,
        sensealg = GaussAdjoint(autojacvec = ZygoteVJP())
    ))

    # This should not throw MethodError anymore
    du0, dp = Zygote.gradient(loss, u0, p)

    @test du0 !== nothing
    @test dp !== nothing
    @test length(du0) == 2
    @test length(dp) == 4

    # Test with explicit ZygoteVJP specification
    (dp2,) = Zygote.gradient(p) do p
        sum(solve(prob, KenCarp4(), p = p, saveat = 0.1,
            sensealg = GaussAdjoint(autojacvec = ZygoteVJP())))
    end

    @test dp2 !== nothing
    @test length(dp2) == 4
end

# Test out-of-place still works
@testset "GaussAdjoint with ZygoteVJP and out-of-place ODE" begin
    function foop(u, p, t)
        dx = p[1] * u[1] - p[2] * u[1] * u[2]
        dy = -p[3] * u[2] + p[4] * u[1] * u[2]
        [dx, dy]
    end

    p = [1.5, 1.0, 3.0, 1.0]
    u0 = [1.0; 1.0]
    prob = ODEProblem(foop, u0, (0.0, 10.0), p)

    # Test that gradient computation with ZygoteVJP works for out-of-place
    loss(u0,
        p) = sum(solve(
        prob, Tsit5(), u0 = u0, p = p, saveat = 0.1,
        sensealg = GaussAdjoint(autojacvec = ZygoteVJP())
    ))

    du0, dp = Zygote.gradient(loss, u0, p)

    @test du0 !== nothing
    @test dp !== nothing
    @test length(du0) == 2
    @test length(dp) == 4
end
