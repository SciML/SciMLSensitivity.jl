using OrdinaryDiffEq, Zygote
using SciMLSensitivity, Test, ForwardDiff

abstol = 1e-12
reltol = 1e-12
savingtimes = 0.5

function test_continuous_wrt_discrete_callback()
    # test the continuous callbacks wrt to the equivalent discrete callback
    function f(du, u, p, t)
        #Bouncing Ball
        du[1] = u[2]
        du[2] = -p[1]
    end

    # no saving in Callbacks; prescribed vafter and vbefore; loss on the endpoint

    tstop = 3.1943828249997
    vbefore = -31.30495168499705
    vafter = 25.04396134799764

    u0 = [50.0, 0.0]
    tspan = (0.0, 5.0)
    p = [9.8, 0.8]

    prob = ODEProblem(f, u0, tspan, p)

    function condition(u, t, integrator) # Event when event_f(u,t) == 0
        t - tstop
    end
    function affect!(integrator)
        integrator.u[2] += vafter - vbefore
    end
    cb = ContinuousCallback(condition, affect!, save_positions = (false, false))

    condition2(u, t, integrator) = t == tstop
    cb2 = DiscreteCallback(condition2, affect!, save_positions = (false, false))

    du01, dp1 = Zygote.gradient(
        (u0, p) -> sum(solve(prob, Tsit5(), u0 = u0, p = p,
            callback = cb2, tstops = [tstop],
            sensealg = BacksolveAdjoint(),
            saveat = tspan[2], save_start = false)),
        u0, p)

    du02, dp2 = Zygote.gradient(
        (u0, p) -> sum(solve(prob, Tsit5(), u0 = u0, p = p,
            callback = cb,
            sensealg = BacksolveAdjoint(),
            saveat = tspan[2], save_start = false)),
        u0, p)

    dstuff = ForwardDiff.gradient(
        (θ) -> sum(solve(prob, Tsit5(), u0 = θ[1:2], p = θ[3:4],
            callback = cb, saveat = tspan[2],
            save_start = false)),
        [u0; p])

    @info dstuff
    @test du01 ≈ dstuff[1:2]
    @test dp1 ≈ dstuff[3:4]
    @test du02 ≈ dstuff[1:2]
    @test dp2 ≈ dstuff[3:4]

    # no saving in Callbacks; prescribed vafter and vbefore; loss on the endpoint by slicing
    du01, dp1 = Zygote.gradient(
        (u0, p) -> sum(solve(prob, Tsit5(), u0 = u0, p = p,
            callback = cb2, tstops = [tstop],
            sensealg = BacksolveAdjoint())[end]),
        u0, p)

    du02, dp2 = Zygote.gradient(
        (u0, p) -> sum(solve(prob, Tsit5(), u0 = u0, p = p,
            callback = cb,
            sensealg = BacksolveAdjoint())[end]),
        u0, p)

    dstuff = ForwardDiff.gradient(
        (θ) -> sum(solve(prob, Tsit5(), u0 = θ[1:2], p = θ[3:4],
            callback = cb)[end]),
        [u0; p])

    @info dstuff
    @test du01 ≈ dstuff[1:2]
    @test dp1 ≈ dstuff[3:4]
    @test du02 ≈ dstuff[1:2]
    @test dp2 ≈ dstuff[3:4]

    # with saving in Callbacks; prescribed vafter and vbefore; loss on the endpoint
    cb = ContinuousCallback(condition, affect!, save_positions = (true, true))
    cb2 = DiscreteCallback(condition2, affect!, save_positions = (true, true))

    du01, dp1 = Zygote.gradient(
        (u0, p) -> sum(solve(prob, Tsit5(), u0 = u0, p = p,
            callback = cb2, tstops = [tstop],
            sensealg = BacksolveAdjoint(),
            saveat = tspan[2], save_start = false)),
        u0, p)

    du02, dp2 = Zygote.gradient(
        (u0, p) -> sum(solve(prob, Tsit5(), u0 = u0, p = p,
            callback = cb,
            sensealg = BacksolveAdjoint(),
            saveat = tspan[2], save_start = false)),
        u0, p)

    dstuff = ForwardDiff.gradient(
        (θ) -> sum(solve(prob, Tsit5(), u0 = θ[1:2], p = θ[3:4],
            callback = cb, saveat = tspan[2],
            save_start = false)),
        [u0; p])

    @info dstuff
    @test du01 ≈ dstuff[1:2]
    @test dp1 ≈ dstuff[3:4]
    @test du02 ≈ dstuff[1:2]
    @test dp2 ≈ dstuff[3:4]

    # with saving in Callbacks; prescribed vafter and vbefore; loss on the endpoint by slicing
    du01, dp1 = Zygote.gradient(
        (u0, p) -> sum(solve(prob, Tsit5(), u0 = u0, p = p,
            callback = cb2, tstops = [tstop],
            sensealg = BacksolveAdjoint())[end]),
        u0, p)

    du02, dp2 = Zygote.gradient(
        (u0, p) -> sum(solve(prob, Tsit5(), u0 = u0, p = p,
            callback = cb,
            sensealg = BacksolveAdjoint())[end]),
        u0, p)

    dstuff = ForwardDiff.gradient(
        (θ) -> sum(solve(prob, Tsit5(), u0 = θ[1:2], p = θ[3:4],
            callback = cb)[end]),
        [u0; p])

    @info dstuff
    @test du01 ≈ dstuff[1:2]
    @test dp1 ≈ dstuff[3:4]
    @test du02 ≈ dstuff[1:2]
    @test dp2 ≈ dstuff[3:4]

    # with saving in Callbacks;  different affect function
    function affect2!(integrator)
        integrator.u[2] = -integrator.p[2] * integrator.u[2]
    end
    cb = ContinuousCallback(condition, affect2!, save_positions = (true, true))

    cb2 = DiscreteCallback(condition2, affect2!, save_positions = (true, true))

    du01, dp1 = Zygote.gradient(
        (u0, p) -> sum(solve(prob, Tsit5(), u0 = u0, p = p,
            callback = cb2, tstops = [tstop],
            sensealg = BacksolveAdjoint(),
            saveat = tspan[2], save_start = false)),
        u0, p)

    du02, dp2 = Zygote.gradient(
        (u0, p) -> sum(solve(prob, Tsit5(), u0 = u0, p = p,
            callback = cb,
            sensealg = BacksolveAdjoint(),
            saveat = tspan[2], save_start = false)),
        u0, p)

    dstuff = ForwardDiff.gradient(
        (θ) -> sum(solve(prob, Tsit5(), u0 = θ[1:2], p = θ[3:4],
            callback = cb, saveat = tspan[2],
            save_start = false)),
        [u0; p])

    @info dstuff
    @test du01 ≈ dstuff[1:2]
    @test dp1 ≈ dstuff[3:4]
    @test du02 ≈ dstuff[1:2]
    @test dp2 ≈ dstuff[3:4]
    @test du01 ≈ du02
    @test dp1 ≈ dp2
end

@testset "Compare continuous with discrete callbacks" begin
    test_continuous_wrt_discrete_callback()
end
