using OrdinaryDiffEq, DiffEqCallbacks
using SciMLSensitivity, Zygote, Test
import ForwardDiff
import FiniteDiff

abstol = 1e-6
reltol = 1e-6
savingtimes = 0.1

function test_discrete_callback(cb, tstops, g)
    function fiip(du, u, p, t)
        #du[1] = dx = p[1]*u[1]
        du[:] .= p[1] * u
    end

    p = Float64[0.8123198]
    u0 = Float64[1.0]

    prob = ODEProblem(fiip, u0, (0.0, 1.0), p)

    @show g(solve(prob, Tsit5(), callback = cb, tstops = tstops, abstol = abstol,
        reltol = reltol, saveat = savingtimes))

    du01, dp1 = Zygote.gradient(
        (u0, p) -> g(solve(prob, Tsit5(), u0 = u0, p = p,
            callback = cb, tstops = tstops,
            abstol = abstol, reltol = reltol,
            saveat = savingtimes,
            sensealg = ForwardDiffSensitivity(;
                convert_tspan = true))),
        u0, p)

    dstuff1 = ForwardDiff.gradient(
        (θ) -> g(solve(prob, Tsit5(), u0 = θ[1:1], p = θ[2:2],
            callback = cb, tstops = tstops,
            abstol = abstol, reltol = reltol,
            saveat = savingtimes)),
        [u0; p])

    dstuff2 = FiniteDiff.finite_difference_gradient(
        (θ) -> g(solve(prob, Tsit5(),
            u0 = θ[1:1], p = θ[2:2],
            callback = cb,
            tstops = tstops,
            abstol = abstol,
            reltol = reltol,
            saveat = savingtimes)),
        [u0; p])

    @show du01 dp1 dstuff1 dstuff2
    @test du01≈dstuff1[1:1] atol=1e-6
    @test dp1≈dstuff1[2:2] atol=1e-6
    @test du01≈dstuff2[1:1] atol=1e-6
    @test dp1≈dstuff2[2:2] atol=1e-6
end

@testset "ForwardDiffSensitivity: Discrete callbacks" begin
    g(u) = sum(Array(u) .^ 2)
    @testset "reset to initial condition" begin
        affecttimes = range(0.0, 1.0, length = 6)[2:end]
        u0 = [1.0]
        condition(u, t, integrator) = t ∈ affecttimes
        affect!(integrator) = (integrator.u .= u0; @show "triggered!")
        cb = DiscreteCallback(condition, affect!, save_positions = (false, false))
        test_discrete_callback(cb, affecttimes, g)
    end
end
