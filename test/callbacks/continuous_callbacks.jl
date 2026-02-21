using OrdinaryDiffEq, Zygote, Reactant
using SciMLSensitivity, Test, ForwardDiff, FiniteDiff

abstol = 1.0e-12
reltol = 1.0e-12
savingtimes = 0.5

function test_continuous_callback(cb, g, dg!; only_backsolve = false)
    function fiip(du, u, p, t)
        du[1] = u[2]
        du[2] = -p[1]
        return nothing
    end
    function foop(u, p, t)
        dx = u[2]
        dy = -p[1]
        return [dx, dy]
    end

    u0 = [5.0, 0.0]
    tspan = (0.0, 2.5)
    p = [9.8, 0.8]

    prob = ODEProblem(fiip, u0, tspan, p)
    proboop = ODEProblem(fiip, u0, tspan, p)

    sol1 = solve(
        prob, Tsit5(); u0, p, callback = cb, abstol, reltol,
        saveat = savingtimes
    )
    sol2 = solve(prob, Tsit5(); u0, p, abstol, reltol, saveat = savingtimes)

    if cb.save_positions == [1, 1]
        @test length(sol1.t) != length(sol2.t)
    else
        @test length(sol1.t) == length(sol2.t)
    end

    du01, dp1 = @time Zygote.gradient(
        (u0, p) -> g(
            solve(
                prob, Tsit5(); u0, p,
                callback = cb, abstol, reltol,
                saveat = savingtimes,
                sensealg = BacksolveAdjoint()
            )
        ),
        u0, p
    )

    du01b, dp1b = Zygote.gradient(
        (u0, p) -> g(
            solve(
                proboop, Tsit5(); u0, p,
                callback = cb, abstol, reltol,
                saveat = savingtimes,
                sensealg = BacksolveAdjoint()
            )
        ),
        u0, p
    )

    du01c, dp1c = Zygote.gradient(
        (u0, p) -> g(
            solve(
                proboop, Tsit5(); u0, p,
                callback = cb,
                abstol, reltol,
                saveat = savingtimes,
                sensealg = BacksolveAdjoint(checkpointing = false)
            )
        ),
        u0, p
    )

    if !only_backsolve
        du02, dp2 = @time Zygote.gradient(
            (u0, p) -> g(
                solve(
                    prob, Tsit5(); u0, p,
                    callback = cb, abstol, reltol,
                    saveat = savingtimes,
                    sensealg = ReverseDiffAdjoint()
                )
            ),
            u0, p
        )

        du03, dp3 = @time Zygote.gradient(
            (u0, p) -> g(
                solve(
                    prob, Tsit5(); u0, p,
                    callback = cb, abstol, reltol,
                    saveat = savingtimes,
                    sensealg = InterpolatingAdjoint(checkpointing = true)
                )
            ),
            u0, p
        )

        du03c, dp3c = Zygote.gradient(
            (u0, p) -> g(
                solve(
                    prob, Tsit5(); u0, p,
                    callback = cb, abstol, reltol,
                    saveat = savingtimes,
                    sensealg = InterpolatingAdjoint(checkpointing = false)
                )
            ),
            u0, p
        )

        du04, dp4 = @time Zygote.gradient(
            (u0, p) -> g(
                solve(
                    prob, Tsit5(); u0, p,
                    callback = cb, abstol, reltol,
                    saveat = savingtimes,
                    sensealg = QuadratureAdjoint()
                )
            ),
            u0, p
        )
    end
    dstuff = @time ForwardDiff.gradient(
        (θ) -> g(
            solve(
                prob, Tsit5(); u0 = θ[1:2],
                p = θ[3:4], callback = cb,
                abstol, reltol,
                saveat = savingtimes
            )
        ),
        [u0; p]
    )

    @info dstuff

    @test du01 ≈ dstuff[1:2] rtol = 1.0e-5
    @test dp1 ≈ dstuff[3:4] rtol = 1.0e-5
    @test du01b ≈ dstuff[1:2] rtol = 1.0e-5
    @test dp1b ≈ dstuff[3:4] rtol = 1.0e-5
    @test du01c ≈ dstuff[1:2] rtol = 1.0e-5
    @test dp1c ≈ dstuff[3:4] rtol = 1.0e-5
    if !only_backsolve
        @test du01 ≈ du02
        @test du01 ≈ du03 rtol = 1.0e-7
        @test du01 ≈ du03c rtol = 1.0e-7
        @test du03 ≈ du03c
        @test du01 ≈ du04
        @test dp1 ≈ dp2
        @test dp1 ≈ dp3
        @test dp1 ≈ dp3c
        @test dp3 ≈ dp3c
        @test dp1 ≈ dp4 rtol = 1.0e-7

        @test du02 ≈ dstuff[1:2] rtol = 1.0e-5
        @test dp2 ≈ dstuff[3:4] rtol = 1.0e-5
    end

    cb2 = SciMLSensitivity.track_callbacks(
        CallbackSet(cb), prob.tspan[1], prob.u0, prob.p,
        BacksolveAdjoint(autojacvec = ReverseDiffVJP())
    )
    sol_track = solve(
        prob, Tsit5(); u0, p, callback = cb2, abstol, reltol,
        saveat = savingtimes
    )

    adj_prob = ODEAdjointProblem(
        sol_track, BacksolveAdjoint(autojacvec = ReverseDiffVJP()),
        Tsit5(),
        sol_track.t, dg!;
        callback = cb2,
        abstol, reltol
    )
    adj_sol = solve(adj_prob, Tsit5(); abstol, reltol)
    @test du01 ≈ adj_sol[1:2, end]
    return @test dp1 ≈ adj_sol[3:4, end]
end

println("Continuous Callbacks")
@testset "Continuous callbacks" begin
    @testset "simple loss function bouncing ball" begin
        g(sol) = sum(Array(sol))
        function dg!(out, u, p, t, i)
            (out .= 1)
        end

        @testset "callbacks with no effect" begin
            condition(u, t, integrator) = u[1] # Event when event_f(u,t) == 0
            affect!(integrator) = (integrator.u[2] += 0)
            cb = ContinuousCallback(condition, affect!, save_positions = (false, false))
            test_continuous_callback(cb, g, dg!)
        end
        @testset "callbacks with no effect except saving the state" begin
            condition(u, t, integrator) = u[1]
            affect!(integrator) = (integrator.u[2] += 0)
            cb = ContinuousCallback(condition, affect!, save_positions = (true, true))
            test_continuous_callback(cb, g, dg!)
        end
        @testset "+= callback" begin
            condition(u, t, integrator) = u[1]
            affect!(integrator) = (integrator.u[2] += 50.0)
            cb = ContinuousCallback(condition, affect!, save_positions = (true, true))
            test_continuous_callback(cb, g, dg!)
        end
        @testset "= callback with parameter dependence and save" begin
            condition(u, t, integrator) = u[1]
            affect!(integrator) = (integrator.u[2] = -integrator.p[2] * integrator.u[2])
            cb = ContinuousCallback(condition, affect!, save_positions = (true, true))
            test_continuous_callback(cb, g, dg!)
        end
        @testset "= callback with parameter dependence but without save" begin
            condition(u, t, integrator) = u[1]
            affect!(integrator) = (integrator.u[2] = -integrator.p[2] * integrator.u[2])
            cb = ContinuousCallback(condition, affect!, save_positions = (false, false))
            test_continuous_callback(cb, g, dg!; only_backsolve = true)
        end
        @testset "= callback with non-linear affect" begin
            condition(u, t, integrator) = u[1]
            affect!(integrator) = (integrator.u[2] = integrator.u[2]^2)
            cb = ContinuousCallback(condition, affect!, save_positions = (true, true))
            test_continuous_callback(cb, g, dg!)
        end
        @testset "= callback with terminate" begin
            condition(u, t, integrator) = u[1]
            function affect!(integrator)
                (
                    integrator.u[2] = -integrator.p[2] * integrator.u[2];
                    terminate!(integrator)
                )
            end
            cb = ContinuousCallback(condition, affect!, save_positions = (true, true))
            test_continuous_callback(cb, g, dg!; only_backsolve = true)
        end
    end
    @testset "MSE loss function bouncing-ball like" begin
        g(u) = sum((1.0 .- u) .^ 2) ./ 2
        dg!(out, u, p, t, i) = (out .= -1.0 .+ u)
        condition(u, t, integrator) = u[1]
        @testset "callback with non-linear affect" begin
            function affect!(integrator)
                integrator.u[1] += 3.0
                integrator.u[2] = integrator.u[2]^2
            end
            cb = ContinuousCallback(condition, affect!, save_positions = (true, true))
            test_continuous_callback(cb, g, dg!)
        end
        @testset "callback with non-linear affect and terminate" begin
            function affect!(integrator)
                integrator.u[1] += 3.0
                integrator.u[2] = integrator.u[2]^2
                terminate!(integrator)
            end
            cb = ContinuousCallback(condition, affect!, save_positions = (true, true))
            test_continuous_callback(cb, g, dg!; only_backsolve = true)
        end
    end
    @testset "MSE loss function free particle" begin
        g(u) = sum((1.0 .- u) .^ 2) ./ 2
        function fiip(du, u, p, t)
            du[1] = u[2]
            du[2] = 0
        end
        function foop(u, p, t)
            dx = u[2]
            dy = 0
            [dx, dy]
        end

        u0 = [5.0, -1.0]
        p = [0.0, 0.0]
        tspan = (0.0, 2.0)

        prob = ODEProblem(fiip, u0, tspan, p)
        proboop = ODEProblem(fiip, u0, tspan, p)

        condition(u, t, integrator) = u[1] # Event when event_f(u,t) == 0
        affect!(integrator) = (integrator.u[2] = -integrator.u[2])
        cb = ContinuousCallback(condition, affect!)

        du01, dp1 = Zygote.gradient(
            (
                u0,
                p,
            ) -> g(
                solve(
                    prob, Tsit5(); u0, p,
                    callback = cb, abstol, reltol,
                    saveat = savingtimes,
                    sensealg = BacksolveAdjoint()
                )
            ),
            u0, p
        )

        dstuff = @time ForwardDiff.gradient(
            (θ) -> g(
                solve(
                    prob, Tsit5(); u0 = θ[1:2],
                    p = θ[3:4], callback = cb,
                    abstol, reltol,
                    saveat = savingtimes
                )
            ),
            [u0; p]
        )

        @info dstuff

        @test du01 ≈ dstuff[1:2]
        @test dp1 ≈ dstuff[3:4]
    end
    @testset "Re-compile tape in ReverseDiffVJP" begin
        N0 = [0.0] # initial population
        p = [100.0, 50.0] # steady-state pop., M
        tspan = (0.0, 10.0) # integration time

        # system
        f(D, u, p, t) = (D[1] = p[1] - u[1])

        # when N = 3α/4 we inject M cells.
        condition(u, t, integrator) = u[1] - 3 // 4 * integrator.p[1]
        affect!(integrator) = integrator.u[1] += integrator.p[2]
        cb = ContinuousCallback(condition, affect!, save_positions = (false, false))
        prob = ODEProblem(f, N0, tspan, p)

        function loss(p, cb, sensealg)
            _prob = remake(prob; p)
            _sol = solve(
                _prob, Tsit5(); callback = cb,
                abstol = 1.0e-14, reltol = 1.0e-14,
                sensealg
            )
            _sol.u[end][1]
        end

        gND = FiniteDiff.finite_difference_gradient(p -> loss(p, cb, nothing), p)
        gFD = ForwardDiff.gradient(p -> loss(p, cb, nothing), p)
        # @show gND # [0.9999546000702386, 0.00018159971904994378]
        @test gND ≈ gFD rtol = 1.0e-10

        for compile_tape in (true, false)
            sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(compile_tape))
            gZy = Zygote.gradient(p -> loss(p, cb, sensealg), p)[1]
            @test gFD ≈ gZy rtol = 1.0e-10
        end
        sensealg = InterpolatingAdjoint(autojacvec = EnzymeVJP())
        gZy = Zygote.gradient(p -> loss(p, cb, sensealg), p)[1]
        @test gFD ≈ gZy rtol = 1.0e-10

        sensealg = GaussAdjoint(autojacvec = EnzymeVJP())
        gZy = Zygote.gradient(p -> loss(p, cb, sensealg), p)[1]
        @test gFD ≈ gZy rtol = 1.0e-10

        # ReactantVJP with callbacks
        sensealg = InterpolatingAdjoint(autojacvec = ReactantVJP())
        gZy = Zygote.gradient(p -> loss(p, cb, sensealg), p)[1]
        @test gFD ≈ gZy rtol = 1.0e-10

        sensealg = GaussAdjoint(autojacvec = ReactantVJP())
        gZy = Zygote.gradient(p -> loss(p, cb, sensealg), p)[1]
        @test gFD ≈ gZy rtol = 1.0e-10
    end
end
