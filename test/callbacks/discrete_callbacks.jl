using OrdinaryDiffEq, Zygote, Reactant
using SciMLSensitivity, Test, ForwardDiff

abstol = 1.0e-12
reltol = 1.0e-12
savingtimes = 0.5

function test_discrete_callback(cb, tstops, g, dg!, cboop = nothing, tprev = false)
    function fiip(du, u, p, t)
        du[1] = dx = p[1] * u[1] - p[2] * u[1] * u[2]
        du[2] = dy = -p[3] * u[2] + p[4] * u[1] * u[2]
        return nothing
    end
    function foop(u, p, t)
        dx = p[1] * u[1] - p[2] * u[1] * u[2]
        dy = -p[3] * u[2] + p[4] * u[1] * u[2]
        return [dx, dy]
    end

    p = [1.5, 1.0, 3.0, 1.0]
    u0 = [1.0; 1.0]

    prob = ODEProblem(fiip, u0, (0.0, 10.0), p)
    proboop = ODEProblem(foop, u0, (0.0, 10.0), p)

    sol1 = solve(
        prob, Tsit5(); u0, p, callback = cb, tstops,
        abstol, reltol, saveat = savingtimes
    )
    sol2 = solve(
        prob, Tsit5(); u0, p, tstops, abstol, reltol, saveat = savingtimes
    )

    if cb.save_positions == [1, 1]
        @test length(sol1.t) != length(sol2.t)
    else
        @test length(sol1.t) == length(sol2.t)
    end

    du01, dp1 = Zygote.gradient(
        (
            u0,
            p,
        ) -> g(
            solve(
                prob, Tsit5(); u0, p,
                callback = cb, tstops,
                abstol, reltol,
                saveat = savingtimes,
                sensealg = BacksolveAdjoint()
            )
        ),
        u0, p
    )

    du01b, dp1b = Zygote.gradient(
        (
            u0,
            p,
        ) -> g(
            solve(
                proboop, Tsit5(); u0, p,
                callback = cb, tstops,
                abstol, reltol,
                saveat = savingtimes,
                sensealg = BacksolveAdjoint()
            )
        ),
        u0, p
    )

    du01c, dp1c = Zygote.gradient(
        (
            u0,
            p,
        ) -> g(
            solve(
                proboop, Tsit5(); u0, p,
                callback = cb, tstops,
                abstol, reltol,
                saveat = savingtimes,
                sensealg = BacksolveAdjoint(checkpointing = false)
            )
        ),
        u0, p
    )

    if cboop === nothing
        du02, dp2 = Zygote.gradient(
            (
                u0,
                p,
            ) -> g(
                solve(
                    prob, Tsit5(); u0, p,
                    callback = cb, tstops,
                    abstol, reltol,
                    saveat = savingtimes,
                    sensealg = ReverseDiffAdjoint()
                )
            ),
            u0, p
        )
    else
        du02, dp2 = Zygote.gradient(
            (
                u0,
                p,
            ) -> g(
                solve(
                    prob, Tsit5(); u0, p,
                    callback = cboop, tstops,
                    abstol, reltol,
                    saveat = savingtimes,
                    sensealg = ReverseDiffAdjoint()
                )
            ),
            u0, p
        )
    end

    du03, dp3 = Zygote.gradient(
        (
            u0,
            p,
        ) -> g(
            solve(
                prob, Tsit5(), u0 = u0, p = p,
                callback = cb, tstops = tstops,
                abstol = abstol, reltol = reltol,
                saveat = savingtimes,
                sensealg = InterpolatingAdjoint(checkpointing = true)
            )
        ),
        u0, p
    )

    du03c, dp3c = Zygote.gradient(
        (
            u0,
            p,
        ) -> g(
            solve(
                prob, Tsit5(); u0, p,
                callback = cb, tstops,
                abstol, reltol,
                saveat = savingtimes,
                sensealg = InterpolatingAdjoint(checkpointing = false)
            )
        ),
        u0, p
    )

    du04, dp4 = Zygote.gradient(
        (
            u0,
            p,
        ) -> g(
            solve(
                prob, Tsit5(); u0, p,
                callback = cb, tstops,
                abstol, reltol,
                saveat = savingtimes,
                sensealg = QuadratureAdjoint()
            )
        ),
        u0, p
    )

    du05, dp5 = Zygote.gradient(
        (
            u0,
            p,
        ) -> g(
            solve(
                prob, Tsit5(); u0, p,
                callback = cb, tstops,
                abstol, reltol,
                saveat = savingtimes,
                sensealg = GaussAdjoint()
            )
        ),
        u0, p
    )

    dstuff = ForwardDiff.gradient(
        (θ) -> g(
            solve(
                prob, Tsit5(); u0 = θ[1:2], p = θ[3:6],
                callback = cb, tstops,
                abstol, reltol,
                saveat = savingtimes
            )
        ),
        [u0; p]
    )

    @info dstuff

    # tests wrt discrete sensitivities
    if tprev
        # tprev depends on stepping behaviour of integrator. Thus sensitivities are necessarily (slightly) different.
        @test du02 ≈ dstuff[1:2] rtol = 1.0e-3
        @test dp2 ≈ dstuff[3:6] rtol = 1.0e-3
        @test du01 ≈ dstuff[1:2] rtol = 1.0e-3
        @test dp1 ≈ dstuff[3:6] rtol = 1.0e-3
        @test du01 ≈ du02 rtol = 1.0e-3
        @test dp1 ≈ dp2 rtol = 1.0e-3
    else
        @test du02 ≈ dstuff[1:2]
        @test dp2 ≈ dstuff[3:6]
        @test du01 ≈ dstuff[1:2]
        @test dp1 ≈ dstuff[3:6]
        @test du01 ≈ du02
        @test dp1 ≈ dp2
    end

    # tests wrt continuous sensitivities
    @test du01b ≈ du01
    @test dp1b ≈ dp1
    @test du01c ≈ du01
    @test dp1c ≈ dp1
    @test du01 ≈ du03 rtol = 1.0e-7
    @test du01 ≈ du03c rtol = 1.0e-7
    @test du03 ≈ du03c
    @test du01 ≈ du04
    @test du01 ≈ du05
    @test dp1 ≈ dp3
    @test dp1 ≈ dp3c
    @test dp1 ≈ dp4 rtol = 1.0e-7
    @test dp1 ≈ dp5 rtol = 1.0e-7

    cb2 = SciMLSensitivity.track_callbacks(
        CallbackSet(cb), prob.tspan[1], prob.u0, prob.p,
        BacksolveAdjoint(autojacvec = ReverseDiffVJP())
    )
    sol_track = solve(
        prob, Tsit5(); u0, p, callback = cb2, tstops,
        abstol, reltol, saveat = savingtimes
    )
    #cb_adj = SciMLSensitivity.setup_reverse_callbacks(cb2,BacksolveAdjoint())

    adj_prob = ODEAdjointProblem(
        sol_track, BacksolveAdjoint(autojacvec = ReverseDiffVJP()),
        Tsit5(),
        sol_track.t, dg!;
        callback = cb2,
        abstol, reltol
    )
    adj_sol = solve(adj_prob, Tsit5(); abstol, reltol)
    @test du01 ≈ adj_sol[1:2, end]
    return @test dp1 ≈ adj_sol[3:6, end]
end

@testset "Discrete callbacks" begin
    @testset "ODEs" begin
        println("ODEs")
        @testset "simple loss function" begin
            g(sol) = sum(sol)
            function dg!(out, u, p, t, i)
                (out .= 1)
            end
            @testset "callbacks with no effect" begin
                condition(u, t, integrator) = t == 5
                affect!(integrator) = integrator.u[1] += 0.0
                cb = DiscreteCallback(condition, affect!, save_positions = (false, false))
                tstops = [5.0]
                test_discrete_callback(cb, tstops, g, dg!)
            end
            @testset "callbacks with no effect except saving the state" begin
                condition(u, t, integrator) = t == 5
                affect!(integrator) = integrator.u[1] += 0.0
                cb = DiscreteCallback(condition, affect!)
                tstops = [5.0]
                test_discrete_callback(cb, tstops, g, dg!)
            end
            @testset "callback at single time point" begin
                condition(u, t, integrator) = t == 5
                affect!(integrator) = integrator.u[1] += 2.0
                cb = DiscreteCallback(condition, affect!)
                tstops = [5.0]
                test_discrete_callback(cb, tstops, g, dg!)
            end
            @testset "callback at multiple time points" begin
                affecttimes = [2.03, 4.0, 8.0]
                condition(u, t, integrator) = t ∈ affecttimes
                affect!(integrator) = integrator.u[1] += 2.0
                cb = DiscreteCallback(condition, affect!)
                test_discrete_callback(cb, affecttimes, g, dg!)
            end
            @testset "state-dependent += callback at single time point" begin
                condition(u, t, integrator) = t == 5
                function affect!(integrator)
                    (integrator.u .+= integrator.p[2] / 8 * sin.(integrator.u))
                end
                cb = DiscreteCallback(condition, affect!)
                tstops = [5.0]
                test_discrete_callback(cb, tstops, g, dg!)
            end
            @testset "other callback at single time point" begin
                condition(u, t, integrator) = t == 5
                affect!(integrator) = (integrator.u[1] = 2.0; @show "triggered!")
                cb = DiscreteCallback(condition, affect!)
                tstops = [5.0]
                test_discrete_callback(cb, tstops, g, dg!)
            end
            @testset "parameter changing callback at single time point" begin
                condition(u, t, integrator) = t == 5.1
                affect!(integrator) = (integrator.p .= 2 * integrator.p .- 0.5)
                affect(integrator) = (integrator.p = 2 * integrator.p .- 0.5)
                cb = DiscreteCallback(condition, affect!)
                cboop = DiscreteCallback(condition, affect)
                cb = DiscreteCallback(condition, affect!)
                tstops = [5.1]
                test_discrete_callback(cb, tstops, g, dg!, cboop)
            end
            @testset "tprev dependent callback" begin
                condition(u, t, integrator) = t == 5
                function affect!(integrator)
                    (
                        @show integrator.tprev;
                        integrator.u[1] += integrator.t -
                            integrator.tprev
                    )
                end
                cb = DiscreteCallback(condition, affect!)
                tstops = [4.999, 5.0]
                test_discrete_callback(cb, tstops, g, dg!, nothing, true)
            end
        end
        @testset "MSE loss function" begin
            g(u) = sum((1.0 .- u) .^ 2) ./ 2
            dg!(out, u, p, t, i) = (out .= -1.0 .+ u)
            @testset "callbacks with no effect" begin
                condition(u, t, integrator) = t == 5
                affect!(integrator) = integrator.u[1] += 0.0
                cb = DiscreteCallback(condition, affect!, save_positions = (false, false))
                tstops = [5.0]
                test_discrete_callback(cb, tstops, g, dg!)
            end
            @testset "callbacks with no effect except saving the state" begin
                condition(u, t, integrator) = t == 5
                affect!(integrator) = integrator.u[1] += 0.0
                cb = DiscreteCallback(condition, affect!)
                tstops = [5.0]
                test_discrete_callback(cb, tstops, g, dg!)
            end
            @testset "callback at single time point" begin
                condition(u, t, integrator) = t == 5
                affect!(integrator) = integrator.u[1] += 2.0
                cb = DiscreteCallback(condition, affect!)
                tstops = [5.0]
                test_discrete_callback(cb, tstops, g, dg!)
            end
            @testset "callback at multiple time points" begin
                affecttimes = [2.03, 4.0, 8.0]
                condition(u, t, integrator) = t ∈ affecttimes
                affect!(integrator) = integrator.u[1] += 2.0
                cb = DiscreteCallback(condition, affect!)
                test_discrete_callback(cb, affecttimes, g, dg!)
            end
            @testset "state-dependent += callback at single time point" begin
                condition(u, t, integrator) = t == 5
                function affect!(integrator)
                    (integrator.u .+= integrator.p[2] / 8 * sin.(integrator.u))
                end
                cb = DiscreteCallback(condition, affect!)
                tstops = [5.0]
                test_discrete_callback(cb, tstops, g, dg!)
            end
            @testset "other callback at single time point" begin
                condition(u, t, integrator) = t == 5
                affect!(integrator) = (integrator.u[1] = 2.0; @show "triggered!")
                cb = DiscreteCallback(condition, affect!)
                tstops = [5.0]
                test_discrete_callback(cb, tstops, g, dg!)
            end
            @testset "parameter changing callback at single time point" begin
                condition(u, t, integrator) = t == 5.1
                affect!(integrator) = (integrator.p .= 2 * integrator.p .- 0.5)
                affect(integrator) = (integrator.p = 2 * integrator.p .- 0.5)
                cb = DiscreteCallback(condition, affect!)
                cboop = DiscreteCallback(condition, affect)
                tstops = [5.1]
                test_discrete_callback(cb, tstops, g, dg!, cboop)
            end
            @testset "tprev dependent callback" begin
                condition(u, t, integrator) = t == 5
                function affect!(integrator)
                    (
                        @show integrator.tprev;
                        integrator.u[1] += integrator.t -
                            integrator.tprev
                    )
                end
                cb = DiscreteCallback(condition, affect!)
                tstops = [4.999, 5.0]
                test_discrete_callback(cb, tstops, g, dg!, nothing, true)
            end
        end
        @testset "Dosing example" begin
            N0 = [0.0] # initial population
            p = [100.0, 50.0] # steady-state pop., M
            tspan = (0.0, 10.0) # integration time
            f(D, u, p, t) = (D[1] = p[1] - u[1]) # system
            prob = ODEProblem(f, N0, tspan, p)

            # at time tinject1 we inject M1 cells
            tinject = 8.0
            condition(u, t, integrator) = t == tinject
            affect(integrator) = integrator.u[1] += integrator.p[2]
            cb = DiscreteCallback(condition, affect)

            function loss(p)
                _prob = remake(prob; p)
                _sol = solve(
                    _prob, Tsit5(); callback = cb,
                    abstol = 1.0e-14, reltol = 1.0e-14, tstops = [tinject],
                    sensealg = BacksolveAdjoint(autojacvec = EnzymeVJP())
                )
                _sol.u[end][1]
            end

            gFD = ForwardDiff.gradient(loss, p)
            gZy = Zygote.gradient(loss, p)[1]
            @test gFD ≈ gZy
        end
        # ReactantVJP: f and callbacks use scalar indexing (D[1], u[1], p[1], etc.)
        # which can fail during Reactant tracing (upstream limitation).
        @testset "Dosing example ReactantVJP" begin
            @test_broken begin
                N0 = [0.0]
                p_dose = [100.0, 50.0]
                tspan_dose = (0.0, 10.0)
                f_dose(D, u, p, t) = (D[1] = p[1] - u[1])

                prob_dose = ODEProblem(f_dose, N0, tspan_dose, p_dose)

                tinject = 8.0
                condition_dose(u, t, integrator) = t == tinject
                affect_dose(integrator) = integrator.u[1] += integrator.p[2]
                cb_dose = DiscreteCallback(condition_dose, affect_dose)

                function loss_dose(p)
                    _prob = remake(prob_dose; p)
                    _sol = solve(
                        _prob, Tsit5(); callback = cb_dose,
                        abstol = 1.0e-14, reltol = 1.0e-14, tstops = [tinject],
                        sensealg = BacksolveAdjoint(autojacvec = ReactantVJP())
                    )
                    _sol.u[end][1]
                end

                gFD_dose = ForwardDiff.gradient(loss_dose, p_dose)
                gZy_dose = Zygote.gradient(loss_dose, p_dose)[1]
                gFD_dose ≈ gZy_dose
            end
        end
    end
end
