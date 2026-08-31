using ModelingToolkit, OrdinaryDiffEq, SciMLSensitivity
using ModelingToolkit: t_nounits as t, D_nounits as D
using Zygote, ForwardDiff, FiniteDiff
using SciMLBase
import SciMLStructures as SS
using SymbolicIndexingInterface
using Test

const H0 = 1.0
const V0 = 0.0
const G0 = 9.81
const E0 = 0.7

const ABSTOL = 1.0e-10
const RELTOL = 1.0e-10
const SAVEAT = 0.0:0.25:2.0

function test_event_gradients(prob, idx; sensealgs, rtol = 1.0e-4, name = "")
    tunables, repack, _ = SS.canonicalize(SS.Tunable(), prob.p)
    tunables = collect(tunables)

    lossu0(u0, sensealg) = sum(
        abs2,
        Array(
            solve(
                remake(prob; u0), Tsit5();
                saveat = SAVEAT, abstol = ABSTOL, reltol = RELTOL, sensealg
            )
        )[idx, :]
    )
    lossp(x, sensealg) = sum(
        abs2,
        Array(
            solve(
                prob, Tsit5(); p = repack(x),
                saveat = SAVEAT, abstol = ABSTOL, reltol = RELTOL, sensealg
            )
        )[idx, :]
    )

    fd_u0 = FiniteDiff.finite_difference_gradient(u0 -> lossu0(u0, nothing), prob.u0)
    fd_p = FiniteDiff.finite_difference_gradient(x -> lossp(x, nothing), tunables)

    for (sensename, sensealg) in sensealgs
        @testset "$name $sensename" begin
            du0 = Zygote.gradient(u0 -> lossu0(u0, sensealg), prob.u0)[1]
            @test du0 ≈ fd_u0 rtol = rtol
            if !isempty(tunables)
                dp = Zygote.gradient(x -> lossp(x, sensealg), tunables)[1]
                @test dp ≈ fd_p rtol = rtol
            end
        end
    end
    return fd_u0, fd_p
end

const CONTINUOUS_SENSEALGS = [
    "InterpolatingAdjoint" => InterpolatingAdjoint(autojacvec = ReverseDiffVJP()),
    "InterpolatingAdjoint compiled tape" => InterpolatingAdjoint(
        autojacvec = ReverseDiffVJP(true)
    ),
    "GaussAdjoint" => GaussAdjoint(autojacvec = ReverseDiffVJP()),
    "QuadratureAdjoint" => QuadratureAdjoint(autojacvec = ReverseDiffVJP()),
    "BacksolveAdjoint" => BacksolveAdjoint(autojacvec = ReverseDiffVJP()),
]

@testset "MTK systems with events" begin
    @testset "Bouncing ball with a symbolic affect" begin
        @parameters g e
        @variables h(t) v(t)
        eqs = [D(h) ~ v, D(v) ~ -g]
        @mtkcompile ball = System(
            eqs, t, [h, v], [g, e];
            continuous_events = [[h ~ 0] => [v ~ -e * Pre(v)]]
        )
        prob = ODEProblem{true, SciMLBase.FullSpecialize}(
            ball, [h => H0, v => V0, g => G0, e => E0], (0.0, 2.0)
        )
        idx = variable_index(ball, h)

        sol = solve(prob, Tsit5(); saveat = SAVEAT, abstol = ABSTOL, reltol = RELTOL)
        @test length(sol.t) > length(SAVEAT)

        fd_u0, fd_p = test_event_gradients(
            prob, idx; sensealgs = CONTINUOUS_SENSEALGS, name = "ball"
        )

        @testset "ball ForwardDiff" begin
            fwd = ForwardDiff.gradient(
                u0 -> sum(
                    abs2,
                    Array(
                        solve(
                            remake(prob; u0), Tsit5();
                            saveat = SAVEAT, abstol = ABSTOL, reltol = RELTOL
                        )
                    )[idx, :]
                ),
                prob.u0
            )
            @test fwd ≈ fd_u0 rtol = 1.0e-4
        end

        @testset "ball EnzymeVJP $sensename" for (sensename, sensealg) in [
                "InterpolatingAdjoint" => InterpolatingAdjoint(autojacvec = EnzymeVJP()),
                "GaussAdjoint" => GaussAdjoint(autojacvec = EnzymeVJP()),
            ]
            du0 = Zygote.gradient(
                u0 -> sum(
                    abs2,
                    Array(
                        solve(
                            remake(prob; u0), Tsit5(); saveat = SAVEAT,
                            abstol = ABSTOL, reltol = RELTOL, sensealg
                        )
                    )[idx, :]
                ),
                prob.u0
            )[1]
            @test du0 ≈ fd_u0 rtol = 1.0e-4
        end
    end

    @testset "Bouncing ball with an ImperativeAffect" begin
        @parameters g e
        @variables h(t) v(t)
        eqs = [D(h) ~ v, D(v) ~ -g]
        affect = ModelingToolkit.ImperativeAffect(
            modified = (; v), observed = (; e)
        ) do m, o, c, i
            return (; v = -o.e * m.v)
        end
        @mtkcompile ball = System(
            eqs, t, [h, v], [g, e]; continuous_events = [[h ~ 0] => affect]
        )
        prob = ODEProblem{true, SciMLBase.FullSpecialize}(
            ball, [h => H0, v => V0, g => G0, e => E0], (0.0, 2.0)
        )
        idx = variable_index(ball, h)

        sol = solve(prob, Tsit5(); saveat = SAVEAT, abstol = ABSTOL, reltol = RELTOL)
        @test length(sol.t) > length(SAVEAT)
        test_event_gradients(
            prob, idx; sensealgs = CONTINUOUS_SENSEALGS[1:3], name = "imperative"
        )
    end

    @testset "Affect that changes a parameter" begin
        @parameters g
        @discretes e(t) = E0
        @variables h(t) v(t)
        eqs = [D(h) ~ v, D(v) ~ -g]
        @mtkcompile ball = System(
            eqs, t, [h, v], [g, e];
            continuous_events = [
                ModelingToolkit.SymbolicContinuousCallback(
                    [h ~ 0], [v ~ -Pre(e) * Pre(v), e ~ 0.9 * Pre(e)];
                    discrete_parameters = [e], iv = t
                ),
            ]
        )
        prob = ODEProblem{true, SciMLBase.FullSpecialize}(
            ball, [h => H0, v => V0, g => G0], (0.0, 2.0)
        )
        idx = variable_index(ball, h)

        sol = solve(prob, Tsit5(); saveat = SAVEAT, abstol = ABSTOL, reltol = RELTOL)
        @test length(sol.t) > length(SAVEAT)
        fd_u0, _ = test_event_gradients(
            prob, idx;
            sensealgs = CONTINUOUS_SENSEALGS[[1, 3]], name = "parameter affect"
        )

        @testset "parameter affect compiled tape" begin
            sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))
            du0 = Zygote.gradient(
                u0 -> sum(
                    abs2,
                    Array(
                        solve(
                            remake(prob; u0), Tsit5(); saveat = SAVEAT,
                            abstol = ABSTOL, reltol = RELTOL, sensealg
                        )
                    )[idx, :]
                ),
                prob.u0
            )[1]
            @test du0 ≈ fd_u0 rtol = 1.0e-4
        end
    end

    @testset "Two events (VectorContinuousCallback)" begin
        @parameters g e
        @variables h(t) v(t)
        eqs = [D(h) ~ v, D(v) ~ -g]
        @mtkcompile ball = System(
            eqs, t, [h, v], [g, e];
            continuous_events = [
                [h ~ 0] => [v ~ -e * Pre(v)],
                [h ~ 1.5] => [v ~ -e * Pre(v)],
            ]
        )
        prob = ODEProblem{true, SciMLBase.FullSpecialize}(
            ball, [h => H0, v => 4.0, g => G0, e => E0], (0.0, 2.0)
        )
        idx = variable_index(ball, h)

        @test prob.kwargs[:callback] isa VectorContinuousCallback
        sol = solve(prob, Tsit5(); saveat = SAVEAT, abstol = ABSTOL, reltol = RELTOL)
        @test length(sol.t) > length(SAVEAT)
        test_event_gradients(
            prob, idx; sensealgs = CONTINUOUS_SENSEALGS[1:3], name = "two events"
        )
    end

    @testset "Discrete (time-triggered) event" begin
        @parameters g e
        @variables h(t) v(t)
        eqs = [D(h) ~ v, D(v) ~ -g]
        @mtkcompile ball = System(
            eqs, t, [h, v], [g, e];
            continuous_events = [[h ~ 0] => [v ~ -e * Pre(v)]],
            discrete_events = [1.0 => [v ~ Pre(v) + 0.5]]
        )
        prob = ODEProblem{true, SciMLBase.FullSpecialize}(
            ball, [h => H0, v => V0, g => G0, e => E0], (0.0, 2.0)
        )
        idx = variable_index(ball, h)

        test_event_gradients(
            prob, idx; sensealgs = CONTINUOUS_SENSEALGS[1:3], name = "discrete event"
        )
    end

    @testset "Plain vector parameters are unchanged" begin
        N0 = [0.0]
        p = [100.0, 50.0]
        tspan = (0.0, 10.0)

        f(D, u, p, t) = (D[1] = p[1] - u[1])
        condition(u, t, integrator) = u[1] - 3 // 4 * integrator.p[1]
        affect!(integrator) = integrator.u[1] += integrator.p[2]
        cb = ContinuousCallback(condition, affect!, save_positions = (false, false))
        prob = ODEProblem(f, N0, tspan, p)

        function loss(p, sensealg)
            _prob = remake(prob; p)
            _sol = solve(
                _prob, Tsit5(); callback = cb, abstol = 1.0e-14, reltol = 1.0e-14,
                sensealg
            )
            return _sol.u[end][1]
        end

        gFD = ForwardDiff.gradient(p -> loss(p, nothing), p)
        @testset "plain vector $i" for (i, sensealg) in enumerate(
                [
                    InterpolatingAdjoint(autojacvec = ReverseDiffVJP()),
                    InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)),
                    GaussAdjoint(autojacvec = ReverseDiffVJP()),
                    InterpolatingAdjoint(autojacvec = EnzymeVJP()),
                ]
            )
            @test gFD ≈ Zygote.gradient(p -> loss(p, sensealg), p)[1] rtol = 1.0e-10
        end
    end
end
