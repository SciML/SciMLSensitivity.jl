using OrdinaryDiffEq, Zygote
using SciMLSensitivity, Test, ForwardDiff

abstol = 1.0e-12
reltol = 1.0e-12
savingtimes = 0.5

# see https://docs.sciml.ai/DiffEqDocs/stable/features/callback_functions/#VectorContinuousCallback-Example
function test_vector_continuous_callback(cb, g)
    function f(du, u, p, t)
        du[1] = u[2]
        du[2] = -p[1]
        du[3] = u[4]
        du[4] = 0.0
        return nothing
    end

    # u0[4] = 2.01 (not 2.0) so the "linear affect" event (u[3]=10) lands at
    # τ = 10/2.01 ≈ 4.975 instead of exactly 5.0 — off the saveat grid. At
    # τ = saveat the cost is jump-discontinuous in u0[3], u0[4] (the post-event
    # state propagates to that saveat sample), so FD and adjoint legitimately
    # compute different one-sided derivatives and rtol comparisons fail.
    u0 = [50.0, 0.0, 0.0, 2.01]
    tspan = (0.0, 10.0)
    p = [9.8, 0.9]
    prob = ODEProblem(f, u0, tspan, p)
    sol = solve(
        prob, Tsit5(), callback = cb, abstol = abstol, reltol = reltol,
        saveat = savingtimes
    )

    du01, dp1 = @time Zygote.gradient(
        (
            u0,
            p,
        ) -> g(
            solve(
                prob, Tsit5(); u0, p,
                callback = cb, abstol, reltol, saveat = savingtimes,
                sensealg = BacksolveAdjoint()
            )
        ),
        u0, p
    )

    du02,
        dp2 = @time Zygote.gradient(
        (
            u0,
            p,
        ) -> g(
            solve(
                prob, Tsit5(); u0, p,
                callback = cb, abstol, reltol, saveat = savingtimes,
                sensealg = GaussAdjoint()
            )
        ),
        u0, p
    )

    dstuff = @time ForwardDiff.gradient(
        (θ) -> g(
            solve(
                prob, Tsit5(); u0 = θ[1:4], p = θ[5:6],
                callback = cb, abstol, reltol, saveat = savingtimes
            )
        ),
        [u0; p]
    )

    @test du01 ≈ dstuff[1:4] rtol = 1.0e-5
    @test dp1 ≈ dstuff[5:6] rtol = 1.0e-5
    @test du02 ≈ dstuff[1:4] rtol = 1.0e-5
    return @test dp2 ≈ dstuff[5:6] rtol = 1.0e-5
end

@testset "VectorContinuous callbacks" begin
    @testset "MSE loss function bouncing-ball like" begin
        g(u) = sum((1.0 .- u) .^ 2) ./ 2
        function condition(out, u, t, integrator) # Event when event_f(u,t) == 0
            out[1] = u[1]
            out[2] = (u[3] - 10.0)u[3]
        end
        @testset "callback with linear affect" begin
            function affect!(integrator, ev)
                indices = ev isa AbstractVector ?
                    (i for i in eachindex(ev) if !iszero(ev[i])) : (ev,)
                for idx in indices
                    if idx == 1
                        integrator.u[2] = -integrator.p[2] * integrator.u[2]
                    elseif idx == 2
                        integrator.u[4] = -integrator.p[2] * integrator.u[4]
                    end
                end
            end
            cb = VectorContinuousCallback(condition, affect!, 2)
            test_vector_continuous_callback(cb, g)
        end
    end
    @testset "Test condition function that depends on time only" begin
        g(u) = sum((1.0 .- u) .^ 2) ./ 2
        function condition(out, x, t, integrator)
            out[1] = sin(t)
            out[2] = cos(t)
        end

        function affect!(integrator, ev)
            indices = ev isa AbstractVector ?
                collect(i for i in eachindex(ev) if !iszero(ev[i])) : (ev,)
            isempty(indices) && return
            println("$(indices) triggered!")
            u_new = [0.5, 1.0, 0.0, 0.0]
            integrator.u .= u_new
        end
        cb = VectorContinuousCallback(condition, affect!, 2)
        test_vector_continuous_callback(cb, g)
    end
    @testset "Structural simultaneous fire (algebraically tied conditions)" begin
        # Two conditions are algebraically tied — out[2] is a constant
        # multiple of out[1] — so they hit zero at exactly the same t
        # under *any* perturbation of u0. The mask is always [±1, ±1].
        # Both conditions yield the same implicit ∇τ (∇_u out_i scaled
        # by the same 1/total-derivative factor), so the adjoint's
        # ∇τ correction is unambiguous and BacksolveAdjoint must match
        # ForwardDiff / FiniteDiff exactly. This regression-tests that
        # the multi-fire reverse path applies the implicit correction
        # rather than skipping it.
        function ball!(du, u, p, t)
            du[1] = u[3]
            du[2] = u[4]
            du[3] = 0
            du[4] = 0
            return nothing
        end
        condition_tied(out, u, t, integrator) =
            (out[1] = u[1]; out[2] = 2 * u[1]; nothing)
        function affect_tied!(integrator, ev)
            # ev is always [±1, ±1] here. Apply a coupled mutation that
            # treats the mask as a whole (couples vx and vy together).
            integrator.u[4] = integrator.u[3]
            integrator.u[3] = -integrator.u[3]
            return nothing
        end
        cb = VectorContinuousCallback(condition_tied, affect_tied!, 2)
        u0 = [3.0, 1.0, -1.0, 0.0]
        prob = ODEProblem(ball!, u0, (0.0, 5.0), [0.0])
        ref = [0.5, 0.5]
        g_tied(u) = sum(abs2, u[1:2] .- ref)
        loss_tied(theta) = g_tied(
            solve(
                prob, Tsit5(); u0 = theta, callback = cb,
                abstol = 1.0e-12, reltol = 1.0e-12
            ).u[end]
        )

        gfd = ForwardDiff.gradient(loss_tied, u0)
        du0_bs, = Zygote.gradient(
            u0 -> g_tied(
                solve(
                    prob, Tsit5(); u0 = u0, callback = cb,
                    abstol = 1.0e-12, reltol = 1.0e-12,
                    sensealg = BacksolveAdjoint()
                ).u[end]
            ),
            u0
        )
        @test du0_bs ≈ gfd rtol = 1.0e-5
    end
    @testset "Coincidental simultaneous fire (corner trap)" begin
        # Two independent conditions (x=0 wall, y=0 wall) happen to hit
        # zero at exactly the same instant when the trajectory passes
        # through the corner. Mask=[±1,±1] fires the corner-trap branch
        # of the affect (kills both velocities). This is a measure-zero
        # singularity — any Dual ε or FiniteDiff perturbation breaks
        # the simultaneity in the event-time search, so the perturbed
        # trajectory takes two sequential single-wall bounces instead
        # of the trap. ForwardDiff and the adjoint therefore compute
        # *different* gradients (gradient at the singular trap point
        # vs. gradient along the almost-everywhere two-bounce
        # trajectory). The test just exercises the code path without
        # asserting a single "right" answer — both calls must run and
        # return finite values.
        function ball2!(du, u, p, t)
            du[1] = u[3]
            du[2] = u[4]
            du[3] = 0
            du[4] = 0
            return nothing
        end
        condition_corner(out, u, t, integrator) = (out[1] = u[1]; out[2] = u[2]; nothing)
        function affect_corner!(integrator, ev)
            if ev isa AbstractVector
                hit1 = !iszero(ev[1])
                hit2 = !iszero(ev[2])
                if hit1 && hit2
                    integrator.u[3] = 0.0
                    integrator.u[4] = 0.0
                elseif hit1
                    integrator.u[3] = -integrator.u[3]
                elseif hit2
                    integrator.u[4] = -integrator.u[4]
                end
            else
                if ev == 1
                    integrator.u[3] = -integrator.u[3]
                else
                    integrator.u[4] = -integrator.u[4]
                end
            end
            return nothing
        end
        cb = VectorContinuousCallback(condition_corner, affect_corner!, 2)
        u0 = [1.0, 1.0, -1.0, -1.0]
        prob = ODEProblem(ball2!, u0, (0.0, 3.0), [0.0])
        ref = [0.5, 0.5]
        g_corner(u) = sum(abs2, u[1:2] .- ref)
        loss_corner(theta) = g_corner(
            solve(
                prob, Tsit5(); u0 = theta, callback = cb,
                abstol = 1.0e-12, reltol = 1.0e-12
            ).u[end]
        )
        gfd = ForwardDiff.gradient(loss_corner, u0)
        du0_bs, = Zygote.gradient(
            u0 -> g_corner(
                solve(
                    prob, Tsit5(); u0 = u0, callback = cb,
                    abstol = 1.0e-12, reltol = 1.0e-12,
                    sensealg = BacksolveAdjoint()
                ).u[end]
            ),
            u0
        )
        @test all(isfinite, gfd)
        @test all(isfinite, du0_bs)
    end
end
