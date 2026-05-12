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

    u0 = [50.0, 0.0, 0.0, 2.0]
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
    @testset "Simultaneous fire (corner trap)" begin
        # Two walls at x=0 and y=0. Mask interpretation:
        #   [±1, 0]: bounce off x-wall   (flip vx)
        #   [0, ±1]: bounce off y-wall   (flip vy)
        #   [±1, ±1]: hit the corner     (both velocities → 0, trap)
        # The trap is fundamentally not decomposable into two single-wall
        # bounces — different mask combinations do different things — so
        # the adjoint must use the full event_idxs mask in the affect VJP.
        function ball!(du, u, p, t)
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
        prob = ODEProblem(ball!, u0, (0.0, 3.0), [0.0])
        ref = [0.5, 0.5]
        g_corner(u) = sum(abs2, u[1:2] .- ref)
        loss_corner(theta) = g_corner(
            solve(
                prob, Tsit5(); u0 = theta, callback = cb,
                abstol = 1.0e-12, reltol = 1.0e-12
            ).u[end]
        )

        # Trap-aware analytical gradient: at t=1 the forward solve fires
        # mask=[-1,-1], affect sets u[3]=u[4]=0 (velocities killed), and
        # the ball sits at the origin for t > 1. Backward through that:
        # the affect's VJP zeros velocity adjoints, integration to t=0
        # accumulates +1 in each velocity slot; dL/du[1:2] = 2*(0-0.5) = -1.
        analytical = [-1.0, -1.0, -1.0, -1.0]
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
        @test du0_bs ≈ analytical rtol = 1.0e-8

        # ForwardDiff sees the perturbed trajectory: a Dual ε perturbation
        # in any component breaks the simultaneity in the event-time
        # search, so the perturbed ball does TWO sequential single-wall
        # bounces instead of the corner trap. Hence FD differs from the
        # adjoint at this measure-zero singularity. It still has to
        # *run* without erroring, which is what this exercises.
        gfd = ForwardDiff.gradient(loss_corner, u0)
        @test all(isfinite, gfd)
    end
end
