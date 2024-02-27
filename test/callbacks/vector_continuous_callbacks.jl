using OrdinaryDiffEq, Zygote
using SciMLSensitivity, Test, ForwardDiff

abstol = 1e-12
reltol = 1e-12
savingtimes = 0.5

# see https://docs.sciml.ai/DiffEqDocs/stable/features/callback_functions/#VectorContinuousCallback-Example
function test_vector_continuous_callback(cb, g)
    function f(du, u, p, t)
        du[1] = u[2]
        du[2] = -p[1]
        du[3] = u[4]
        du[4] = 0.0
    end

    u0 = [50.0, 0.0, 0.0, 2.0]
    tspan = (0.0, 10.0)
    p = [9.8, 0.9]
    prob = ODEProblem(f, u0, tspan, p)
    sol = solve(prob, Tsit5(), callback = cb, abstol = abstol, reltol = reltol,
        saveat = savingtimes)

    du01, dp1 = @time Zygote.gradient(
        (u0, p) -> g(solve(prob, Tsit5(), u0 = u0, p = p,
            callback = cb, abstol = abstol,
            reltol = reltol,
            saveat = savingtimes,
            sensealg = BacksolveAdjoint())),
        u0, p)

    dstuff = @time ForwardDiff.gradient(
        (θ) -> g(solve(prob, Tsit5(), u0 = θ[1:4],
            p = θ[5:6], callback = cb,
            abstol = abstol, reltol = reltol,
            saveat = savingtimes)),
        [u0; p])

    @test du01 ≈ dstuff[1:4]
    @test dp1 ≈ dstuff[5:6]
end

@testset "VectorContinuous callbacks" begin
    @testset "MSE loss function bouncing-ball like" begin
        g(u) = sum((1.0 .- u) .^ 2) ./ 2
        function condition(out, u, t, integrator) # Event when event_f(u,t) == 0
            out[1] = u[1]
            out[2] = (u[3] - 10.0)u[3]
        end
        @testset "callback with linear affect" begin
            function affect!(integrator, idx)
                if idx == 1
                    integrator.u[2] = -integrator.p[2] * integrator.u[2]
                elseif idx == 2
                    integrator.u[4] = -integrator.p[2] * integrator.u[4]
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

        function affect!(integrator, idx)
            println("$(idx) triggered!")
            u_new = [0.5, 1.0, 0.0, 0.0]
            integrator.u .= u_new
        end
        cb = VectorContinuousCallback(condition, affect!, 2)
        test_vector_continuous_callback(cb, g)
    end
end
