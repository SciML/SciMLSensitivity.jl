using SciMLSensitivity, OrdinaryDiffEqLowOrderRK, Zygote, Test

@testset "Checkpoint regressions" begin
    @testset "Checkpoint forward timestep" begin
        f!(du, u, p, t) = (du .= -p[1] .* u; nothing)
        prob = ODEProblem(f!, [1.0, 2.0], (0.0, 0.2), [1.0])
        for sensealg in (InterpolatingAdjoint(autojacvec = false), GaussAdjoint(autojacvec = false))
            @testset "$(nameof(typeof(sensealg)))" begin
                loss(u0) = sum(solve(prob, Euler(); u0, dt = 0.01, adaptive = false, save_everystep = false, sensealg).u[end])
                @test Zygote.gradient(loss, prob.u0)[1] ≈ fill((1 - 0.01)^20, 2) rtol = 1.0e-12
            end
        end
    end

    @testset "Checkpoint stage extrapolation" begin
        f!(du, u, p, t) = (du .= -p[1]; nothing)
        prob = ODEProblem(f!, [1.0, 2.0], (0.0, 0.2), [1.0]; dt = 0.03, adaptive = false)
        for sensealg in (InterpolatingAdjoint(autojacvec = false), GaussAdjoint(autojacvec = false)),
                checkpoints in ([0.0, 0.2], [0.0, 0.1, 0.2])
            @testset "$(nameof(typeof(sensealg))), $checkpoints" begin
                sol = solve(prob, RK4(); saveat = checkpoints, dense = false)
                cost(u, p, t) = sum(u)
                if sensealg isa InterpolatingAdjoint
                    adj_prob = ODEAdjointProblem(sol, sensealg, RK4(), nothing, nothing, nothing, nothing, nothing, cost; checkpoints, abstol = 1.0e-12, reltol = 1.0e-12)
                else
                    integrand = SciMLSensitivity.GaussIntegrand(sol, sensealg, checkpoints)
                    adj_prob, _, _ = ODEAdjointProblem(sol, sensealg, RK4(), integrand, nothing, nothing, nothing, nothing, nothing, nothing, cost; checkpoints, abstol = 1.0e-12, reltol = 1.0e-12)
                end
                sense = adj_prob.f.f
                u = zero(adj_prob.u0)
                du = similar(u)
                @test sense.checkpoint_sol.cpsol.t[2] - sense.checkpoint_sol.cpsol.t[1] ≈ 0.03 rtol = 1.0e-12
                if length(checkpoints) > 2
                    SciMLSensitivity.split_states(du, u, 0.05, sense)
                    @test sense.checkpoint_sol.cpsol.t[2] - sense.checkpoint_sol.cpsol.t[1] ≈ 0.03 rtol = 1.0e-12
                end
                for t in (0.21, 0.22, -0.01, -0.02, 0.15, 0.21)
                    cp = sense.checkpoint_sol
                    previous = cp.cpsol
                    previous_cursor = cp.cursor
                    y = SciMLSensitivity.split_states(du, u, t, sense)[3]
                    expected_cursor = t < 0 ? 1 : length(checkpoints) - 1
                    @test cp.cursor == expected_cursor
                    @test y ≈ prob.u0 .- t rtol = 1.0e-12
                    if previous_cursor == expected_cursor
                        @test cp.cpsol === previous
                    end
                    if sensealg isa GaussAdjoint
                        @test sense.GaussInt.sol === cp.cpsol
                    end
                end
                if sensealg isa GaussAdjoint
                    for t in (-0.01, 0.21, -0.02, 0.15)
                        y = SciMLSensitivity.split_states(u, t, sense)[3]
                        @test y ≈ prob.u0 .- t rtol = 1.0e-12
                        @test sense.GaussInt.sol === sense.checkpoint_sol.cpsol
                    end
                end
            end
        end
    end

end
