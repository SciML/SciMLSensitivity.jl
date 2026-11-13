using ForwardDiff, OrdinaryDiffEqLowOrderRK, SciMLSensitivity, Test

intervals = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
tag = ForwardDiff.Tag{typeof(intervals), Float64}

@test SciMLSensitivity.Gaussfindcursor(intervals, 3.0) == 3
@test SciMLSensitivity.Gaussfindcursor(
    intervals, ForwardDiff.Dual{tag}(3.0, 1.0),
) == 3
@test SciMLSensitivity.Gaussfindcursor(
    intervals, ForwardDiff.Dual{tag}(3.0, -1.0),
) == 3
@test SciMLSensitivity.Gaussfindcursor(
    intervals, ForwardDiff.Dual{tag}(2.5, 1.0),
) == 3
@test SciMLSensitivity.Gaussfindcursor(
    intervals, ForwardDiff.Dual{tag}(0.0, -1.0),
) == 1

@testset "One-point Gauss checkpoint" begin
    f(du, u, p, t) = (du .= p .* u)
    prob = ODEProblem(f, [1.0], (0.0, 1.0), [0.5])
    sol = solve(prob, DP5(); abstol = 1.0e-10, reltol = 1.0e-10)
    checkpoints = [0.0, 0.5, 1.0]
    sensealg = GaussAdjoint(checkpointing = true, autojacvec = false)
    dgdu(out, u, p, t, i) = fill!(out, 1.0)
    @testset "inplace=$inplace" for inplace in (true, false)
        integrand = SciMLSensitivity.GaussIntegrand(sol, sensealg, checkpoints, nothing)
        adj_prob = ODEAdjointProblem(
            sol, sensealg, DP5(), integrand, nothing,
            [1.0], dgdu; checkpoints, abstol = 1.0e-10, reltol = 1.0e-10
        )[1]
        S = adj_prob.f.f
        S.checkpoint_sol.cpsol = solve(
            remake(prob; tspan = (0.5, 1.0), u0 = sol(0.5)), DP5();
            save_everystep = false, save_end = false
        )
        @test length(S.checkpoint_sol.cpsol.t) == 1
        y = if inplace
            SciMLSensitivity.split_states(zeros(1), zeros(1), 0.25, S)[3]
        else
            SciMLSensitivity.split_states(zeros(1), 0.25, S)[3]
        end
        @test only(y) ≈ exp(0.125) rtol = 1.0e-8
        @test S.checkpoint_sol.cursor == 1
        @test S.GaussInt.sol === S.checkpoint_sol.cpsol
    end
end
