using SciMLSensitivity, OrdinaryDiffEq, Zygote
using DiffEqCallbacks
using Test

abstol = 1e-12
reltol = 1e-12

@testset "Non-tracked callbacks" begin
    function f(du, u, p, t)
        du[1] = dx = p[1] * u[1] - p[2] * u[1] * u[2] * t
        du[2] = dy = -p[3] * u[2] + t * p[4] * u[1] * u[2]
    end

    p = [1.5, 1.0, 3.0, 1.0]
    u0 = [1.0; 1.0]
    prob = ODEProblem(f, u0, (0.0, 10.0), p)

    sol = solve(prob, Tsit5(), abstol = abstol, reltol = reltol)

    # Do a discrete adjoint problem
    t = 0.0:0.5:10.0
    # g(t,u,i) = (1-u)^2/2, L2 away from 1
    function dg(out, u, p, t, i)
        (out .= -2.0 .+ u)
    end

    saved_values = SavedValues(Float64, Vector{Float64})
    cb = SavingCallback((u, t, integrator) -> copy(u[(end - 1):end]), saved_values)

    _, res = adjoint_sensitivities(sol, Tsit5(), sensealg = BacksolveAdjoint(), t = t,
        dgdu_discrete = dg, callback = cb)
    _, res2 = adjoint_sensitivities(sol, Tsit5(), sensealg = BacksolveAdjoint(), t = t,
        dgdu_discrete = dg)

    @test res≈res2 rtol=1e-10
    @test sol(saved_values.t).u≈saved_values.saveval rtol=1e-10
end
