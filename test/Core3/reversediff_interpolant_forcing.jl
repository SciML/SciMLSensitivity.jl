using SciMLSensitivity, OrdinaryDiffEq, LinearAlgebra, Test

# Regression test for https://github.com/SciML/SciMLSensitivity.jl/issues/1649
# RHS parameter dependence comes from the dense interpolant of another
# ODESolution, `ref(t)`. A cached ReverseDiff tape traced once at `tspan[2]`
# freezes `t`-dependent control flow inside the interpolant, giving a ~5% wrong
# `df/dp` in QuadratureAdjoint/GaussAdjoint. InterpolatingAdjoint re-traces its
# tape per evaluation and was already correct.
@testset "ReverseDiffVJP with solution-interpolant forcing (#1649)" begin
    A = [-0.3 1.0; -1.0 -0.3]
    u0 = [1.0, 0.5]
    tspan = (0.0, 3.0)
    v = [1.0, 0.0]

    base!(du, u, p, t) = (mul!(du, A, u); nothing)
    ref = solve(
        ODEProblem(base!, u0, tspan), Tsit5();
        abstol = 1e-10, reltol = 1e-10, dense = true, save_everystep = true
    )

    function forced!(du, u, eps, t)
        mul!(du, A, u)
        r = ref(t)
        du[1] += eps[1] * r[1]
        du[2] += eps[1] * r[2]
        return nothing
    end

    Gof(e) = begin
        s = solve(
            ODEProblem(forced!, u0, tspan, [e]), Tsit5();
            abstol = 1e-11, reltol = 1e-11
        )
        dot(v, s.u[end])
    end
    fd = (Gof(1e-6) - Gof(-1e-6)) / 2e-6

    sol = solve(
        ODEProblem(forced!, u0, tspan, [0.0]), Tsit5();
        abstol = 1e-10, reltol = 1e-10, dense = true, save_everystep = true
    )
    dgdu = (out, u, p, t, i) -> (copyto!(out, v); nothing)

    for (name, sa) in [
            ("InterpolatingAdjoint", InterpolatingAdjoint(autojacvec = ReverseDiffVJP())),
            ("QuadratureAdjoint", QuadratureAdjoint(autojacvec = ReverseDiffVJP())),
            ("GaussAdjoint", GaussAdjoint(autojacvec = ReverseDiffVJP())),
            ("QuadratureAdjoint(false)", QuadratureAdjoint(autojacvec = false)),
        ]
        _, dp = adjoint_sensitivities(
            sol, Tsit5(); sensealg = sa, t = [tspan[2]],
            dgdu_discrete = dgdu, abstol = 1e-10, reltol = 1e-10
        )
        @test only(dp) ≈ fd rtol = 1e-5
    end
end
