using OrdinaryDiffEq, SciMLSensitivity, ForwardDiff, Zygote, SciMLBase, Test
using OrdinaryDiffEqBDF: DFBDF, DImplicitEuler
using OrdinaryDiffEqNonlinearSolve: BrownFullBasicInit

# Fully implicit DAEProblem adjoints via the augmented adjoint DAE
# (Cao, Li, Petzold & Serban, SIAM Journal on Scientific Computing 24(3), 2003).

sumsq_dg(out, u, p, t, i) = (out .= u) # g_i = sum(u.^2)/2 at each data point

@testset "Index-1: fully implicit Robertson" begin
    function rober_dae!(res, du, u, p, t)
        y₁, y₂, y₃ = u
        k₁, k₂, k₃ = p
        res[1] = -k₁ * y₁ + k₃ * y₂ * y₃ - du[1]
        res[2] = k₁ * y₁ - k₂ * y₂^2 - k₃ * y₂ * y₃ - du[2]
        res[3] = y₁ + y₂ + y₃ - 1
        return nothing
    end

    u0 = [1.0, 0.0, 0.0]
    p = [0.04, 3.0e7, 1.0e4]
    du0 = [-0.04, 0.04, 0.0]
    tspan = (0.0, 10.0)
    prob = DAEProblem(
        rober_dae!, du0, u0, tspan, p, differential_vars = [true, true, false]
    )
    sol = solve(prob, DFBDF(), abstol = 1.0e-10, reltol = 1.0e-10)
    ts = collect(0.0:1.0:10.0)

    function G(p)
        du0_ = [
            -p[1] * u0[1] + p[3] * u0[2] * u0[3],
            p[1] * u0[1] - p[2] * u0[2]^2 - p[3] * u0[2] * u0[3], zero(eltype(p)),
        ]
        _prob = DAEProblem(
            rober_dae!, du0_, eltype(p).(u0), tspan, p,
            differential_vars = [true, true, false]
        )
        _sol = solve(_prob, DFBDF(), abstol = 1.0e-10, reltol = 1.0e-10, saveat = ts)
        sum(sum(abs2, u) / 2 for u in _sol.u)
    end
    fd_p = ForwardDiff.gradient(G, p)

    # consistent perturbations of the differential u0 components (u0[3] adjusts
    # through the algebraic constraint)
    function Gu(u0d)
        u0_ = [u0d[1], u0d[2], 1 - u0d[1] - u0d[2]]
        du0_ = [
            -p[1] * u0_[1] + p[3] * u0_[2] * u0_[3],
            p[1] * u0_[1] - p[2] * u0_[2]^2 - p[3] * u0_[2] * u0_[3],
            zero(eltype(u0d)),
        ]
        _prob = DAEProblem(
            rober_dae!, du0_, u0_, tspan, p,
            differential_vars = [true, true, false]
        )
        _sol = solve(_prob, DFBDF(), abstol = 1.0e-10, reltol = 1.0e-10, saveat = ts)
        sum(sum(abs2, u) / 2 for u in _sol.u)
    end
    fd_u = ForwardDiff.gradient(Gu, u0[1:2])

    # Both adjoint-problem forms — mass-matrix `ODEProblem` (ODE algorithm) and
    # fully implicit residual `DAEProblem` (DAE algorithm) — are emitted by the
    # shared `_dae_adjoint_problem`, so exercising both once (here, through
    # `InterpolatingAdjoint`) covers the form dispatch; the other methods are then
    # checked with the mass-matrix form only.
    for adjalg in (FBDF(), DFBDF())
        du0g, dpg = adjoint_sensitivities(
            sol, adjalg; t = ts, dgdu_discrete = sumsq_dg,
            sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP()),
            abstol = 1.0e-8, reltol = 1.0e-8
        )
        @test vec(dpg) ≈ fd_p rtol = 1.0e-4
        @test du0g[1:2] ≈ fd_u rtol = 1.0e-4
        @test abs(du0g[3]) < 1.0e-10
    end

    # every DAE-capable adjoint method (mass-matrix form)
    @testset "sensealg = $(nameof(typeof(sensealg)))" for sensealg in (
            QuadratureAdjoint(autojacvec = ReverseDiffVJP()),
            GaussAdjoint(autojacvec = ReverseDiffVJP()),
            GaussKronrodAdjoint(autojacvec = ReverseDiffVJP()),
        )
        du0g, dpg = adjoint_sensitivities(
            sol, FBDF(); t = ts, dgdu_discrete = sumsq_dg,
            sensealg, abstol = 1.0e-8, reltol = 1.0e-8
        )
        @test vec(dpg) ≈ fd_p rtol = 1.0e-4
        @test du0g[1:2] ≈ fd_u rtol = 1.0e-4
        @test abs(du0g[3]) < 1.0e-10
    end

    # EnzymeVJP through the interpolating adjoint
    du0e, dpe = adjoint_sensitivities(
        sol, FBDF(); t = ts, dgdu_discrete = sumsq_dg,
        sensealg = InterpolatingAdjoint(autojacvec = EnzymeVJP()),
        abstol = 1.0e-8, reltol = 1.0e-8
    )
    @test vec(dpe) ≈ fd_p rtol = 1.0e-4
    @test du0e[1:2] ≈ fd_u rtol = 1.0e-4

    # default vjp choice
    du0d, dpd = adjoint_sensitivities(
        sol, FBDF(); t = ts, dgdu_discrete = sumsq_dg,
        abstol = 1.0e-8, reltol = 1.0e-8
    )
    @test vec(dpd) ≈ fd_p rtol = 1.0e-4

    # Zygote through the solve interface, adjoint solved by the forward DAE alg
    loss(p) = sum(
        sum(abs2, u) / 2 for u in solve(
                prob, DFBDF(); p = p, saveat = ts,
                abstol = 1.0e-10, reltol = 1.0e-10,
                sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP())
            ).u
    )
    gz = Zygote.gradient(loss, p)[1]
    @test gz ≈ fd_p rtol = 1.0e-4

    lossu(u0v) = sum(
        sum(abs2, u) / 2 for u in solve(
                prob, DFBDF(); u0 = u0v, saveat = ts,
                abstol = 1.0e-10, reltol = 1.0e-10,
                sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP())
            ).u
    )
    gu = Zygote.gradient(lossu, u0)[1]
    @test gu[1:2] ≈ fd_u rtol = 1.0e-4
end

@testset "Index-1: parameter-dependent algebraic constraint" begin
    function rober2!(res, du, u, p, t)
        y₁, y₂, y₃ = u
        k₁, k₂, k₃, s = p
        res[1] = -k₁ * y₁ + k₃ * y₂ * y₃ - du[1]
        res[2] = k₁ * y₁ - k₂ * y₂^2 - k₃ * y₂ * y₃ - du[2]
        res[3] = y₁ + y₂ + y₃ - s
        return nothing
    end
    p = [0.04, 3.0e7, 1.0e4, 1.0]
    u0 = [1.0, 0.0, 0.0]
    du0 = [-0.04, 0.04, 0.0]
    tspan = (0.0, 10.0)
    prob = DAEProblem(rober2!, du0, u0, tspan, p, differential_vars = [true, true, false])
    sol = solve(prob, DFBDF(), abstol = 1.0e-10, reltol = 1.0e-10)
    ts = collect(0.0:2.0:10.0)

    function G2(p)
        u0_ = [one(eltype(p)), zero(eltype(p)), p[4] - 1]
        du0_ = [
            -p[1] * u0_[1] + p[3] * u0_[2] * u0_[3],
            p[1] * u0_[1] - p[2] * u0_[2]^2 - p[3] * u0_[2] * u0_[3], zero(eltype(p)),
        ]
        _prob = DAEProblem(
            rober2!, du0_, u0_, tspan, p, differential_vars = [true, true, false]
        )
        _sol = solve(_prob, DFBDF(), abstol = 1.0e-11, reltol = 1.0e-11, saveat = ts)
        sum(sum(abs2, u) / 2 for u in _sol.u)
    end
    fd_p = ForwardDiff.gradient(G2, p)

    # the Δλa' ∂F_alg/∂p jump correction flows through a different mechanism in
    # each method (state augmentation vs quadrature vs integrating callback)
    for _sensealg in (
            InterpolatingAdjoint(autojacvec = ReverseDiffVJP()),
            QuadratureAdjoint(autojacvec = ReverseDiffVJP()),
            GaussAdjoint(autojacvec = ReverseDiffVJP()),
            GaussKronrodAdjoint(autojacvec = ReverseDiffVJP()),
        )
        # 1e-10 adjoint tolerances: the quadrature methods integrate over the
        # interpolated algebraic λ block of the adjoint solution, whose
        # interpolation error dominates at looser tolerances (converges as the
        # adjoint tolerance tightens; verified 1.6e-4 → 5.4e-10 from 1e-8 → 1e-10).
        du0g, dpg = adjoint_sensitivities(
            sol, FBDF(); t = ts, dgdu_discrete = sumsq_dg,
            sensealg = _sensealg,
            abstol = 1.0e-10, reltol = 1.0e-10
        )
        @test vec(dpg) ≈ fd_p rtol = 1.0e-4
    end
end

@testset "Index-0: implicit-form ODE vs ODE adjoint" begin
    function lotka_dae!(res, du, u, p, t)
        res[1] = p[1] * u[1] - p[2] * u[1] * u[2] - du[1]
        res[2] = -p[3] * u[2] + p[4] * u[1] * u[2] - du[2]
        return nothing
    end
    function lotka!(du, u, p, t)
        du[1] = p[1] * u[1] - p[2] * u[1] * u[2]
        du[2] = -p[3] * u[2] + p[4] * u[1] * u[2]
        return nothing
    end
    plk = [1.5, 1.0, 3.0, 1.0]
    u0lk = [1.0, 1.0]
    du0lk = zeros(2)
    lotka!(du0lk, u0lk, plk, 0.0)
    problk = DAEProblem(lotka_dae!, du0lk, u0lk, (0.0, 5.0), plk)
    sollk = solve(problk, DFBDF(), abstol = 1.0e-10, reltol = 1.0e-10)
    probode = ODEProblem(lotka!, u0lk, (0.0, 5.0), plk)
    solode = solve(probode, Vern9(), abstol = 1.0e-12, reltol = 1.0e-12)
    tslk = collect(0.0:0.5:5.0)

    du0r, dpr = adjoint_sensitivities(
        solode, Vern9(); t = tslk, dgdu_discrete = sumsq_dg,
        abstol = 1.0e-10, reltol = 1.0e-10
    )

    for vjp in (ReverseDiffVJP(), ReverseDiffVJP(true), false)
        du0a, dpa = adjoint_sensitivities(
            sollk, FBDF(); t = tslk, dgdu_discrete = sumsq_dg,
            sensealg = InterpolatingAdjoint(autojacvec = vjp),
            abstol = 1.0e-10, reltol = 1.0e-10
        )
        @test du0a ≈ du0r rtol = 1.0e-5
        @test dpa ≈ dpr rtol = 1.0e-5
    end

    # continuous cost, both by explicit dgdu and by AD of g
    gcont(u, p, t) = sum(abs2, u) / 2
    dgcont(out, u, p, t) = (out .= u)
    du0cr, dpcr = adjoint_sensitivities(
        solode, Vern9(); dgdu_continuous = dgcont,
        abstol = 1.0e-10, reltol = 1.0e-10
    )
    du0c, dpc = adjoint_sensitivities(
        sollk, FBDF(); dgdu_continuous = dgcont,
        sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP()),
        abstol = 1.0e-10, reltol = 1.0e-10
    )
    @test du0c ≈ du0cr rtol = 1.0e-4
    @test dpc ≈ dpcr rtol = 1.0e-4
    du0c2, dpc2 = adjoint_sensitivities(
        sollk, FBDF(); g = gcont,
        sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP()),
        abstol = 1.0e-10, reltol = 1.0e-10
    )
    @test dpc2 ≈ dpcr rtol = 1.0e-4

    # out-of-place residual with ZygoteVJP
    lotka_dae(du, u, p, t) = [
        p[1] * u[1] - p[2] * u[1] * u[2] - du[1],
        -p[3] * u[2] + p[4] * u[1] * u[2] - du[2],
    ]
    proboop = DAEProblem(lotka_dae, du0lk, u0lk, (0.0, 5.0), plk)
    soloop = solve(proboop, DFBDF(), abstol = 1.0e-10, reltol = 1.0e-10)
    du0z, dpz = adjoint_sensitivities(
        soloop, FBDF(); t = tslk, dgdu_discrete = sumsq_dg,
        sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP()),
        abstol = 1.0e-10, reltol = 1.0e-10
    )
    @test dpz ≈ dpr rtol = 1.0e-5
end

@testset "Index-2 Hessenberg with p-dependent constraint" begin
    #   ẏ1 = p1*y2 + z,   ẏ2 = -p2*y1,   0 = y1 - sin(p3*t)
    # Analytic: y1 = sin(p3 t), y2 = y20 + p2 (cos(p3 t) - 1)/p3
    function hess2!(res, du, u, p, t)
        y1, y2, z = u
        res[1] = p[1] * y2 + z - du[1]
        res[2] = -p[2] * y1 - du[2]
        res[3] = y1 - sin(p[3] * t)
        return nothing
    end

    p = [1.0, 2.0, 3.0]
    y20 = 1.0
    u0 = [0.0, y20, p[3] - p[1] * y20]
    du0 = [p[3], 0.0, 0.0]
    tspan = (0.0, 1.0)
    prob = DAEProblem(hess2!, du0, u0, tspan, p, differential_vars = [true, true, false])
    # adaptive BDF error control fails on index-2 systems; fixed-step implicit
    # Euler converges (first order), so the comparison tolerances below reflect
    # the O(dt) forward error.
    sol = solve(
        prob, DImplicitEuler(), adaptive = false, dt = 2.0e-5,
        initializealg = SciMLBase.NoInit()
    )
    @test sol.retcode == ReturnCode.Success

    ts = collect(0.2:0.2:1.0)
    dg2(out, u, p, t, i) = (out .= [u[1], u[2], 0.0])

    y1e(t) = sin(p[3] * t)
    y2e(t) = y20 + p[2] * (cos(p[3] * t) - 1) / p[3]
    dG_dy20 = sum(y2e(ti) for ti in ts)
    dG_dp2 = sum(y2e(ti) * (cos(p[3] * ti) - 1) / p[3] for ti in ts)
    dy2_dp3(ti) = p[2] * (-ti * sin(p[3] * ti) * p[3] - (cos(p[3] * ti) - 1)) / p[3]^2
    dG_dp3 = sum(y1e(ti) * ti * cos(p[3] * ti) + y2e(ti) * dy2_dp3(ti) for ti in ts)

    du0g, dpg = adjoint_sensitivities(
        sol, FBDF(); t = ts, dgdu_discrete = dg2,
        sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP()),
        abstol = 1.0e-8, reltol = 1.0e-8
    )
    @test du0g[2] ≈ dG_dy20 rtol = 1.0e-3
    @test du0g[3] == 0
    @test abs(vec(dpg)[1]) < 1.0e-6
    @test vec(dpg)[2] ≈ dG_dp2 rtol = 1.0e-3
    @test vec(dpg)[3] ≈ dG_dp3 rtol = 1.0e-3

    # The quadrature-style methods cannot capture the impulsive index-2 adjoint
    # multiplier at the cost times (its contribution sits at the quadrature
    # interval endpoints), so they reject index-2 DAEs.
    @test_throws ErrorException adjoint_sensitivities(
        sol, FBDF(); t = ts, dgdu_discrete = dg2,
        sensealg = QuadratureAdjoint(autojacvec = ReverseDiffVJP()),
        abstol = 1.0e-8, reltol = 1.0e-8
    )
    @test_throws ErrorException adjoint_sensitivities(
        sol, FBDF(); t = ts, dgdu_discrete = dg2,
        sensealg = GaussAdjoint(autojacvec = ReverseDiffVJP()),
        abstol = 1.0e-8, reltol = 1.0e-8
    )
end

@testset "informative errors" begin
    function rober_dae!(res, du, u, p, t)
        res[1] = -p[1] * u[1] + p[3] * u[2] * u[3] - du[1]
        res[2] = p[1] * u[1] - p[2] * u[2]^2 - p[3] * u[2] * u[3] - du[2]
        res[3] = u[1] + u[2] + u[3] - 1
        return nothing
    end
    prob = DAEProblem(
        rober_dae!, [-0.04, 0.04, 0.0], [1.0, 0.0, 0.0], (0.0, 1.0),
        [0.04, 3.0e7, 1.0e4], differential_vars = [true, true, false]
    )
    sol = solve(prob, DFBDF(), abstol = 1.0e-8, reltol = 1.0e-8)
    @test_throws ErrorException adjoint_sensitivities(
        sol, FBDF(); t = [0.5, 1.0], dgdu_discrete = sumsq_dg,
        sensealg = BacksolveAdjoint(autojacvec = ReverseDiffVJP())
    )
    @test_throws ErrorException adjoint_sensitivities(
        sol, FBDF(); t = [0.5, 1.0], dgdu_discrete = sumsq_dg,
        sensealg = InterpolatingAdjoint(
            autojacvec = ReverseDiffVJP(), checkpointing = true
        )
    )
end

@testset "no algorithm: adjoint in residual form" begin
    # The heat equation on a grid with the Dirichlet values as algebraic rows, the layout a PDE
    # discretization produces. Solved without an algorithm, the adjoint must stay in residual
    # form: its mass-matrix form under the default ODE algorithm does not finish at this size.
    n = 26
    h = 1 / (n - 1)
    function heat_dae!(res, du, u, p, t)
        a = p[1] + p[2]
        res[1] = u[1] - exp(-t)
        for i in 2:(n - 1)
            res[i] = a * (u[i - 1] - 2u[i] + u[i + 1]) / h^2 - du[i]
        end
        res[n] = u[n] - exp(-t) * cos(1)
        return nothing
    end
    p = [1.2, 2.1]
    u0 = cos.(range(0, 1; length = n))
    prob = DAEProblem(
        heat_dae!, zeros(n), u0, (0.0, 1.0), p;
        differential_vars = [false; trues(n - 2); false]
    )
    kw = (; abstol = 1.0e-8, reltol = 1.0e-8, initializealg = BrownFullBasicInit())
    sol = solve(prob; kw...)
    @test sol.alg isa DFBDF
    ts = collect(0.0:0.1:1.0)
    sensealg = InterpolatingAdjoint(autojacvec = false)
    @test SciMLSensitivity.DAEAdjointProblem(sol, sensealg, nothing, ts, sumsq_dg) isa DAEProblem
    dp_default = adjoint_sensitivities(
        sol, nothing; t = ts, dgdu_discrete = sumsq_dg, sensealg, abstol = 1.0e-8, reltol = 1.0e-8
    )[2]
    dp_dfbdf = adjoint_sensitivities(
        sol, DFBDF(); t = ts, dgdu_discrete = sumsq_dg, sensealg, abstol = 1.0e-8, reltol = 1.0e-8
    )[2]
    @test dp_default ≈ dp_dfbdf rtol = 1.0e-6
    loss(p) = sum(sum(abs2, u) / 2 for u in solve(remake(prob; p); saveat = ts, sensealg, kw...).u)
    @test Zygote.gradient(loss, p)[1] ≈ ForwardDiff.gradient(loss, p) rtol = 1.0e-5
end
