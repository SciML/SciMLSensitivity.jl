using Test, LinearAlgebra
using SciMLSensitivity, StochasticDiffEq
using DiffEqNoiseProcess
using FiniteDiff, ForwardDiff, Zygote
using Random

@info "SDE Non-Diagonal Noise Adjoints"

seed = 100
Random.seed!(seed)

tstart = 0.0
tend = 0.1
dt = 0.005
trange = (tstart, tend)
t = tstart:dt:tend
tarray = collect(t)

function g(u, p, t)
    sum(u .^ 2.0 / 2.0)
end

function dg!(out, u, p, t, i)
    (out .= u)
end

# non-diagonal noise
@testset "Non-diagonal noise tests" begin
    Random.seed!(seed)

    u₀ = [0.75, 0.5]
    p = [-1.5, 0.05, 0.2, 0.01]

    dtnd = tend / 1e3

    # Example from Roessler, SIAM J. NUMER. ANAL, 48, 922–952 with d = 2; m = 2
    function f_nondiag!(du, u, p, t)
        du[1] = p[1] * u[1] + p[2] * u[2]
        du[2] = p[2] * u[1] + p[1] * u[2]
        nothing
    end

    function g_nondiag!(du, u, p, t)
        du[1, 1] = p[3] * u[1] + p[4] * u[2]
        du[1, 2] = p[3] * u[1] + p[4] * u[2]
        du[2, 1] = p[4] * u[1] + p[3] * u[2]
        du[2, 2] = p[4] * u[1] + p[3] * u[2]
        nothing
    end

    function f_nondiag(u, p, t)
        dx = p[1] * u[1] + p[2] * u[2]
        dy = p[2] * u[1] + p[1] * u[2]
        [dx, dy]
    end

    function g_nondiag(u, p, t)
        du11 = p[3] * u[1] + p[4] * u[2]
        du12 = p[3] * u[1] + p[4] * u[2]
        du21 = p[4] * u[1] + p[3] * u[2]
        du22 = p[4] * u[1] + p[3] * u[2]

        [du11 du12
         du21 du22]
    end

    function f_nondiag_analytic(u0, p, t, W)
        A = [[p[1], p[2]] [p[2], p[1]]]
        B = [[p[3], p[4]] [p[4], p[3]]]
        tmp = A * t + B * W[1] + B * W[2]
        exp(tmp) * u0
    end

    noise_matrix = similar(p, 2, 2)
    noise_matrix .= false

    Random.seed!(seed)
    prob = SDEProblem(f_nondiag!, g_nondiag!, u₀, trange, p,
        noise_rate_prototype = noise_matrix)
    sol = solve(prob, EulerHeun(), dt = dtnd, save_noise = true)

    noise_matrix = similar(p, 2, 2)
    noise_matrix .= false
    Random.seed!(seed)
    proboop = SDEProblem(f_nondiag, g_nondiag, u₀, trange, p,
        noise_rate_prototype = noise_matrix)
    soloop = solve(proboop, EulerHeun(), dt = dtnd, save_noise = true)

    res_sde_u0, res_sde_p = adjoint_sensitivities(soloop, EulerHeun(), t = Array(t),
        dgdu_discrete = dg!,
        dt = dtnd, adaptive = false,
        sensealg = BacksolveAdjoint())

    @info res_sde_p

    res_sde_u0a, res_sde_pa = adjoint_sensitivities(soloop, EulerHeun(), t = Array(t),
        dgdu_discrete = dg!,
        dt = dtnd, adaptive = false,
        sensealg = BacksolveAdjoint(autojacvec = ReverseDiffVJP()))

    @test isapprox(res_sde_u0a, res_sde_u0, rtol = 1e-6)
    @test isapprox(res_sde_pa, res_sde_p, rtol = 1e-6)

    @info res_sde_pa

    res_sde_u0a, res_sde_pa = adjoint_sensitivities(soloop, EulerHeun(), t = Array(t),
        dgdu_discrete = dg!,
        dt = dtnd, adaptive = false,
        sensealg = BacksolveAdjoint(autojacvec = false))

    @test isapprox(res_sde_u0a, res_sde_u0, rtol = 1e-6)
    @test isapprox(res_sde_pa, res_sde_p, rtol = 1e-6)

    @info res_sde_pa

    res_sde_u0a, res_sde_pa = adjoint_sensitivities(soloop, EulerHeun(), t = Array(t),
        dgdu_discrete = dg!,
        dt = dtnd, adaptive = false,
        sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP()))

    @test isapprox(res_sde_u0a, res_sde_u0, rtol = 1e-5)
    @test isapprox(res_sde_pa, res_sde_p, rtol = 1e-4)

    @info res_sde_pa

    res_sde_u0a, res_sde_pa = adjoint_sensitivities(soloop, EulerHeun(), t = Array(t),
        dgdu_discrete = dg!,
        dt = dtnd, adaptive = false,
        sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP()))

    @test isapprox(res_sde_u0a, res_sde_u0, rtol = 1e-5)
    @test isapprox(res_sde_pa, res_sde_p, rtol = 1e-4)

    @info res_sde_pa

    res_sde_u0a, res_sde_pa = adjoint_sensitivities(soloop, EulerHeun(), t = Array(t),
        dgdu_discrete = dg!,
        dt = dtnd, adaptive = false,
        sensealg = InterpolatingAdjoint(autojacvec = false))

    @test isapprox(res_sde_u0a, res_sde_u0, rtol = 1e-5)
    @test isapprox(res_sde_pa, res_sde_p, rtol = 1e-4)

    @info res_sde_pa

    res_sde_u0a, res_sde_pa = adjoint_sensitivities(sol, EulerHeun(),
        t = Array(t),
        dgdu_discrete = dg!,
        dt = dtnd,
        adaptive = false,
        sensealg = BacksolveAdjoint())

    @test isapprox(res_sde_u0a, res_sde_u0, rtol = 1e-6)
    @test isapprox(res_sde_pa, res_sde_p, rtol = 1e-5)

    @info res_sde_pa

    res_sde_u0a, res_sde_pa = adjoint_sensitivities(sol, EulerHeun(), t = Array(t),
        dgdu_discrete = dg!,
        dt = dtnd, adaptive = false,
        sensealg = BacksolveAdjoint(autojacvec = ZygoteVJP()))

    @test all((isapprox(res_sde_u0a, res_sde_u0, rtol = 1e-6),
        isapprox(res_sde_pa, res_sde_p, rtol = 1e-6)))

    @info res_sde_pa

    res_sde_u0a, res_sde_pa = adjoint_sensitivities(sol, EulerHeun(), t = Array(t),
        dgdu_discrete = dg!,
        dt = dtnd, adaptive = false,
        sensealg = BacksolveAdjoint(autojacvec = false))

    @test isapprox(res_sde_u0a, res_sde_u0, rtol = 1e-6)
    @test isapprox(res_sde_pa, res_sde_p, rtol = 1e-6)

    @info res_sde_pa

    res_sde_u0a, res_sde_pa = adjoint_sensitivities(sol, EulerHeun(),
        t = Array(t),
        dgdu_discrete = dg!,
        dt = dtnd,
        adaptive = false,
        sensealg = InterpolatingAdjoint())

    @test isapprox(res_sde_u0a, res_sde_u0, rtol = 1e-5)
    @test isapprox(res_sde_pa, res_sde_p, rtol = 1e-4)

    @info res_sde_pa

    res_sde_u0a, res_sde_pa = adjoint_sensitivities(sol, EulerHeun(), t = Array(t),
        dgdu_discrete = dg!,
        dt = dtnd, adaptive = false,
        sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP()))

    @test all((isapprox(res_sde_u0a, res_sde_u0, rtol = 1e-5),
        isapprox(res_sde_pa, res_sde_p, rtol = 1e-4)))

    @info res_sde_pa

    res_sde_u0a, res_sde_pa = adjoint_sensitivities(sol, EulerHeun(), t = Array(t),
        dgdu_discrete = dg!,
        dt = dtnd, adaptive = false,
        sensealg = InterpolatingAdjoint(autojacvec = false))

    @test isapprox(res_sde_u0a, res_sde_u0, rtol = 1e-5)
    @test isapprox(res_sde_pa, res_sde_p, rtol = 1e-4)

    @info res_sde_pa

    function compute_grads_nd(sol)
        xdis = sol(tarray)

        mat1 = Matrix{Int}(I, 2, 2)
        mat2 = ones(2, 2) - mat1

        tmp1 = similar(p)
        tmp1 *= false

        tmp2 = similar(xdis.u[1])
        tmp2 *= false

        for (i, u) in enumerate(xdis)
            tmp1[1] += xdis.t[i] * u' * mat1 * u
            tmp1[2] += xdis.t[i] * u' * mat2 * u
            tmp1[3] += sum(sol.W(xdis.t[i])[1]) * u' * mat1 * u
            tmp1[4] += sum(sol.W(xdis.t[i])[1]) * u' * mat2 * u

            tmp2 += u .^ 2
        end

        return tmp2 ./ xdis.u[1], tmp1
    end

    res1, res2 = compute_grads_nd(soloop)

    @test isapprox(res1, res_sde_u0, rtol = 1e-4)
    @test isapprox(res2, res_sde_p', rtol = 1e-4)
end

@testset "diagonal but mixing noise tests" begin
    Random.seed!(seed)
    u₀ = [0.75, 0.5]
    p = [-1.5, 0.05, 0.2, 0.01]
    dtmix = tend / 1e3

    # Example from Roessler, SIAM J. NUMER. ANAL, 48, 922–952 with d = 2; m = 2
    function f_mixing!(du, u, p, t)
        du[1] = p[1] * u[1] + p[2] * u[2]
        du[2] = p[2] * u[1] + p[1] * u[2]
        nothing
    end

    function g_mixing!(du, u, p, t)
        du[1] = p[3] * u[1] + p[4] * u[2]
        du[2] = p[3] * u[1] + p[4] * u[2]
        nothing
    end

    function f_mixing(u, p, t)
        dx = p[1] * u[1] + p[2] * u[2]
        dy = p[2] * u[1] + p[1] * u[2]
        [dx, dy]
    end

    function g_mixing(u, p, t)
        dx = p[3] * u[1] + p[4] * u[2]
        dy = p[3] * u[1] + p[4] * u[2]
        [dx, dy]
    end

    Random.seed!(seed)
    prob = SDEProblem(f_mixing!, g_mixing!, u₀, trange, p)

    soltsave = collect(trange[1]:dtmix:trange[2])
    sol = solve(prob, EulerHeun(), dt = dtmix, save_noise = true, saveat = soltsave)

    Random.seed!(seed)
    proboop = SDEProblem(f_mixing, g_mixing, u₀, trange, p)
    soloop = solve(proboop, EulerHeun(), dt = dtmix, save_noise = true, saveat = soltsave)

    #oop

    res_sde_u0, res_sde_p = adjoint_sensitivities(soloop, EulerHeun(), t = tarray,
        dgdu_discrete = dg!,
        dt = dtmix, adaptive = false,
        sensealg = BacksolveAdjoint(noisemixing = true))

    @info res_sde_p

    res_sde_u0a, res_sde_pa = adjoint_sensitivities(soloop, EulerHeun(), t = Array(t),
        dgdu_discrete = dg!,
        dt = dtmix, adaptive = false,
        sensealg = BacksolveAdjoint(autojacvec = ZygoteVJP(),
            noisemixing = true))

    @test isapprox(res_sde_u0a, res_sde_u0, rtol = 1e-6)
    @test isapprox(res_sde_pa, res_sde_p, rtol = 1e-6)

    @info res_sde_pa

    res_sde_u0a, res_sde_pa = adjoint_sensitivities(soloop, EulerHeun(), t = Array(t),
        dgdu_discrete = dg!,
        dt = dtmix, adaptive = false,
        sensealg = BacksolveAdjoint(autojacvec = false,
            noisemixing = true))

    @test isapprox(res_sde_u0a, res_sde_u0, rtol = 1e-6)
    @test isapprox(res_sde_pa, res_sde_p, rtol = 1e-6)

    @info res_sde_pa

    res_sde_u0a, res_sde_pa = adjoint_sensitivities(soloop, EulerHeun(), t = Array(t),
        dgdu_discrete = dg!,
        dt = dtmix, adaptive = false,
        sensealg = BacksolveAdjoint(autojacvec = ReverseDiffVJP(),
            noisemixing = true))

    @test isapprox(res_sde_u0a, res_sde_u0, rtol = 1e-6)
    @test isapprox(res_sde_pa, res_sde_p, rtol = 1e-6)

    @info res_sde_pa

    res_sde_u0a, res_sde_pa = adjoint_sensitivities(soloop, EulerHeun(), t = tarray,
        dgdu_discrete = dg!,
        dt = dtmix, adaptive = false,
        sensealg = InterpolatingAdjoint(noisemixing = true))

    @test isapprox(res_sde_u0a, res_sde_u0, rtol = 1e-5)
    @test isapprox(res_sde_pa, res_sde_p, rtol = 1e-5)

    @info res_sde_pa

    res_sde_u0a, res_sde_pa = adjoint_sensitivities(soloop, EulerHeun(), t = tarray,
        dgdu_discrete = dg!,
        dt = dtmix, adaptive = false,
        sensealg = InterpolatingAdjoint(noisemixing = true,
            autojacvec = ZygoteVJP()))

    @test isapprox(res_sde_u0a, res_sde_u0, rtol = 1e-5)
    @test isapprox(res_sde_pa, res_sde_p, rtol = 1e-5)

    @info res_sde_pa

    res_sde_u0a, res_sde_pa = adjoint_sensitivities(soloop, EulerHeun(), t = tarray,
        dgdu_discrete = dg!,
        dt = dtmix, adaptive = false,
        sensealg = InterpolatingAdjoint(autojacvec = false,
            noisemixing = true))

    @test isapprox(res_sde_u0a, res_sde_u0, rtol = 1e-5)
    @test isapprox(res_sde_pa, res_sde_p, rtol = 1e-5)

    @info res_sde_pa

    res_sde_u0a, res_sde_pa = adjoint_sensitivities(soloop, EulerHeun(), t = Array(t),
        dgdu_discrete = dg!,
        dt = dtmix, adaptive = false,
        sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(),
            noisemixing = true))

    res_sde_u0a, res_sde_pa = adjoint_sensitivities(soloop, EulerHeun(), t = tarray,
        dgdu_discrete = dg!,
        dt = dtmix, adaptive = false,
        sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(),
            noisemixing = true))

    @test isapprox(res_sde_u0a, res_sde_u0, rtol = 1e-5)
    @test isapprox(res_sde_pa, res_sde_p, rtol = 1e-5)

    @info res_sde_pa

    res_sde_u0, res_sde_p = adjoint_sensitivities(sol, EulerHeun(),
        t = Array(t),
        dgdu_discrete = dg!,
        dt = dtmix, adaptive = false,
        sensealg = BacksolveAdjoint(noisemixing = true))

    @test isapprox(res_sde_u0a, res_sde_u0, rtol = 1e-5)
    @test isapprox(res_sde_pa, res_sde_p, rtol = 1e-5)

    @info res_sde_p

    res_sde_u0a, res_sde_pa = adjoint_sensitivities(sol, EulerHeun(), t = Array(t),
        dgdu_discrete = dg!,
        dt = dtmix, adaptive = false,
        sensealg = BacksolveAdjoint(autojacvec = ZygoteVJP(),
            noisemixing = true))

    @test isapprox(res_sde_u0a, res_sde_u0, rtol = 1e-6)
    @test isapprox(res_sde_pa, res_sde_p, rtol = 1e-6)

    @info res_sde_pa

    res_sde_u0a, res_sde_pa = adjoint_sensitivities(sol, EulerHeun(), t = Array(t),
        dgdu_discrete = dg!,
        dt = dtmix, adaptive = false,
        sensealg = BacksolveAdjoint(autojacvec = false,
            noisemixing = true))

    @test isapprox(res_sde_u0a, res_sde_u0, rtol = 1e-6)
    @test isapprox(res_sde_pa, res_sde_p, rtol = 1e-6)

    @info res_sde_pa

    res_sde_u0a, res_sde_pa = adjoint_sensitivities(sol, EulerHeun(), t = Array(t),
        dgdu_discrete = dg!,
        dt = dtmix, adaptive = false,
        sensealg = BacksolveAdjoint(autojacvec = ReverseDiffVJP(),
            noisemixing = true))

    @test isapprox(res_sde_u0a, res_sde_u0, rtol = 1e-6)
    @test isapprox(res_sde_pa, res_sde_p, rtol = 1e-6)

    @info res_sde_pa

    res_sde_u0a, res_sde_pa = adjoint_sensitivities(sol, EulerHeun(),
        t = Array(t),
        dgdu_discrete = dg!,
        dt = dtmix,
        adaptive = false,
        sensealg = InterpolatingAdjoint(noisemixing = true))

    @test isapprox(res_sde_u0a, res_sde_u0, rtol = 1e-5)
    @test isapprox(res_sde_pa, res_sde_p, rtol = 1e-5) # would pass with 1e-4 but last noise value is off

    @info res_sde_pa

    res_sde_u0a, res_sde_pa = adjoint_sensitivities(sol, EulerHeun(), t = Array(t),
        dgdu_discrete = dg!,
        dt = dtmix, adaptive = false,
        sensealg = InterpolatingAdjoint(noisemixing = true,
            autojacvec = ZygoteVJP()))

    @test isapprox(res_sde_u0a, res_sde_u0, rtol = 1e-5)
    @test isapprox(res_sde_pa, res_sde_p, rtol = 1e-5)

    @info res_sde_pa

    res_sde_u0a, res_sde_pa = adjoint_sensitivities(sol, EulerHeun(), t = tarray,
        dgdu_discrete = dg!,
        dt = dtmix, adaptive = false,
        sensealg = InterpolatingAdjoint(autojacvec = false,
            noisemixing = true))

    @test isapprox(res_sde_u0a, res_sde_u0, rtol = 1e-5)
    @test isapprox(res_sde_pa, res_sde_p, rtol = 1e-5)

    @info res_sde_pa

    res_sde_u0a, res_sde_pa = adjoint_sensitivities(sol, EulerHeun(), t = Array(t),
        dgdu_discrete = dg!,
        dt = dtmix, adaptive = false,
        sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(),
            noisemixing = true))

    @test isapprox(res_sde_u0a, res_sde_u0, rtol = 1e-5)
    @test isapprox(res_sde_pa, res_sde_p, rtol = 1e-5)

    @info res_sde_pa

    function GSDE(p)
        Random.seed!(seed)
        tmp_prob = remake(prob, u0 = eltype(p).(prob.u0), p = p,
            tspan = eltype(p).(prob.tspan))
        _sol = solve(tmp_prob, EulerHeun(), dt = dtmix, adaptive = false, saveat = Array(t))
        A = convert(Array, _sol)
        res = g(A, p, nothing)
    end

    res_sde_forward = ForwardDiff.gradient(GSDE, p)

    @test isapprox(res_sde_p', res_sde_forward, rtol = 1e-5)

    function GSDE2(u0)
        Random.seed!(seed)
        tmp_prob = remake(prob, u0 = u0, p = eltype(p).(prob.p),
            tspan = eltype(p).(prob.tspan))
        _sol = solve(tmp_prob, EulerHeun(), dt = dtmix, adaptive = false, saveat = Array(t))
        A = convert(Array, _sol)
        res = g(A, p, nothing)
    end

    res_sde_forward = ForwardDiff.gradient(GSDE2, u₀)

    @test isapprox(res_sde_forward, res_sde_u0, rtol = 1e-5)
end

@testset "mixing noise inplace/oop tests" begin
    Random.seed!(seed)
    u₀ = [0.75, 0.5]
    p = [-1.5, 0.05, 0.2, 0.01]
    dtmix = tend / 1e3

    # Example from Roessler, SIAM J. NUMER. ANAL, 48, 922–952 with d = 2; m = 2
    function f_mixing!(du, u, p, t)
        du[1] = p[1] * u[1] + p[2] * u[2]
        du[2] = p[2] * u[1] + p[1] * u[2]
        nothing
    end

    function g_mixing!(du, u, p, t)
        du[1] = p[3] * u[1] + p[4] * u[2]
        du[2] = p[3] * u[1] + p[4] * u[2]
        nothing
    end

    function f_mixing(u, p, t)
        dx = p[1] * u[1] + p[2] * u[2]
        dy = p[2] * u[1] + p[1] * u[2]
        [dx, dy]
    end

    function g_mixing(u, p, t)
        dx = p[3] * u[1] + p[4] * u[2]
        dy = p[3] * u[1] + p[4] * u[2]
        [dx, dy]
    end

    Random.seed!(seed)
    prob = SDEProblem(f_mixing!, g_mixing!, u₀, trange, p)
    soltsave = collect(trange[1]:dtmix:trange[2])
    sol = solve(prob, EulerHeun(), dt = dtmix, save_noise = true, saveat = soltsave)

    Random.seed!(seed)
    proboop = SDEProblem(f_mixing, g_mixing, u₀, trange, p)
    soloop = solve(proboop, EulerHeun(), dt = dtmix, save_noise = true, saveat = soltsave)

    @test sol.u≈soloop.u atol=1e-14

    # BacksolveAdjoint

    res_sde_u0, res_sde_p = adjoint_sensitivities(soloop, EulerHeun(), t = tarray,
        dgdu_discrete = dg!,
        dt = dtmix, adaptive = false,
        sensealg = BacksolveAdjoint(noisemixing = true))

    res_sde_u02, res_sde_p2 = adjoint_sensitivities(sol, EulerHeun(),
        t = tarray,
        dgdu_discrete = dg!,
        dt = dtmix,
        adaptive = false,
        sensealg = BacksolveAdjoint(noisemixing = true))

    @test res_sde_u0≈res_sde_u02 atol=1e-14
    @test res_sde_p≈res_sde_p2 atol=1e-14

    @show res_sde_u0

    adjproboop = SDEAdjointProblem(soloop,
        BacksolveAdjoint(autojacvec = ZygoteVJP(),
            noisemixing = true), EulerHeun(),
        tarray, dg!)
    adj_soloop = solve(adjproboop, EulerHeun(); dt = dtmix, tstops = soloop.t,
        adaptive = false)

    @test adj_soloop[end][(length(p) + length(u₀) + 1):end] == soloop.u[1]
    @test adj_soloop[end][1:length(u₀)] == res_sde_u0
    @test adj_soloop[end][(length(u₀) + 1):(end - length(u₀))] == res_sde_p'

    adjprob = SDEAdjointProblem(sol,
        BacksolveAdjoint(autojacvec = ReverseDiffVJP(),
            noisemixing = true, checkpointing = true),
        EulerHeun(), tarray, dg!)
    adj_sol = solve(adjprob, EulerHeun(); dt = dtmix, adaptive = false, tstops = soloop.t)

    @test adj_soloop[end]≈adj_sol.u[end] rtol=1e-15

    adjprob = SDEAdjointProblem(sol,
        BacksolveAdjoint(autojacvec = ReverseDiffVJP(),
            noisemixing = true, checkpointing = false),
        EulerHeun(), tarray, dg!)
    adj_sol = solve(adjprob, EulerHeun(); dt = dtmix, adaptive = false, tstops = soloop.t)

    @test adj_soloop[end]≈adj_sol.u[end] rtol=1e-8

    # InterpolatingAdjoint

    res_sde_u0, res_sde_p = adjoint_sensitivities(soloop, EulerHeun(), t = tarray,
        dgdu_discrete = dg!,
        dt = dtmix, adaptive = false,
        sensealg = InterpolatingAdjoint(noisemixing = true))

    res_sde_u02, res_sde_p2 = adjoint_sensitivities(sol, EulerHeun(),
        t = tarray,
        dgdu_discrete = dg!,
        dt = dtmix,
        adaptive = false,
        sensealg = InterpolatingAdjoint(noisemixing = true))

    @test res_sde_u0≈res_sde_u02 atol=1e-8
    @test res_sde_p≈res_sde_p2 atol=5e-8

    @show res_sde_u0

    adjproboop = SDEAdjointProblem(soloop,
        InterpolatingAdjoint(autojacvec = ReverseDiffVJP(),
            noisemixing = true), EulerHeun(),
        tarray, dg!)
    adj_soloop = solve(adjproboop, EulerHeun(); dt = dtmix, tstops = soloop.t,
        adaptive = false)

    @test adj_soloop[end][1:length(u₀)]≈res_sde_u0 atol=1e-14
    @test adj_soloop[end][(length(u₀) + 1):end]≈res_sde_p' atol=1e-14

    adjprob = SDEAdjointProblem(sol,
        InterpolatingAdjoint(autojacvec = ReverseDiffVJP(),
            noisemixing = true,
            checkpointing = true), EulerHeun(),
        tarray, dg!)
    adj_sol = solve(adjprob, EulerHeun(); dt = dtmix, adaptive = false, tstops = soloop.t)

    @test adj_soloop[end]≈adj_sol.u[end] rtol=1e-8

    adjprob = SDEAdjointProblem(sol,
        InterpolatingAdjoint(autojacvec = ReverseDiffVJP(),
            noisemixing = true,
            checkpointing = false), EulerHeun(),
        tarray, dg!)
    adj_sol = solve(adjprob, EulerHeun(); dt = dtmix, adaptive = false, tstops = soloop.t)

    @test adj_soloop[end]≈adj_sol.u[end] rtol=1e-8
end

@testset "mutating non-diagonal noise" begin
    a!(du, u, _p, t) = (du .= -u)
    a(u, _p, t) = -u

    function b!(du, u, _p, t)
        KR, KI = _p[1:2]

        du[1, 1] = KR
        du[2, 1] = KI
    end

    function b(u, _p, t)
        KR, KI = _p[1:2]

        [KR zero(KR)
         KI zero(KR)]
    end

    p = [1.0, 0.0]

    prob! = SDEProblem{true}(a!, b!, [0.0, 0.0], (0.0, 0.1), p,
        noise_rate_prototype = eltype(p).(zeros(2, 2)))
    prob = SDEProblem{false}(a, b, [0.0, 0.0], (0.0, 0.1), p,
        noise_rate_prototype = eltype(p).(zeros(2, 2)))

    function loss(p; SDEprob = prob, sensealg = BacksolveAdjoint())
        _prob = remake(SDEprob, p = p)
        sol = solve(_prob, EulerHeun(), dt = 1e-5, sensealg = sensealg)
        return sum(Array(sol))
    end

    function compute_dp(p, SDEprob, sensealg)
        Random.seed!(seed)
        Zygote.gradient(p -> loss(p, SDEprob = SDEprob, sensealg = sensealg), p)[1]
    end

    # test mutating against non-mutating

    # non-mutating

    dp1 = compute_dp(p, prob, ForwardDiffSensitivity())
    dp2 = compute_dp(p, prob, BacksolveAdjoint())
    dp3 = compute_dp(p, prob, InterpolatingAdjoint())

    @show dp1 dp2 dp3

    # different vjp choice
    _dp2 = compute_dp(p, prob, BacksolveAdjoint(autojacvec = ReverseDiffVJP()))
    @test dp2≈_dp2 rtol=1e-8
    _dp3 = compute_dp(p, prob, InterpolatingAdjoint(autojacvec = ReverseDiffVJP()))
    @test dp3≈_dp3 rtol=1e-8

    # mutating
    _dp1 = compute_dp(p, prob!, ForwardDiffSensitivity())
    _dp2 = compute_dp(p, prob!, BacksolveAdjoint(autojacvec = ReverseDiffVJP()))
    _dp3 = compute_dp(p, prob!, InterpolatingAdjoint(autojacvec = ReverseDiffVJP()))
    @test !any(isnan, compute_dp(p, prob!, InterpolatingAdjoint()))

    @test dp1≈_dp1 rtol=1e-8
    @test dp2≈_dp2 rtol=1e-8
    @test dp3≈_dp3 rtol=1e-8
    @test_broken dp3≈_dp4 rtol=1e-8
end

@testset "Non-square diffusion matrix" begin
    tstart = 0.0
    tend = 0.1
    dt = 0.005
    tspan = (tstart, tend)

    p = [1.0, 0.1]

    dim = 3 # dimension of the system
    m = 2 # number of Brownian motions
    u0 = ones(dim)
    noise_rate_prototype = zeros(dim, m) # diffusion matrix

    function b_system(u, p, t)
        [p[1] * u[i] for i in 1:dim]
    end

    function σ_system(u, p, t)
        # construct a 3x2 matrix
        dx = diagm([p[2] * u[i] for i in 1:m])
        [dx; zeros(eltype(dx), 1, m)] # 1, m
    end

    prob = SDEProblem(b_system, σ_system, u0, tspan, p,
        noise_rate_prototype = noise_rate_prototype)
    sol = solve(prob, EM(), dt = dt, save_noise = true)
    ts = sol.t
    Ws = sol.W.W
    Z = DiffEqNoiseProcess.NoiseGrid(ts, Ws)

    function loss(p; sensealg = nothing, Z = nothing)
        _prob = remake(prob,
            u0 = convert.(eltype(p), prob.u0),
            p = p,
            noise = Z            # noise_rate_prototype = noise_rate_prototype
        )
        sol = solve(_prob, EulerHeun(), dt = dt, sensealg = sensealg)
        sum(abs2, sol.u[end])
    end

    loss(p, Z = Z)

    gFinD = FiniteDiff.finite_difference_gradient(p -> loss(p, Z = Z), p)
    gFD = ForwardDiff.gradient(p -> loss(p, Z = Z), p)
    gZy = Zygote.gradient(p -> loss(p, Z = Z), p)[1]

    @test gFinD≈gFD rtol=1e-4
    @test gFinD≈gZy rtol=1e-4

    @show gFinD - gFD
    @show gFinD - gZy

    # ReverseDiffVJP(), ZygoteVJP()
    gZy = Zygote.gradient(
        p -> loss(p, Z = Z,
            sensealg = BacksolveAdjoint(autojacvec = ZygoteVJP())),
        p)[1]
    @test gFinD≈gZy rtol=1e-4
    gZy = Zygote.gradient(
        p -> loss(p, Z = Z,
            sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP())),
        p)[1]
    @test gFinD≈gZy rtol=1e-4
    gZy = Zygote.gradient(p -> loss(p, Z = Z, sensealg = ForwardDiffSensitivity()), p)[1]
    @test gFinD≈gZy rtol=1e-4
end

@testset "Diagonal forward SDE and Non-Diagonal Reverse SDE with Non-Square input" begin
    u = Float32.([0.0f0 1.0f0 2.0f0;
                  0.0f0 0.0f0 1.0f0])

    p = Float32.([-1.5 0.05 0.2;
                  0.01 0.4 1.9])

    tspan = (0.0f0, 1.0f0)

    function f(u, p, t)
        u .* p
    end

    function g(u, p, t)
        u .* p + p
    end

    function dg!(out, u, p, t, i)
        (out .= vec(u))
    end

    sdefun = SDEFunction(f, g)
    prob2 = SDEProblem(sdefun, g, u, tspan, p)

    sol = solve(prob2, EulerHeun(), dt = 0.01)

    du, dp = adjoint_sensitivities(sol, EulerHeun(), t = tspan,
        dgdu_discrete = dg!,
        dt = 0.01, adaptive = false,
        sensealg = BacksolveAdjoint())
    @test !iszero(du)
    @test !iszero(dp)
end
