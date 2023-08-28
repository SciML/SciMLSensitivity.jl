using Test, LinearAlgebra
using SciMLSensitivity, SteadyStateDiffEq, DiffEqBase, NLsolve
using OrdinaryDiffEq
using NonlinearSolve, SciMLNLSolve
using ForwardDiff, Calculus
using Zygote
using Random
Random.seed!(12345)

@testset "Adjoint sensitivities of steady state solver" begin
    function f!(du, u, p, t)
        du[1] = p[1] + p[2] * u[1]
        du[2] = p[3] * u[1] + p[4] * u[2]
    end

    function jac!(J, u, p, t) #df/dx
        J[1, 1] = p[2]
        J[2, 1] = p[3]
        J[1, 2] = 0
        J[2, 2] = p[4]
        nothing
    end

    function paramjac!(fp, u, p, t) #df/dp
        fp[1, 1] = 1
        fp[2, 1] = 0
        fp[1, 2] = u[1]
        fp[2, 2] = 0
        fp[1, 3] = 0
        fp[2, 3] = u[1]
        fp[1, 4] = 0
        fp[2, 4] = u[2]
        nothing
    end

    function dgdu!(out, u, p, t, i)
        (out .= -2.0 .+ u)
    end

    function dgdp!(out, u, p, t, i)
        (out .= p)
    end

    function g(u, p, t)
        sum((2.0 .- u) .^ 2) / 2 + sum(p .^ 2) / 2
    end

    u0 = zeros(2)
    p = [2.0, -2.0, 1.0, -4.0]
    prob = SteadyStateProblem(f!, u0, p)
    abstol = 1e-10
    @testset "for p" begin
        println("Calculate adjoint sensitivities from Jacobians")

        sol_analytical = [-p[1] / p[2], p[1] * p[3] / (p[2] * p[4])]

        J = zeros(2, 2)
        fp = zeros(2, 4)
        gp = zeros(4)
        gx = zeros(1, 2)
        delg_delp = copy(p)

        jac!(J, sol_analytical, p, nothing)
        dgdu!(vec(gx), sol_analytical, p, nothing, nothing)
        paramjac!(fp, sol_analytical, p, nothing)

        lambda = J' \ gx'
        res_analytical = delg_delp' - lambda' * fp # = -gx*inv(J)*fp

        @info "Expected result" sol_analytical, res_analytical,
        delg_delp' - gx * inv(J) * fp

        @info "Calculate adjoint sensitivities from autodiff & numerical diff"
        function G(p)
            tmp_prob = remake(prob, u0 = convert.(eltype(p), prob.u0), p = p)
            sol = solve(tmp_prob, DynamicSS(Rodas5()))
            A = convert(Array, sol)
            g(A, p, nothing)
        end
        res1 = ForwardDiff.gradient(G, p)
        res2 = Calculus.gradient(G, p)
        #@info res1, res2, res_analytical

        @test res1≈res_analytical' rtol=1e-7
        @test res2≈res_analytical' rtol=1e-7
        @test res1≈res2 rtol=1e-7

        @info "Adjoint sensitivities"

        # with jac, param_jac
        f1 = ODEFunction(f!; jac = jac!, paramjac = paramjac!)
        prob1 = SteadyStateProblem(f1, u0, p)
        sol1 = solve(prob1, DynamicSS(Rodas5(), reltol = 1e-14, abstol = 1e-14),
            reltol = 1e-14, abstol = 1e-14)

        res1a = adjoint_sensitivities(sol1, DynamicSS(Rodas5()),
            sensealg = SteadyStateAdjoint(), dgdu = dgdu!,
            dgdp = dgdp!, g = g)
        res1b = adjoint_sensitivities(sol1, DynamicSS(Rodas5()),
            sensealg = SteadyStateAdjoint(), g = g)
        res1c = adjoint_sensitivities(sol1, DynamicSS(Rodas5()),
            sensealg = SteadyStateAdjoint(autodiff = false),
            g = g)
        res1d = adjoint_sensitivities(sol1, DynamicSS(Rodas5()),
            sensealg = SteadyStateAdjoint(autojacvec = TrackerVJP()),
            g = g)
        res1e = adjoint_sensitivities(sol1, DynamicSS(Rodas5()),
            sensealg = SteadyStateAdjoint(autojacvec = ReverseDiffVJP()),
            g = g)
        res1f = adjoint_sensitivities(sol1, DynamicSS(Rodas5()),
            sensealg = SteadyStateAdjoint(autojacvec = ZygoteVJP()),
            g = g)
        res1g = adjoint_sensitivities(sol1, DynamicSS(Rodas5()),
            sensealg = SteadyStateAdjoint(autodiff = false,
                autojacvec = false),
            g = g)
        res1h = adjoint_sensitivities(sol1, DynamicSS(Rodas5()),
            sensealg = SteadyStateAdjoint(autojacvec = EnzymeVJP()),
            g = g)

        # with jac, without param_jac
        f2 = ODEFunction(f!; jac = jac!)
        prob2 = SteadyStateProblem(f2, u0, p)
        sol2 = solve(prob2, DynamicSS(Rodas5(), reltol = 1e-14, abstol = 1e-14),
            reltol = 1e-14, abstol = 1e-14)
        res2a = adjoint_sensitivities(sol2, DynamicSS(Rodas5()),
            sensealg = SteadyStateAdjoint(), dgdu = dgdu!,
            dgdp = dgdp!, g = g)
        res2b = adjoint_sensitivities(sol2, DynamicSS(Rodas5()),
            sensealg = SteadyStateAdjoint(), g = g)
        res2c = adjoint_sensitivities(sol2, DynamicSS(Rodas5()),
            sensealg = SteadyStateAdjoint(autodiff = false),
            g = g)
        res2d = adjoint_sensitivities(sol2, DynamicSS(Rodas5()),
            sensealg = SteadyStateAdjoint(autojacvec = TrackerVJP()),
            g = g)
        res2e = adjoint_sensitivities(sol2, DynamicSS(Rodas5()),
            sensealg = SteadyStateAdjoint(autojacvec = ReverseDiffVJP()),
            g = g)
        res2f = adjoint_sensitivities(sol2, DynamicSS(Rodas5()),
            sensealg = SteadyStateAdjoint(autojacvec = ZygoteVJP()),
            g = g)
        res2g = adjoint_sensitivities(sol2, DynamicSS(Rodas5()),
            sensealg = SteadyStateAdjoint(autodiff = false,
                autojacvec = false),
            g = g)
        res2h = adjoint_sensitivities(sol2, DynamicSS(Rodas5()),
            sensealg = SteadyStateAdjoint(autojacvec = EnzymeVJP()),
            g = g)

        # without jac, without param_jac
        f3 = ODEFunction(f!)
        prob3 = SteadyStateProblem(f3, u0, p)
        sol3 = solve(prob3, DynamicSS(Rodas5(), reltol = 1e-14, abstol = 1e-14),
            reltol = 1e-14, abstol = 1e-14)
        res3a = adjoint_sensitivities(sol3, DynamicSS(Rodas5()),
            sensealg = SteadyStateAdjoint(), dgdu = dgdu!,
            dgdp = dgdp!, g = g)
        res3b = adjoint_sensitivities(sol3, DynamicSS(Rodas5()),
            sensealg = SteadyStateAdjoint(), g = g)
        res3c = adjoint_sensitivities(sol3, DynamicSS(Rodas5()),
            sensealg = SteadyStateAdjoint(autodiff = false),
            g = g)
        res3d = adjoint_sensitivities(sol3, DynamicSS(Rodas5()),
            sensealg = SteadyStateAdjoint(autojacvec = TrackerVJP()),
            g = g)
        res3e = adjoint_sensitivities(sol3, DynamicSS(Rodas5()),
            sensealg = SteadyStateAdjoint(autojacvec = ReverseDiffVJP()),
            g = g)
        res3f = adjoint_sensitivities(sol3, DynamicSS(Rodas5()),
            sensealg = SteadyStateAdjoint(autojacvec = ZygoteVJP()),
            g = g)
        res3g = adjoint_sensitivities(sol3, DynamicSS(Rodas5()),
            sensealg = SteadyStateAdjoint(autodiff = false,
                autojacvec = false),
            g = g)
        res3h = adjoint_sensitivities(sol3, DynamicSS(Rodas5()),
            sensealg = SteadyStateAdjoint(autojacvec = EnzymeVJP()),
            g = g)

        @test norm(res_analytical' .- res1a) < 1e-7
        @test norm(res_analytical' .- res1b) < 1e-7
        @test norm(res_analytical' .- res1c) < 1e-7
        @test norm(res_analytical' .- res1d) < 1e-7
        @test norm(res_analytical' .- res1e) < 1e-7
        @test norm(res_analytical' .- res1f) < 1e-7
        @test norm(res_analytical' .- res1g) < 1e-7
        @test norm(res_analytical' .- res1h) < 1e-7
        @test norm(res_analytical' .- res2a) < 1e-7
        @test norm(res_analytical' .- res2b) < 1e-7
        @test norm(res_analytical' .- res2c) < 1e-7
        @test norm(res_analytical' .- res2d) < 1e-7
        @test norm(res_analytical' .- res2e) < 1e-7
        @test norm(res_analytical' .- res2f) < 1e-7
        @test norm(res_analytical' .- res2g) < 1e-7
        @test norm(res_analytical' .- res2h) < 1e-7
        @test norm(res_analytical' .- res3a) < 1e-7
        @test norm(res_analytical' .- res3b) < 1e-7
        @test norm(res_analytical' .- res3c) < 1e-7
        @test norm(res_analytical' .- res3d) < 1e-7
        @test norm(res_analytical' .- res3e) < 1e-7
        @test norm(res_analytical' .- res3f) < 1e-7
        @test norm(res_analytical' .- res3g) < 1e-7
        @test norm(res_analytical' .- res3h) < 1e-7

        @info "oop checks"
        function foop(u, p, t)
            dx = p[1] + p[2] * u[1]
            dy = p[3] * u[1] + p[4] * u[2]
            [dx, dy]
        end
        proboop = SteadyStateProblem(foop, u0, p)
        soloop = solve(proboop, DynamicSS(Rodas5(), reltol = 1e-14, abstol = 1e-14),
            reltol = 1e-14, abstol = 1e-14)

        res4a = adjoint_sensitivities(soloop, DynamicSS(Rodas5()),
            sensealg = SteadyStateAdjoint(), dgdu = dgdu!,
            dgdp = dgdp!, g = g)
        res4b = adjoint_sensitivities(soloop, DynamicSS(Rodas5()),
            sensealg = SteadyStateAdjoint(), g = g)
        res4c = adjoint_sensitivities(soloop, DynamicSS(Rodas5()),
            sensealg = SteadyStateAdjoint(autodiff = false),
            g = g)
        res4d = adjoint_sensitivities(soloop, DynamicSS(Rodas5()),
            sensealg = SteadyStateAdjoint(autojacvec = TrackerVJP()),
            g = g)
        res4e = adjoint_sensitivities(soloop, DynamicSS(Rodas5()),
            sensealg = SteadyStateAdjoint(autojacvec = ReverseDiffVJP()),
            g = g)
        res4f = adjoint_sensitivities(soloop, DynamicSS(Rodas5()),
            sensealg = SteadyStateAdjoint(autojacvec = ZygoteVJP()),
            g = g)
        res4g = adjoint_sensitivities(soloop, DynamicSS(Rodas5()),
            sensealg = SteadyStateAdjoint(autodiff = false,
                autojacvec = false),
            g = g)
        res4h = adjoint_sensitivities(soloop, DynamicSS(Rodas5()),
            sensealg = SteadyStateAdjoint(autodiff = true,
                autojacvec = false),
            g = g)

        @test norm(res_analytical' .- res4a) < 1e-7
        @test norm(res_analytical' .- res4b) < 1e-7
        @test norm(res_analytical' .- res4c) < 1e-7
        @test norm(res_analytical' .- res4d) < 1e-7
        @test norm(res_analytical' .- res4e) < 1e-7
        @test norm(res_analytical' .- res4f) < 1e-7
        @test norm(res_analytical' .- res4g) < 1e-7
        @test norm(res_analytical' .- res4h) < 1e-7
    end

    @testset "for u0: (should be zero, steady state does not depend on initial condition)" begin
        res5 = ForwardDiff.gradient(prob.u0) do u0
            tmp_prob = remake(prob, u0 = u0)
            sol = solve(tmp_prob, DynamicSS(Rodas5()))
            A = convert(Array, sol)
            g(A, p, nothing)
        end
        @test abs(dot(res5, res5)) < 1e-7
    end
end

@testset "concrete_solve derivatives steady state solver" begin
    function g1(u, p, t)
        sum(u)
    end

    function g2(u, p, t)
        sum((2.0 .- u) .^ 2) / 2
    end

    u0 = zeros(2)
    p = [2.0, -2.0, 1.0, -4.0]

    @testset "iip" begin
        function f!(du, u, p, t)
            du[1] = p[1] + p[2] * u[1]
            du[2] = p[3] * u[1] + p[4] * u[2]
        end
        prob = SteadyStateProblem(f!, u0, p)

        sol = solve(prob, DynamicSS(Rodas5()))
        res1 = adjoint_sensitivities(sol, DynamicSS(Rodas5()),
            sensealg = SteadyStateAdjoint(), g = g1)
        res2 = adjoint_sensitivities(sol, DynamicSS(Rodas5()),
            sensealg = SteadyStateAdjoint(), g = g2)

        dp1 = Zygote.gradient(p -> sum(solve(prob, DynamicSS(Rodas5()), u0 = u0, p = p,
                sensealg = SteadyStateAdjoint())), p)
        dp2 = Zygote.gradient(p -> sum((2.0 .-
                                        solve(prob, DynamicSS(Rodas5()), u0 = u0, p = p,
                sensealg = SteadyStateAdjoint())) .^ 2) / 2.0,
            p)

        dp1d = Zygote.gradient(p -> sum(solve(prob, DynamicSS(Rodas5()), u0 = u0, p = p)),
            p)
        dp2d = Zygote.gradient(p -> sum((2.0 .-
                                         solve(prob,
                DynamicSS(Rodas5()),
                u0 = u0,
                p = p)) .^
                                        2) / 2.0, p)

        @test res1≈dp1[1] rtol=1e-12
        @test res2≈dp2[1] rtol=1e-12
        @test res1≈dp1d[1] rtol=1e-12
        @test res2≈dp2d[1] rtol=1e-12

        res1 = Zygote.gradient(p -> sum(Array(solve(prob, DynamicSS(Rodas5()), u0 = u0,
                p = p, sensealg = SteadyStateAdjoint()))[1]),
            p)
        dp1 = Zygote.gradient(p -> sum(solve(prob, DynamicSS(Rodas5()), u0 = u0, p = p,
                save_idxs = 1:1,
                sensealg = SteadyStateAdjoint())), p)
        dp2 = Zygote.gradient(p -> solve(prob, DynamicSS(Rodas5()), u0 = u0, p = p,
                save_idxs = 1, sensealg = SteadyStateAdjoint())[1],
            p)

        dp1d = Zygote.gradient(p -> sum(solve(prob, DynamicSS(Rodas5()), u0 = u0, p = p,
                save_idxs = 1:1)), p)
        dp2d = Zygote.gradient(p -> solve(prob, DynamicSS(Rodas5()), u0 = u0, p = p,
                save_idxs = 1)[1], p)
        @test res1[1]≈dp1[1] rtol=1e-10
        @test res1[1]≈dp2[1] rtol=1e-10
        @test res1[1]≈dp1d[1] rtol=1e-10
        @test res1[1]≈dp2d[1] rtol=1e-10
    end

    @testset "oop" begin
        function f(u, p, t)
            dx = p[1] + p[2] * u[1]
            dy = p[3] * u[1] + p[4] * u[2]
            [dx, dy]
        end
        proboop = SteadyStateProblem(f, u0, p)

        soloop = solve(proboop, DynamicSS(Rodas5()))
        res1oop = adjoint_sensitivities(soloop, DynamicSS(Rodas5()),
            sensealg = SteadyStateAdjoint(), g = g1)
        res2oop = adjoint_sensitivities(soloop, DynamicSS(Rodas5()),
            sensealg = SteadyStateAdjoint(), g = g2)

        dp1oop = Zygote.gradient(p -> sum(solve(proboop, DynamicSS(Rodas5()), u0 = u0,
                p = p, sensealg = SteadyStateAdjoint())), p)
        dp2oop = Zygote.gradient(p -> sum((2.0 .-
                                           solve(proboop, DynamicSS(Rodas5()), u0 = u0,
                p = p, sensealg = SteadyStateAdjoint())) .^
                                          2) / 2.0, p)
        dp1oopd = Zygote.gradient(p -> sum(solve(proboop, DynamicSS(Rodas5()), u0 = u0,
                p = p)), p)
        dp2oopd = Zygote.gradient(p -> sum((2.0 .-
                                            solve(proboop, DynamicSS(Rodas5()), u0 = u0,
                p = p)) .^ 2) / 2.0, p)

        @test res1oop≈dp1oop[1] rtol=1e-12
        @test res2oop≈dp2oop[1] rtol=1e-12
        @test res1oop≈dp1oopd[1] rtol=1e-8
        @test res2oop≈dp2oopd[1] rtol=1e-8

        res1oop = Zygote.gradient(p -> sum(Array(solve(proboop, DynamicSS(Rodas5()),
                u0 = u0, p = p,
                sensealg = SteadyStateAdjoint()))[1]),
            p)
        dp1oop = Zygote.gradient(p -> sum(solve(proboop, DynamicSS(Rodas5()), u0 = u0,
                p = p, save_idxs = 1:1,
                sensealg = SteadyStateAdjoint())), p)
        dp2oop = Zygote.gradient(p -> solve(proboop, DynamicSS(Rodas5()), u0 = u0, p = p,
                save_idxs = 1, sensealg = SteadyStateAdjoint())[1],
            p)
        dp1oopd = Zygote.gradient(p -> sum(solve(proboop, DynamicSS(Rodas5()), u0 = u0,
                p = p, save_idxs = 1:1)), p)
        dp2oopd = Zygote.gradient(p -> solve(proboop, DynamicSS(Rodas5()), u0 = u0, p = p,
                save_idxs = 1)[1], p)
        @test res1oop[1]≈dp1oop[1] rtol=1e-10
        @test res1oop[1]≈dp2oop[1] rtol=1e-10
    end
end

@testset "NonlinearProblem" begin
    u0 = [0.0]
    p = [2.0, 1.0]
    prob = NonlinearProblem((du, u, p) -> du[1] = u[1] - p[1] + p[2], u0, p)
    prob2 = NonlinearProblem{false}((u, p) -> u .- p[1] .+ p[2], u0, p)

    solve1 = solve(remake(prob, p = p), NewtonRaphson())
    solve2 = solve(prob2, NewtonRaphson())
    @test solve1.u == solve2.u

    prob3 = SteadyStateProblem((u, p, t) -> -u .+ p[1] .- p[2], [0.0], p)
    solve3 = solve(prob3, DynamicSS(Rodas5()))
    @test solve1.u≈solve3.u rtol=1e-6

    prob4 = SteadyStateProblem((du, u, p, t) -> du[1] = -u[1] + p[1] - p[2], [0.0], p)
    solve4 = solve(prob4, DynamicSS(Rodas5()))
    @test solve3.u≈solve4.u rtol=1e-10

    function test_loss(p, prob; alg = NewtonRaphson())
        _prob = remake(prob, p = p)
        sol = sum(solve(_prob, alg,
            sensealg = SteadyStateAdjoint(autojacvec = ReverseDiffVJP())))
        return sol
    end

    test_loss(p, prob)

    test_loss(p, prob2)
    test_loss(p, prob3, alg = DynamicSS(Rodas5()))
    test_loss(p, prob4, alg = DynamicSS(Rodas5()))
    test_loss(p, prob2, alg = SimpleNewtonRaphson())

    dp1 = Zygote.gradient(p -> test_loss(p, prob), p)[1]
    dp2 = Zygote.gradient(p -> test_loss(p, prob2), p)[1]
    dp3 = Zygote.gradient(p -> test_loss(p, prob3, alg = DynamicSS(Rodas5())), p)[1]
    dp4 = Zygote.gradient(p -> test_loss(p, prob4, alg = DynamicSS(Rodas5())), p)[1]
    dp5 = Zygote.gradient(p -> test_loss(p, prob2, alg = SimpleNewtonRaphson()), p)[1]
    dp6 = Zygote.gradient(p -> test_loss(p, prob2, alg = Klement()), p)[1]
    dp7 = Zygote.gradient(p -> test_loss(p, prob2, alg = SimpleTrustRegion()), p)[1]
    dp8 = Zygote.gradient(p -> test_loss(p, prob2, alg = NLSolveJL()), p)[1]

    @test dp1≈dp2 rtol=1e-10
    @test dp1≈dp3 rtol=1e-10
    @test dp1≈dp4 rtol=1e-10
    @test dp1≈dp5 rtol=1e-10
    @test dp1≈dp6 rtol=1e-10
    @test dp1≈dp7 rtol=1e-10
    @test dp1≈dp8 rtol=1e-10

    # Larger Batched Problem: For testing the Iterative Solvers Path
    u0 = zeros(128)
    p = [2.0, 1.0]

    prob = NonlinearProblem((u, p) -> u .- p[1] .+ p[2], u0, p)
    solve1 = solve(remake(prob, p = p), NewtonRaphson())

    function test_loss(p, prob; alg = NewtonRaphson())
        _prob = remake(prob, p = p)
        sol = sum(solve(_prob, alg,
            sensealg = SteadyStateAdjoint(autojacvec = ZygoteVJP())))
        return sol
    end

    test_loss(p, prob)

    dp1 = Zygote.gradient(p -> test_loss(p, prob), p)[1]

    @test dp1[1] ≈ 128
    @test dp1[2] ≈ -128
end

@testset "Continuous sensitivity tools" begin
    function f!(du, u, p, t)
        du[1] = p[1] + p[2] * u[1]
        du[2] = p[3] * u[1] + p[4] * u[2]
    end
    function g(u, p, t)
        sum((2.0 .- u) .^ 2) / 2 + sum(p .^ 2) / 2
    end

    u0 = zeros(2)
    p = [2.0, -2.0, 1.0, -4.0]
    prob = ODEProblem(f!, u0, (0, Inf), p)

    tol = 1e-10
    # steady state callback
    function condition(u, t, integrator)
        testval = first(get_tmp_cache(integrator))
        DiffEqBase.get_du!(testval, integrator)
        all(testval .< tol)
    end
    affect!(integrator) = terminate!(integrator)
    cb_t1 = DiscreteCallback(condition, affect!, save_positions = (false, true))
    cb_t2 = DiscreteCallback(condition, affect!, save_positions = (true, true))

    for cb_t in (cb_t1, cb_t2)
        sol = solve(prob, Tsit5(), reltol = tol, abstol = tol, callback = cb_t,
            save_start = false, save_everystep = false)

        # derivative with respect to u0 and p0
        function loss(u0, p; sensealg = nothing, save_start = false, save_everystep = false)
            _prob = remake(prob, u0 = u0, p = p)
            # saving arguments can have a huge influence here
            sol = solve(_prob, Tsit5(), reltol = tol, abstol = tol, sensealg = sensealg,
                callback = cb_t,
                save_start = save_start, save_everystep = save_everystep)
            res = sol.u[end]
            g(res, p, nothing)
        end

        du0 = ForwardDiff.gradient((u0) -> loss(u0, p), u0)
        dp = ForwardDiff.gradient((p) -> loss(u0, p), p)

        # save_start = false, save_everystep=false
        Zdu0, Zdp = Zygote.gradient((u0, p) -> loss(u0, p,
                sensealg = ForwardDiffSensitivity()),
            u0, p)
        @test du0≈Zdu0 atol=1e-4
        @test dp≈Zdp atol=1e-4
        Zdu0, Zdp = Zygote.gradient((u0, p) -> loss(u0, p, sensealg = BacksolveAdjoint()),
            u0,
            p)
        @test du0≈Zdu0 atol=1e-4
        @test dp≈Zdp atol=1e-4
        Zdu0, Zdp = Zygote.gradient((u0, p) -> loss(u0, p,
                sensealg = InterpolatingAdjoint()),
            u0, p)
        @test du0≈Zdu0 atol=1e-4
        @test dp≈Zdp atol=1e-4

        # save_start = true, save_everystep=false
        Zdu0, Zdp = Zygote.gradient((u0, p) -> loss(u0, p,
                sensealg = ForwardDiffSensitivity(),
                save_start = true), u0, p)
        @test du0≈Zdu0 atol=1e-4
        @test dp≈Zdp atol=1e-4
        Zdu0, Zdp = Zygote.gradient((u0, p) -> loss(u0, p, sensealg = BacksolveAdjoint(),
                save_start = true), u0, p)
        @test du0≈Zdu0 atol=1e-4
        @test dp≈Zdp atol=1e-4
        Zdu0, Zdp = Zygote.gradient((u0, p) -> loss(u0, p,
                sensealg = InterpolatingAdjoint(),
                save_start = true), u0, p)
        @test du0≈Zdu0 atol=1e-4
        @test dp≈Zdp atol=1e-4

        # save_start = true, save_everystep=true
        Zdu0, Zdp = Zygote.gradient((u0, p) -> loss(u0, p,
                sensealg = ForwardDiffSensitivity(),
                save_start = true,
                save_everystep = true),
            u0, p)
        @test du0≈Zdu0 atol=1e-4
        @test dp≈Zdp atol=1e-4

        Zdu0, Zdp = Zygote.gradient((u0, p) -> loss(u0, p, sensealg = BacksolveAdjoint(),
                save_start = true,
                save_everystep = true),
            u0, p)
        @test du0≈Zdu0 atol=1e-4
        @test dp≈Zdp atol=1e-4
        Zdu0, Zdp = Zygote.gradient((u0, p) -> loss(u0, p,
                sensealg = InterpolatingAdjoint(),
                save_start = true,
                save_everystep = true),
            u0, p)
        @test du0≈Zdu0 atol=1e-4
        @test dp≈Zdp atol=1e-4
        # QuadratureAdjoint makes sense only in this case, otherwise Zdp fails
        Zdu0, Zdp = Zygote.gradient((u0, p) -> loss(u0, p, sensealg = QuadratureAdjoint(),
                save_start = true,
                save_everystep = true),
            u0, p)
        @test du0≈Zdu0 atol=1e-4
        @test dp≈Zdp atol=1e-4

        function loss2(u0, p; sensealg = nothing, saveat = 1.0)
            # remake tspan so saveat::Number makes sense
            _prob = remake(prob, tspan = (0.0, 100.0), u0 = u0, p = p)
            # saving arguments can have a huge influence here
            sol = solve(_prob, Tsit5(), reltol = tol, abstol = tol, sensealg = sensealg,
                callback = cb_t, saveat = saveat)
            res = sol.u[end]
            g(res, p, nothing)
        end

        du0 = ForwardDiff.gradient((u0) -> loss2(u0, p), u0)
        dp = ForwardDiff.gradient((p) -> loss2(u0, p), p)

        # saveat::Number
        Zdu0, Zdp = Zygote.gradient((u0, p) -> loss2(u0, p,
                sensealg = ForwardDiffSensitivity()),
            u0, p)
        @test du0≈Zdu0 atol=1e-4
        @test dp≈Zdp atol=1e-4
        Zdu0, Zdp = Zygote.gradient((u0, p) -> loss2(u0, p, sensealg = BacksolveAdjoint()),
            u0,
            p)
        @test du0≈Zdu0 atol=1e-4
        @test dp≈Zdp atol=1e-4
        Zdu0, Zdp = Zygote.gradient((u0, p) -> loss2(u0, p,
                sensealg = InterpolatingAdjoint()),
            u0, p)
        @test du0≈Zdu0 atol=1e-4
        @test dp≈Zdp atol=1e-4
        Zdu0, Zdp = Zygote.gradient((u0, p) -> loss2(u0, p, sensealg = QuadratureAdjoint()),
            u0, p)
        @test du0≈Zdu0 atol=1e-4
        @test dp≈Zdp atol=1e-4
    end
end

@testset "High Level Interface to Control Steady State Adjoint Internals" begin
    u0 = zeros(32)
    p = [2.0, 1.0]

    # Diagonal Jacobian Problem
    prob = NonlinearProblem((u, p) -> u .- p[1] .+ p[2], u0, p)
    solve1 = solve(remake(prob, p = p), NewtonRaphson())

    function test_loss(p, prob; alg = NewtonRaphson(),
        sensealg = SteadyStateAdjoint(autojacvec = ZygoteVJP()))
        _prob = remake(prob, p = p)
        sol = sum(solve(_prob, alg; sensealg))
        return sol
    end

    test_loss(p, prob)

    dp1 = Zygote.gradient(p -> test_loss(p, prob), p)[1]

    @test dp1[1] ≈ 32
    @test dp1[2] ≈ -32

    for uniform_blocked_diagonal_jacobian in (true, false),
        linsolve_method in (SSAdjointFullJacobianLinsolve(),
            SSAdjointIterativeVJPLinsolve(), SSAdjointHeuristicLinsolve()),
        concrete_jac in (true, false)

        sensealg = SteadyStateAdjoint(; autojacvec = ZygoteVJP(),
            uniform_blocked_diagonal_jacobian, linsolve_method, concrete_jac)
        test_loss(p, prob; sensealg)
        dp1 = Zygote.gradient(p -> test_loss(p, prob; sensealg), p)[1]

        @test dp1[1] ≈ 32
        @test dp1[2] ≈ -32
    end

    # Inplace version
    prob = NonlinearProblem((du, u, p) -> (du .= u .- p[1] .+ p[2]), u0, p)

    function test_loss(p, prob; alg = NewtonRaphson(), sensealg = SteadyStateAdjoint())
        _prob = remake(prob, p = p)
        sol = sum(solve(_prob, alg; sensealg))
        return sol
    end

    test_loss(p, prob)

    dp1 = Zygote.gradient(p -> test_loss(p, prob), p)[1]

    @test dp1[1] ≈ 32
    @test dp1[2] ≈ -32

    for uniform_blocked_diagonal_jacobian in (true, false),
        linsolve_method in (SSAdjointFullJacobianLinsolve(),
            SSAdjointIterativeVJPLinsolve(), SSAdjointHeuristicLinsolve()),
        concrete_jac in (true, false)

        sensealg = SteadyStateAdjoint(; uniform_blocked_diagonal_jacobian, linsolve_method,
            concrete_jac)
        test_loss(p, prob; sensealg)
        dp1 = Zygote.gradient(p -> test_loss(p, prob; sensealg), p)[1]

        @test dp1[1] ≈ 32
        @test dp1[2] ≈ -32
    end
end
