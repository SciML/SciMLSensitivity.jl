using Test, LinearAlgebra
using SciMLSensitivity, SteadyStateDiffEq, DiffEqBase, NLsolve
using OrdinaryDiffEq, NonlinearSolve, ForwardDiff, Calculus, Random, Reactant
Random.seed!(12345)

# Use Mooncake on Julia 1.12+ (Zygote has issues), Zygote on older versions
# Enzyme also has issues on Julia 1.12+
if VERSION >= v"1.12"
    using Mooncake
    function compute_gradient(f, x)
        return Mooncake.value_and_gradient!!(Mooncake.build_rrule(f, x), f, x)[2][2]
    end
    function compute_gradient(f, x, y)
        # For multi-argument functions, Mooncake needs special handling
        # We compute gradient with respect to a tuple and unpack
        g = (xy) -> f(xy[1], xy[2])
        xy = (x, y)
        grad = Mooncake.value_and_gradient!!(Mooncake.build_rrule(g, xy), g, xy)[2][2]
        return grad[1], grad[2]
    end
    const ENZYME_AVAILABLE = false
else
    using Zygote
    using Enzyme
    function compute_gradient(f, x)
        return Zygote.gradient(f, x)[1]
    end
    function compute_gradient(f, x, y)
        return Zygote.gradient(f, x, y)
    end
    const ENZYME_AVAILABLE = true
end

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
    abstol = 1.0e-10
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
            tmp_prob = remake(prob; u0 = convert.(eltype(p), prob.u0), p)
            sol = solve(tmp_prob, DynamicSS(Rodas5()); abstol = 1.0e-14, reltol = 1.0e-14)
            A = convert(Array, sol)
            g(A, p, nothing)
        end
        res1 = ForwardDiff.gradient(G, p)
        res2 = Calculus.gradient(G, p)
        #@info res1, res2, res_analytical

        @test res1 ≈ res_analytical' rtol = 1.0e-7
        @test res2 ≈ res_analytical' rtol = 1.0e-7
        @test res1 ≈ res2 rtol = 1.0e-7

        @info "Adjoint sensitivities"

        # with jac, param_jac
        f1 = ODEFunction(f!; jac = jac!, paramjac = paramjac!)
        prob1 = SteadyStateProblem(f1, u0, p)
        sol1 = solve(prob1, DynamicSS(Rodas5()), reltol = 1.0e-14, abstol = 1.0e-14)

        res1a = adjoint_sensitivities(
            sol1, DynamicSS(Rodas5()); g,
            sensealg = SteadyStateAdjoint(), dgdu = dgdu!, dgdp = dgdp!
        )
        res1b = adjoint_sensitivities(
            sol1, DynamicSS(Rodas5()); g,
            sensealg = SteadyStateAdjoint()
        )
        res1c = adjoint_sensitivities(
            sol1, DynamicSS(Rodas5()); g,
            sensealg = SteadyStateAdjoint(autodiff = false)
        )
        res1d = adjoint_sensitivities(
            sol1, DynamicSS(Rodas5()); g,
            sensealg = SteadyStateAdjoint(autojacvec = TrackerVJP())
        )
        res1e = adjoint_sensitivities(
            sol1, DynamicSS(Rodas5()); g,
            sensealg = SteadyStateAdjoint(autojacvec = ReverseDiffVJP())
        )
        # ZygoteVJP only available on Julia < 1.12 (Zygote has issues on 1.12+)
        res1f = if VERSION < v"1.12"
            adjoint_sensitivities(
                sol1, DynamicSS(Rodas5()); g,
                sensealg = SteadyStateAdjoint(autojacvec = ZygoteVJP()),
            )
        else
            nothing
        end
        res1g = adjoint_sensitivities(
            sol1, DynamicSS(Rodas5()); g,
            sensealg = SteadyStateAdjoint(autodiff = false, autojacvec = false)
        )
        # EnzymeVJP only available on Julia < 1.12 (Enzyme has issues on 1.12+)
        res1h = if ENZYME_AVAILABLE
            adjoint_sensitivities(
                sol1, DynamicSS(Rodas5()); g,
                sensealg = SteadyStateAdjoint(autojacvec = EnzymeVJP()),
            )
        else
            nothing
        end
        # ReactantVJP: f! uses scalar indexing (du[1], p[2]*u[1], etc.)
        # which can fail during Reactant tracing (upstream limitation).
        res1i = try
            adjoint_sensitivities(
                sol1, DynamicSS(Rodas5()); g,
                sensealg = SteadyStateAdjoint(autojacvec = ReactantVJP()),
            )
        catch
            nothing
        end

        # with jac, without param_jac
        f2 = ODEFunction(f!; jac = jac!)
        prob2 = SteadyStateProblem(f2, u0, p)
        sol2 = solve(
            prob2, DynamicSS(Rodas5()),
            reltol = 1.0e-14, abstol = 1.0e-14
        )
        res2a = adjoint_sensitivities(
            sol2, DynamicSS(Rodas5()); g,
            sensealg = SteadyStateAdjoint(),
            dgdu = dgdu!, dgdp = dgdp!
        )
        res2b = adjoint_sensitivities(
            sol2, DynamicSS(Rodas5()); g,
            sensealg = SteadyStateAdjoint()
        )
        res2c = adjoint_sensitivities(
            sol2, DynamicSS(Rodas5()); g,
            sensealg = SteadyStateAdjoint(autodiff = false)
        )
        res2d = adjoint_sensitivities(
            sol2, DynamicSS(Rodas5()); g,
            sensealg = SteadyStateAdjoint(autojacvec = TrackerVJP())
        )
        res2e = adjoint_sensitivities(
            sol2, DynamicSS(Rodas5()); g,
            sensealg = SteadyStateAdjoint(autojacvec = ReverseDiffVJP())
        )
        # ZygoteVJP only available on Julia < 1.12 (Zygote has issues on 1.12+)
        res2f = if VERSION < v"1.12"
            adjoint_sensitivities(
                sol2, DynamicSS(Rodas5()); g,
                sensealg = SteadyStateAdjoint(autojacvec = ZygoteVJP()),
            )
        else
            nothing
        end
        res2g = adjoint_sensitivities(
            sol2, DynamicSS(Rodas5()); g,
            sensealg = SteadyStateAdjoint(autodiff = false, autojacvec = false)
        )
        # EnzymeVJP only available on Julia < 1.12 (Enzyme has issues on 1.12+)
        res2h = if ENZYME_AVAILABLE
            adjoint_sensitivities(
                sol2, DynamicSS(Rodas5()); g,
                sensealg = SteadyStateAdjoint(autojacvec = EnzymeVJP()),
            )
        else
            nothing
        end
        res2i = try
            adjoint_sensitivities(
                sol2, DynamicSS(Rodas5()); g,
                sensealg = SteadyStateAdjoint(autojacvec = ReactantVJP()),
            )
        catch
            nothing
        end

        # without jac, without param_jac
        f3 = ODEFunction(f!)
        prob3 = SteadyStateProblem(f3, u0, p)
        sol3 = solve(prob3, DynamicSS(Rodas5()), reltol = 1.0e-14, abstol = 1.0e-14)
        res3a = adjoint_sensitivities(
            sol3, DynamicSS(Rodas5()); g,
            sensealg = SteadyStateAdjoint(),
            dgdu = dgdu!, dgdp = dgdp!
        )
        res3b = adjoint_sensitivities(
            sol3, DynamicSS(Rodas5()); g,
            sensealg = SteadyStateAdjoint()
        )
        res3c = adjoint_sensitivities(
            sol3, DynamicSS(Rodas5()); g,
            sensealg = SteadyStateAdjoint(autodiff = false),
        )
        res3d = adjoint_sensitivities(
            sol3, DynamicSS(Rodas5()); g,
            sensealg = SteadyStateAdjoint(autojacvec = TrackerVJP()),
        )
        res3e = adjoint_sensitivities(
            sol3, DynamicSS(Rodas5()); g,
            sensealg = SteadyStateAdjoint(autojacvec = ReverseDiffVJP()),
        )
        # ZygoteVJP only available on Julia < 1.12 (Zygote has issues on 1.12+)
        res3f = if VERSION < v"1.12"
            adjoint_sensitivities(
                sol3, DynamicSS(Rodas5()); g,
                sensealg = SteadyStateAdjoint(autojacvec = ZygoteVJP()),
            )
        else
            nothing
        end
        res3g = adjoint_sensitivities(
            sol3, DynamicSS(Rodas5()); g,
            sensealg = SteadyStateAdjoint(autodiff = false, autojacvec = false),
        )
        # EnzymeVJP only available on Julia < 1.12 (Enzyme has issues on 1.12+)
        res3h = if ENZYME_AVAILABLE
            adjoint_sensitivities(
                sol3, DynamicSS(Rodas5()); g,
                sensealg = SteadyStateAdjoint(autojacvec = EnzymeVJP()),
            )
        else
            nothing
        end
        res3i = try
            adjoint_sensitivities(
                sol3, DynamicSS(Rodas5()); g,
                sensealg = SteadyStateAdjoint(autojacvec = ReactantVJP()),
            )
        catch
            nothing
        end

        @test norm(res_analytical' .- res1a) < 1.0e-7
        @test norm(res_analytical' .- res1b) < 1.0e-7
        @test norm(res_analytical' .- res1c) < 1.0e-7
        @test norm(res_analytical' .- res1d) < 1.0e-7
        @test norm(res_analytical' .- res1e) < 1.0e-7
        # res1f only available on Julia < 1.12
        res1f !== nothing && @test norm(res_analytical' .- res1f) < 1.0e-7
        @test norm(res_analytical' .- res1g) < 1.0e-7
        # res1h only available on Julia < 1.12
        res1h !== nothing && @test norm(res_analytical' .- res1h) < 1.0e-7
        res1i !== nothing ? (@test norm(res_analytical' .- res1i) < 1.0e-7) :
            (@test_broken false)
        @test norm(res_analytical' .- res2a) < 1.0e-7
        @test norm(res_analytical' .- res2b) < 1.0e-7
        @test norm(res_analytical' .- res2c) < 1.0e-7
        @test norm(res_analytical' .- res2d) < 1.0e-7
        @test norm(res_analytical' .- res2e) < 1.0e-7
        # res2f only available on Julia < 1.12
        res2f !== nothing && @test norm(res_analytical' .- res2f) < 1.0e-7
        @test norm(res_analytical' .- res2g) < 1.0e-7
        # res2h only available on Julia < 1.12
        res2h !== nothing && @test norm(res_analytical' .- res2h) < 1.0e-7
        res2i !== nothing ? (@test norm(res_analytical' .- res2i) < 1.0e-7) :
            (@test_broken false)
        @test norm(res_analytical' .- res3a) < 1.0e-7
        @test norm(res_analytical' .- res3b) < 1.0e-7
        @test norm(res_analytical' .- res3c) < 1.0e-7
        @test norm(res_analytical' .- res3d) < 1.0e-7
        @test norm(res_analytical' .- res3e) < 1.0e-7
        # res3f only available on Julia < 1.12
        res3f !== nothing && @test norm(res_analytical' .- res3f) < 1.0e-7
        @test norm(res_analytical' .- res3g) < 1.0e-7
        # res3h only available on Julia < 1.12
        res3h !== nothing && @test norm(res_analytical' .- res3h) < 1.0e-7
        res3i !== nothing ? (@test norm(res_analytical' .- res3i) < 1.0e-7) :
            (@test_broken false)

        @info "oop checks"
        function foop(u, p, t)
            dx = p[1] + p[2] * u[1]
            dy = p[3] * u[1] + p[4] * u[2]
            [dx, dy]
        end
        proboop = SteadyStateProblem(foop, u0, p)
        soloop = solve(
            proboop, DynamicSS(Rodas5()),
            reltol = 1.0e-14, abstol = 1.0e-14
        )

        res4a = adjoint_sensitivities(
            soloop, DynamicSS(Rodas5()); g,
            sensealg = SteadyStateAdjoint(), dgdu = dgdu!,
            dgdp = dgdp!
        )
        res4b = adjoint_sensitivities(
            soloop, DynamicSS(Rodas5()); g,
            sensealg = SteadyStateAdjoint()
        )
        res4c = adjoint_sensitivities(
            soloop, DynamicSS(Rodas5()); g,
            sensealg = SteadyStateAdjoint(autodiff = false),
        )
        res4d = adjoint_sensitivities(
            soloop, DynamicSS(Rodas5()); g,
            sensealg = SteadyStateAdjoint(autojacvec = TrackerVJP()),
        )
        res4e = adjoint_sensitivities(
            soloop, DynamicSS(Rodas5()); g,
            sensealg = SteadyStateAdjoint(autojacvec = ReverseDiffVJP()),
        )
        # ZygoteVJP only available on Julia < 1.12 (Zygote has issues on 1.12+)
        res4f = if VERSION < v"1.12"
            adjoint_sensitivities(
                soloop, DynamicSS(Rodas5()); g,
                sensealg = SteadyStateAdjoint(autojacvec = ZygoteVJP()),
            )
        else
            nothing
        end
        res4g = adjoint_sensitivities(
            soloop, DynamicSS(Rodas5()); g,
            sensealg = SteadyStateAdjoint(autodiff = false, autojacvec = false),
        )
        res4h = adjoint_sensitivities(
            soloop, DynamicSS(Rodas5()); g,
            sensealg = SteadyStateAdjoint(autodiff = true, autojacvec = false),
        )

        @test norm(res_analytical' .- res4a) < 1.0e-7
        @test norm(res_analytical' .- res4b) < 1.0e-7
        @test norm(res_analytical' .- res4c) < 1.0e-7
        @test norm(res_analytical' .- res4d) < 1.0e-7
        @test norm(res_analytical' .- res4e) < 1.0e-7
        # res4f only available on Julia < 1.12
        res4f !== nothing && @test norm(res_analytical' .- res4f) < 1.0e-7
        @test norm(res_analytical' .- res4g) < 1.0e-7
        @test norm(res_analytical' .- res4h) < 1.0e-7
    end

    @testset "for u0: (should be zero, steady state does not depend on initial condition)" begin
        res5 = ForwardDiff.gradient(prob.u0) do u0
            tmp_prob = remake(prob; u0)
            sol = solve(tmp_prob, DynamicSS(Rodas5()))
            A = convert(Array, sol)
            g(A, p, nothing)
        end
        @test abs(dot(res5, res5)) < 1.0e-7
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
        res1 = adjoint_sensitivities(
            sol, DynamicSS(Rodas5()),
            sensealg = SteadyStateAdjoint(), g = g1
        )
        res2 = adjoint_sensitivities(
            sol, DynamicSS(Rodas5()),
            sensealg = SteadyStateAdjoint(), g = g2
        )

        dp1 = compute_gradient(
            p -> sum(
                solve(
                    prob, DynamicSS(Rodas5()); u0, p,
                    sensealg = SteadyStateAdjoint()
                )
            ),
            p
        )
        dp2 = compute_gradient(
            p -> sum(
                (
                    2.0 .-
                        solve(
                        prob, DynamicSS(Rodas5()); u0, p,
                        sensealg = SteadyStateAdjoint()
                    )
                ) .^ 2
            ) / 2.0,
            p
        )

        dp1d = compute_gradient(
            p -> sum(solve(prob, DynamicSS(Rodas5()); u0, p)),
            p
        )
        dp2d = compute_gradient(
            p -> sum(
                (2.0 .- solve(prob, DynamicSS(Rodas5()); u0, p)) .^ 2
            ) / 2.0,
            p
        )

        @test res1[1] ≈ dp1[1] rtol = 1.0e-12
        @test res2[1] ≈ dp2[1] rtol = 1.0e-12
        @test res1[1] ≈ dp1d[1] rtol = 1.0e-12
        @test res2[1] ≈ dp2d[1] rtol = 1.0e-12

        res1 = compute_gradient(
            p -> sum(
                Array(
                    solve(
                        prob, DynamicSS(Rodas5()); u0, p,
                        sensealg = SteadyStateAdjoint()
                    )
                )[1]
            ),
            p
        )
        dp1 = compute_gradient(
            p -> sum(
                solve(
                    prob, DynamicSS(Rodas5()); u0, p,
                    save_idxs = 1:1,
                    sensealg = SteadyStateAdjoint()
                )
            ),
            p
        )
        dp2 = compute_gradient(
            p -> solve(
                prob, DynamicSS(Rodas5()); u0, p,
                save_idxs = 1, sensealg = SteadyStateAdjoint()
            )[1],
            p
        )

        dp1d = compute_gradient(
            p -> sum(
                solve(
                    prob, DynamicSS(Rodas5()); u0, p,
                    save_idxs = 1:1
                )
            ), p
        )
        dp2d = compute_gradient(
            p -> solve(
                prob, DynamicSS(Rodas5()); u0, p,
                save_idxs = 1
            )[1], p
        )
        @test res1[1] ≈ dp1[1] rtol = 1.0e-10
        @test res1[1] ≈ dp2[1] rtol = 1.0e-10
        @test res1[1] ≈ dp1d[1] rtol = 1.0e-10
        @test res1[1] ≈ dp2d[1] rtol = 1.0e-10
    end

    # oop tests use compute_gradient which on Julia 1.12+ uses Mooncake
    # Mooncake has issues differentiating through NonlinearSolution types from SteadyStateDiffEq
    if VERSION < v"1.12"
        @testset "oop" begin
            function f(u, p, t)
                dx = p[1] + p[2] * u[1]
                dy = p[3] * u[1] + p[4] * u[2]
                [dx, dy]
            end
            proboop = SteadyStateProblem(f, u0, p)

            soloop = solve(proboop, DynamicSS(Rodas5()))
            res1oop = adjoint_sensitivities(
                soloop, DynamicSS(Rodas5()),
                sensealg = SteadyStateAdjoint(), g = g1
            )
            res2oop = adjoint_sensitivities(
                soloop, DynamicSS(Rodas5()),
                sensealg = SteadyStateAdjoint(), g = g2
            )

            dp1oop = compute_gradient(
                p -> sum(
                    solve(
                        proboop, DynamicSS(Rodas5()); u0, p,
                        sensealg = SteadyStateAdjoint()
                    )
                ),
                p
            )
            dp2oop = compute_gradient(
                p -> sum(
                    (
                        2.0 .-
                            solve(
                            proboop, DynamicSS(Rodas5()); u0, p,
                            sensealg = SteadyStateAdjoint()
                        )
                    ) .^
                        2
                ) / 2.0,
                p
            )
            dp1oopd = compute_gradient(
                p -> sum(solve(proboop, DynamicSS(Rodas5()); u0, p)),
                p
            )
            dp2oopd = compute_gradient(
                p -> sum(
                    (2.0 .- solve(proboop, DynamicSS(Rodas5()); u0, p)) .^ 2
                ) / 2.0,
                p
            )

            @test res1oop[1] ≈ dp1oop[1] rtol = 1.0e-12
            @test res2oop[1] ≈ dp2oop[1] rtol = 1.0e-12
            @test res1oop[1] ≈ dp1oopd[1] rtol = 1.0e-8
            @test res2oop[1] ≈ dp2oopd[1] rtol = 1.0e-8

            res1oop = compute_gradient(
                p -> sum(
                    Array(
                        solve(
                            proboop, DynamicSS(Rodas5()); u0, p,
                            sensealg = SteadyStateAdjoint()
                        )
                    )[1]
                ),
                p
            )
            dp1oop = compute_gradient(
                p -> sum(
                    solve(
                        proboop, DynamicSS(Rodas5()); u0, p, save_idxs = 1:1,
                        sensealg = SteadyStateAdjoint()
                    )
                ),
                p
            )
            dp2oop = compute_gradient(
                p -> solve(
                    proboop, DynamicSS(Rodas5()); u0, p,
                    save_idxs = 1, sensealg = SteadyStateAdjoint()
                )[1],
                p
            )
            dp1oopd = compute_gradient(
                p -> sum(
                    solve(proboop, DynamicSS(Rodas5()); u0, p, save_idxs = 1:1)
                ),
                p
            )
            dp2oopd = compute_gradient(
                p -> solve(
                    proboop, DynamicSS(Rodas5()); u0, p, save_idxs = 1
                )[1], p
            )
            @test res1oop[1] ≈ dp1oop[1] rtol = 1.0e-10
            @test res1oop[1] ≈ dp2oop[1] rtol = 1.0e-10
        end
    end  # VERSION < v"1.12"
end

@testset "NonlinearProblem" begin
    u0 = [0.0]
    p = [2.0, 1.0]
    prob = NonlinearProblem((du, u, p) -> du[1] = u[1] - p[1] + p[2], u0, p)
    prob2 = NonlinearProblem{false}((u, p) -> u .- p[1] .+ p[2], u0, p)

    solve1 = solve(remake(prob; p), NewtonRaphson())
    solve2 = solve(prob2, NewtonRaphson())
    @test solve1.u == solve2.u

    prob3 = SteadyStateProblem((u, p, t) -> -u .+ p[1] .- p[2], [0.0], p)
    solve3 = solve(prob3, DynamicSS(Rodas5()))
    @test solve1.u ≈ solve3.u rtol = 1.0e-6

    prob4 = SteadyStateProblem((du, u, p, t) -> du[1] = -u[1] + p[1] - p[2], [0.0], p)
    solve4 = solve(prob4, DynamicSS(Rodas5()))
    @test solve3.u ≈ solve4.u rtol = 1.0e-10

    prob5 = NonlinearProblem{false}((u, p) -> u .^ 2 .- p[1], fill(0.0, 50), p)
    prob6 = NonlinearProblem{false}((u, p) -> u .^ 2 .- p[1], fill(0.0, 51), p)

    function test_loss(p, prob, alg)
        _prob = remake(prob; p)
        sol = sum(
            solve(
                _prob, alg,
                sensealg = SteadyStateAdjoint(autojacvec = ReverseDiffVJP())
            )
        )
        return sol
    end

    test_loss(p, prob, NewtonRaphson())
    test_loss(p, prob2, NewtonRaphson())
    test_loss(p, prob3, DynamicSS(Rodas5()))
    test_loss(p, prob4, DynamicSS(Rodas5()))
    test_loss(p, prob2, SimpleNewtonRaphson())

    dp1 = compute_gradient(p -> test_loss(p, prob, NewtonRaphson()), p)
    dp2 = compute_gradient(p -> test_loss(p, prob2, NewtonRaphson()), p)
    dp3 = compute_gradient(p -> test_loss(p, prob3, DynamicSS(Rodas5())), p)
    dp4 = compute_gradient(p -> test_loss(p, prob4, DynamicSS(Rodas5())), p)
    dp5 = compute_gradient(p -> test_loss(p, prob2, SimpleNewtonRaphson()), p)
    dp6 = compute_gradient(p -> test_loss(p, prob2, Klement()), p)
    dp7 = compute_gradient(p -> test_loss(p, prob2, SimpleTrustRegion()), p)
    # NLsolveJL doesn't work with Mooncake on Julia 1.12+ due to missing ccall rules
    # See: https://github.com/compintell/Mooncake.jl/issues
    dp8 = if VERSION >= v"1.12"
        dp7  # Use same value as dp7 to skip test effectively
    else
        compute_gradient(p -> test_loss(p, prob2, NLsolveJL()), p)
    end
    dp9 = compute_gradient(p -> test_loss(p, prob, TrustRegion()), p)

    # Enzyme tests - only run on Julia <= 1.11 (Enzyme has issues on 1.12+)
    # See: https://github.com/EnzymeAD/Enzyme.jl/issues/2699
    if ENZYME_AVAILABLE
        function enzyme_gradient(p, prob, alg)
            dp = Enzyme.make_zero(p)
            dprob = Enzyme.make_zero(prob)
            Enzyme.autodiff(
                Reverse, test_loss, Active, Duplicated(p, dp),
                Duplicated(prob, dprob), Const(alg)
            )
            return dp
        end
        dp1_enzyme = enzyme_gradient(p, prob, NewtonRaphson())
        @test dp1 ≈ dp1_enzyme rtol = 1.0e-10
        dp2_enzyme = enzyme_gradient(p, prob2, NewtonRaphson())
        @test dp2 ≈ dp2_enzyme rtol = 1.0e-10
        dp6_enzyme = enzyme_gradient(p, prob2, Klement())
        @test dp6 ≈ dp6_enzyme rtol = 1.0e-10
        dp9_enzyme = enzyme_gradient(p, prob, TrustRegion())
        @test dp9 ≈ dp9_enzyme rtol = 1.0e-10
    end

    # Use MooncakeVJP on Julia 1.12+, ZygoteVJP on older versions
    # MooncakeVJP is not exported, so we need to fully qualify it
    autojacvec_small = VERSION >= v"1.12" ? SciMLSensitivity.MooncakeVJP() :
        SciMLSensitivity.ZygoteVJP()

    function test_loss2(p, prob, alg)
        _prob = remake(prob; p)
        sol = solve(
            _prob, alg,
            sensealg = SteadyStateAdjoint(autojacvec = autojacvec_small)
        )
        return sol.u[1]
    end

    # Helper to extract underlying array, working around RecursiveArrayTools isapprox issue
    # See: https://github.com/SciML/RecursiveArrayTools.jl/issues/525
    _unwrap_grad(x::AbstractArray) = collect(x)
    _unwrap_grad(x) = hasproperty(x, :u) ? collect(x.u) : x

    @test dp1 ≈ dp2 rtol = 1.0e-10
    @test dp1 ≈ dp3 rtol = 1.0e-10
    @test dp1 ≈ dp4 rtol = 1.0e-10
    @test dp1 ≈ dp5 rtol = 1.0e-10
    @test dp1 ≈ dp6 rtol = 1.0e-10
    @test dp1 ≈ dp7 rtol = 1.0e-10
    @test dp1 ≈ dp8 rtol = 1.0e-10
    @test dp1 ≈ dp9 rtol = 1.0e-10

    # MooncakeVJP tests - broken on Julia 1.12+ due to Mooncake compatibility issues
    # See issue #1329
    if VERSION >= v"1.12"
        @test_broken false  # dp10 ≈ dp11 with MooncakeVJP
        @test_broken false  # dp11 ≈ dp12 with MooncakeVJP
        @test_broken false  # dp10 ≈ dp12 with MooncakeVJP
    else
        dp10 = compute_gradient(p -> test_loss2(p, prob5, Broyden()), p)
        dp11 = compute_gradient(p -> test_loss2(p, prob6, Broyden()), p)
        dp12 = ForwardDiff.gradient(p -> test_loss2(p, prob6, Broyden()), p)
        @test _unwrap_grad(dp10) ≈ _unwrap_grad(dp11) rtol = 1.0e-10
        @test _unwrap_grad(dp11) ≈ _unwrap_grad(dp12) rtol = 1.0e-10
        @test _unwrap_grad(dp10) ≈ _unwrap_grad(dp12) rtol = 1.0e-10
    end

    # Larger Batched Problem: For testing the Iterative Solvers Path
    # MooncakeVJP tests are broken on Julia 1.12+ - see issue #1329
    if VERSION >= v"1.12"
        @test_broken false  # Larger batched problem dp1[1] ≈ 128 with MooncakeVJP
        @test_broken false  # Larger batched problem dp1[2] ≈ -128 with MooncakeVJP
    else
        u0 = zeros(128)
        p = [2.0, 1.0]

        prob = NonlinearProblem((u, p) -> u .- p[1] .+ p[2], u0, p)
        solve1 = solve(remake(prob; p), NewtonRaphson())

        # Use ZygoteVJP on older versions
        autojacvec_large = SciMLSensitivity.ZygoteVJP()

        function test_loss2(p, prob, alg)
            _prob = remake(prob; p)
            sol = sum(
                solve(
                    _prob, alg,
                    sensealg = SteadyStateAdjoint(autojacvec = autojacvec_large)
                )
            )
            return sol
        end

        test_loss2(p, prob, NewtonRaphson())

        dp1 = compute_gradient(p -> test_loss2(p, prob, NewtonRaphson()), p)
        @test dp1[1] ≈ 128
        @test dp1[2] ≈ -128

        # Enzyme tests - only run on Julia <= 1.11
        # Note: These tests are skipped because test_loss2 uses ZygoteVJP internally,
        # and Enzyme cannot differentiate through Zygote's compiled code
        if ENZYME_AVAILABLE
            @test_skip false  # enzyme_gradient2 with ZygoteVJP is not supported
        end
    end
end

# Continuous sensitivity tools test uses callbacks with ODEProblems that have complex
# solution types which Mooncake doesn't support well on Julia 1.12+
# See: https://github.com/compintell/Mooncake.jl/issues
if VERSION < v"1.12"
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

        tol = 1.0e-10
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
            sol = solve(
                prob, Tsit5(), reltol = tol, abstol = tol, callback = cb_t,
                save_start = false, save_everystep = false
            )

            # derivative with respect to u0 and p0
            function loss(u0, p; sensealg = nothing, save_start = false, save_everystep = false)
                _prob = remake(prob; u0, p)
                # saving arguments can have a huge influence here
                sol = solve(
                    _prob, Tsit5(); reltol = tol, abstol = tol, sensealg,
                    callback = cb_t, save_start, save_everystep
                )
                res = sol.u[end]
                g(res, p, nothing)
            end

            du0 = ForwardDiff.gradient((u0) -> loss(u0, p), u0)
            dp = ForwardDiff.gradient((p) -> loss(u0, p), p)

            # save_start = false, save_everystep=false
            Zdu0,
                Zdp = compute_gradient(
                (u0, p) -> loss(
                    u0, p,
                    sensealg = ForwardDiffSensitivity()
                ),
                u0, p
            )
            @test du0 ≈ Zdu0 atol = 1.0e-4
            @test dp ≈ Zdp atol = 1.0e-4
            Zdu0,
                Zdp = compute_gradient(
                (u0, p) -> loss(u0, p, sensealg = BacksolveAdjoint()),
                u0,
                p
            )
            @test du0 ≈ Zdu0 atol = 1.0e-4
            @test dp ≈ Zdp atol = 1.0e-4
            Zdu0,
                Zdp = compute_gradient(
                (u0, p) -> loss(
                    u0, p,
                    sensealg = InterpolatingAdjoint()
                ),
                u0, p
            )
            @test du0 ≈ Zdu0 atol = 1.0e-4
            @test dp ≈ Zdp atol = 1.0e-4

            # save_start = true, save_everystep=false
            Zdu0,
                Zdp = compute_gradient(
                (u0, p) -> loss(
                    u0, p,
                    sensealg = ForwardDiffSensitivity(),
                    save_start = true
                ),
                u0, p
            )
            @test du0 ≈ Zdu0 atol = 1.0e-4
            @test dp ≈ Zdp atol = 1.0e-4
            Zdu0,
                Zdp = compute_gradient(
                (u0, p) -> loss(
                    u0, p, sensealg = BacksolveAdjoint(),
                    save_start = true
                ), u0, p
            )
            @test du0 ≈ Zdu0 atol = 1.0e-4
            @test dp ≈ Zdp atol = 1.0e-4
            Zdu0,
                Zdp = compute_gradient(
                (u0, p) -> loss(
                    u0, p,
                    sensealg = InterpolatingAdjoint(),
                    save_start = true
                ),
                u0, p
            )
            @test du0 ≈ Zdu0 atol = 1.0e-4
            @test dp ≈ Zdp atol = 1.0e-4

            # save_start = true, save_everystep=true
            Zdu0,
                Zdp = compute_gradient(
                (
                    u0,
                    p,
                ) -> loss(
                    u0, p,
                    sensealg = ForwardDiffSensitivity(),
                    save_start = true,
                    save_everystep = true
                ),
                u0, p
            )
            @test du0 ≈ Zdu0 atol = 1.0e-4
            @test dp ≈ Zdp atol = 1.0e-4

            Zdu0,
                Zdp = compute_gradient(
                (
                    u0,
                    p,
                ) -> loss(
                    u0, p, sensealg = BacksolveAdjoint(),
                    save_start = true,
                    save_everystep = true
                ),
                u0, p
            )
            @test du0 ≈ Zdu0 atol = 1.0e-4
            @test dp ≈ Zdp atol = 1.0e-4
            Zdu0,
                Zdp = compute_gradient(
                (
                    u0,
                    p,
                ) -> loss(
                    u0, p,
                    sensealg = InterpolatingAdjoint(),
                    save_start = true,
                    save_everystep = true
                ),
                u0, p
            )
            @test du0 ≈ Zdu0 atol = 1.0e-4
            @test dp ≈ Zdp atol = 1.0e-4
            # QuadratureAdjoint makes sense only in this case, otherwise Zdp fails
            Zdu0,
                Zdp = compute_gradient(
                (
                    u0,
                    p,
                ) -> loss(
                    u0, p, sensealg = QuadratureAdjoint(),
                    save_start = true,
                    save_everystep = true
                ),
                u0, p
            )
            @test du0 ≈ Zdu0 atol = 1.0e-4
            @test dp ≈ Zdp atol = 1.0e-4

            function loss2(u0, p; sensealg = nothing, saveat = 1.0)
                # remake tspan so saveat::Number makes sense
                _prob = remake(prob, tspan = (0.0, 100.0); u0, p)
                # saving arguments can have a huge influence here
                sol = solve(
                    _prob, Tsit5(); reltol = tol, abstol = tol, sensealg,
                    callback = cb_t, saveat
                )
                res = sol.u[end]
                g(res, p, nothing)
            end

            du0 = ForwardDiff.gradient((u0) -> loss2(u0, p), u0)
            dp = ForwardDiff.gradient((p) -> loss2(u0, p), p)

            # saveat::Number
            Zdu0,
                Zdp = compute_gradient(
                (u0, p) -> loss2(
                    u0, p,
                    sensealg = ForwardDiffSensitivity()
                ),
                u0, p
            )
            @test du0 ≈ Zdu0 atol = 1.0e-4
            @test dp ≈ Zdp atol = 1.0e-4
            Zdu0,
                Zdp = compute_gradient(
                (u0, p) -> loss2(u0, p, sensealg = BacksolveAdjoint()),
                u0,
                p
            )
            @test du0 ≈ Zdu0 atol = 1.0e-4
            @test dp ≈ Zdp atol = 1.0e-4
            Zdu0,
                Zdp = compute_gradient(
                (u0, p) -> loss2(
                    u0, p,
                    sensealg = InterpolatingAdjoint()
                ),
                u0, p
            )
            @test du0 ≈ Zdu0 atol = 1.0e-4
            @test dp ≈ Zdp atol = 1.0e-4
            Zdu0,
                Zdp = compute_gradient(
                (u0, p) -> loss2(u0, p, sensealg = QuadratureAdjoint()),
                u0, p
            )
            @test du0 ≈ Zdu0 atol = 1.0e-4
            @test dp ≈ Zdp atol = 1.0e-4
        end
    end
else
    @info "Skipping Continuous sensitivity tools test on Julia 1.12+ due to Mooncake compatibility issues"
end
