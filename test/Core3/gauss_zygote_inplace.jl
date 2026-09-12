using SciMLSensitivity, OrdinaryDiffEq, Zygote, SciMLBase, ForwardDiff
using OrdinaryDiffEqSDIRK: KenCarp4
using Test

# Test for issue #1282: GaussAdjoint with ZygoteVJP should handle in-place ODE functions
@testset "GaussAdjoint with ZygoteVJP and in-place ODE" begin
    function fiip(du, u, p, t)
        du[1] = p[1] * u[1] - p[2] * u[1] * u[2]
        du[2] = -p[3] * u[2] + p[4] * u[1] * u[2]
        return nothing
    end

    p = [1.5, 1.0, 3.0, 1.0]
    u0 = [1.0; 1.0]
    prob = ODEProblem{true, SciMLBase.FullSpecialize}(
        ODEFunction{true, SciMLBase.FullSpecialize}(fiip),
        u0, (0.0, 10.0), p
    )

    # Test that basic solve works
    sol = solve(prob, KenCarp4(), sensealg = GaussAdjoint())
    @test sol.retcode == ReturnCode.Success

    # Test that gradient computation with ZygoteVJP works
    loss(
        u0,
        p
    ) = sum(
        solve(
            prob, KenCarp4(), u0 = u0, p = p, saveat = 0.1,
            sensealg = GaussAdjoint(autojacvec = ZygoteVJP())
        )
    )

    # This should not throw MethodError anymore
    du0, dp = Zygote.gradient(loss, u0, p)

    @test du0 !== nothing
    @test dp !== nothing
    @test length(du0) == 2
    @test length(dp) == 4

    # Test with explicit ZygoteVJP specification
    (dp2,) = Zygote.gradient(p) do p
        sum(
            solve(
                prob, KenCarp4(), p = p, saveat = 0.1,
                sensealg = GaussAdjoint(autojacvec = ZygoteVJP())
            )
        )
    end

    @test dp2 !== nothing
    @test length(dp2) == 4
end

# Test out-of-place still works
@testset "GaussAdjoint with ZygoteVJP and out-of-place ODE" begin
    function foop(u, p, t)
        dx = p[1] * u[1] - p[2] * u[1] * u[2]
        dy = -p[3] * u[2] + p[4] * u[1] * u[2]
        [dx, dy]
    end

    p = [1.5, 1.0, 3.0, 1.0]
    u0 = [1.0; 1.0]
    prob = ODEProblem(foop, u0, (0.0, 10.0), p)

    # Test that gradient computation with ZygoteVJP works for out-of-place
    loss(
        u0,
        p
    ) = sum(
        solve(
            prob, Tsit5(), u0 = u0, p = p, saveat = 0.1,
            sensealg = GaussAdjoint(autojacvec = ZygoteVJP())
        )
    )

    du0, dp = Zygote.gradient(loss, u0, p)

    @test du0 !== nothing
    @test dp !== nothing
    @test length(du0) == 2
    @test length(dp) == 4
end

# Test for issue #1462: GaussAdjoint(EnzymeVJP()) on an out-of-place ODE hit an
# `UndefVarError: gclosure3` because the out-of-place branch of `vec_pjac!`
# referenced an undefined variable. The gradient must match both the ZygoteVJP
# path and ForwardDiff.
@testset "GaussAdjoint with EnzymeVJP and out-of-place ODE (#1462)" begin
    function foop(u, p, t)
        dx = p[1] * u[1] - p[2] * u[1] * u[2]
        dy = -p[3] * u[2] + p[4] * u[1] * u[2]
        [dx, dy]
    end

    p = [1.5, 1.0, 3.0, 1.0]
    u0 = [1.0, 1.0]
    prob = ODEProblem(foop, u0, (0.0, 10.0), p)

    enzyme_loss(p) = sum(
        solve(
            prob, Tsit5(); p = p, saveat = 0.1, abstol = 1.0e-10, reltol = 1.0e-10,
            sensealg = GaussAdjoint(autojacvec = EnzymeVJP())
        )
    )
    zygote_loss(p) = sum(
        solve(
            prob, Tsit5(); p = p, saveat = 0.1, abstol = 1.0e-10, reltol = 1.0e-10,
            sensealg = GaussAdjoint(autojacvec = ZygoteVJP())
        )
    )
    fd_loss(p) = sum(
        solve(
            remake(prob; p = p), Tsit5(); saveat = 0.1, abstol = 1.0e-12, reltol = 1.0e-12
        )
    )

    dp_enzyme = Zygote.gradient(enzyme_loss, p)[1]
    dp_zygote = Zygote.gradient(zygote_loss, p)[1]
    dp_fd = ForwardDiff.gradient(fd_loss, p)

    @test dp_enzyme !== nothing
    @test length(dp_enzyme) == 4
    @test dp_enzyme ≈ dp_zygote rtol = 1.0e-6
    @test dp_enzyme ≈ dp_fd rtol = 1.0e-6
end

# Regression tests for https://github.com/SciML/SciMLSensitivity.jl/issues/994,
# #995, #996: GaussAdjoint edge cases (non-exact-zero sensitivities, `nothing`
# gradients, complex state).
@testset "GaussAdjoint edge cases" begin
    dynamics(x, p, t) = x

    # #994 / #995: parameters unused by the dynamics — gradient must not error
    # and should be exactly zero.
    function zero_loss(params; autojacvec)
        u0 = zeros(2)
        problem = ODEProblem(dynamics, u0, (0.0, 1.0), params)
        rollout = solve(
            problem, Tsit5(), u0 = u0, p = params,
            sensealg = GaussAdjoint(autojacvec = autojacvec)
        )
        return sum(Array(rollout)[:, end])
    end

    dp_nothing = Zygote.gradient(
        p -> zero_loss(p; autojacvec = ZygoteVJP(allow_nothing = true)), [1.0]
    )[1]
    dp_plain = Zygote.gradient(
        p -> zero_loss(p; autojacvec = ZygoteVJP()), [1.0]
    )[1]
    @test dp_nothing === nothing || all(iszero, dp_nothing)
    @test dp_plain === nothing || all(iszero, dp_plain)

    # #996: complex-valued state.
    dynamics_c(x, p, t) = -x * p[1]
    function loss2_c(params)
        u0 = [1.0 + 1.0im]
        problem = ODEProblem(dynamics_c, u0, (0.0, 1.0), params)
        rollout = solve(
            problem, Tsit5(), u0 = u0, p = params,
            sensealg = GaussAdjoint(autodiff = false, autojacvec = false)
        )
        return abs(sum(Array(rollout)[:, end]))
    end
    function loss2_interp(params)
        u0 = [1.0 + 1.0im]
        problem = ODEProblem(dynamics_c, u0, (0.0, 1.0), params)
        rollout = solve(
            problem, Tsit5(), u0 = u0, p = params,
            sensealg = InterpolatingAdjoint(autodiff = false, autojacvec = false)
        )
        return abs(sum(Array(rollout)[:, end]))
    end
    dp_gauss = Zygote.gradient(loss2_c, ones(1))[1]
    dp_interp = Zygote.gradient(loss2_interp, ones(1))[1]
    @test dp_gauss !== nothing && all(isfinite, dp_gauss)
    @test dp_gauss ≈ dp_interp rtol = 1.0e-2
end
