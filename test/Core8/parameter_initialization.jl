using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using SymbolicIndexingInterface
using SciMLStructures
import SymbolicIndexingInterface as SII
import SciMLStructures as SS
import ModelingToolkit as MTK
using SciMLSensitivity
using OrdinaryDiffEq
using Tracker
using Enzyme
using ForwardDiff
using Zygote
import SciMLBase
using Test

@parameters σ ρ β
@variables x(t) y(t) z(t) w(t)

eqs = [
    D(D(x)) ~ σ * (y - x),
    D(y) ~ x * (ρ - z) - y,
    D(z) ~ x * y - β * z,
    w ~ x + y + z + 2 * β,
]

@mtkbuild sys = ODESystem(
    eqs, t, __legacy_defaults__ = [ρ => missing], guesses = [ρ => 10.0],
)

u0 = [
    D(x) => 2.0,
    x => 1.0,
    y => 0.0,
    z => 0.0,
]

p = [
    σ => 28.0,
    ρ => 10.0,
    β => 8 / 3,
]

tspan = (0.0, 100.0)
prob = ODEProblem(sys, u0, tspan, p, jac = true)
sol = solve(prob, Tsit5())

tunables, repack, _ = SS.canonicalize(SS.Tunable(), parameter_values(prob))

@testset "Adjoint through Parameter Initialization" begin
    @testset "Adjoint through Prob" begin
        sensealg = SciMLSensitivity.GaussAdjoint(
            autojacvec = SciMLSensitivity.EnzymeVJP(),
        )
        gs_prob = Tracker.gradient(tunables) do tunables
            new_prob = remake(prob; p = repack(tunables))
            sol = solve(new_prob; sensealg)
            sum(sol)
        end
        @test any(!iszero, gs_prob[1])

        gs_prob2 = Tracker.gradient(tunables) do tunables
            new_prob = remake(prob; p = repack(tunables))
            sol = solve(
                new_prob; sensealg,
                initializealg = SciMLBase.OverrideInit(), abstol = 1.0e-6,
            )
            o = sol[w]
            o[2]
        end
        @test any(!iszero, gs_prob2[1])
    end

    # Exercises the EnzymeOriginator method of `_init_originator_gradient`
    # via a full ODE solve under `GaussAdjoint(EnzymeVJP())`. We use the
    # idiomatic Enzyme pattern (mirroring the SCC-init rewrite in #1454):
    # express the loss as a plain function whose captured mutable state
    # (the `ODEProblem`) is passed as an explicit `Duplicated` argument,
    # and reconstruct `repack` *inside* the loss from the duplicated
    # `prob_` so its captured parameter template shares the Enzyme
    # shadow. `Const(loss)` annotates the function and
    # `set_runtime_activity(Reverse)` tolerates runtime-activity
    # transitions through MTK's `remake` path.
    #
    # Residual blocker: `Enzyme.autodiff` enters the `GaussAdjoint`
    # `_concrete_solve_adjoint` rule, which calls
    # `_init_originator_gradient` to differentiate the parameter-init
    # `tunables -> new_u0` mapping. That helper currently invokes
    # `Enzyme.gradient(Enzyme.Reverse, Const(init_loss), tunables)`
    # without `set_runtime_activity`, so the inner Enzyme call still
    # raises `EnzymeRuntimeActivityError` at the MTK init `remake`. The
    # outer-call activity setting doesn't propagate into that nested
    # gradient. Fixing this requires teaching
    # `_init_originator_gradient(::EnzymeOriginator, ...)` to wrap with
    # `set_runtime_activity(Reverse)` (or use `autodiff` + `Duplicated`
    # itself); both are outside this test file's scope. When that lifts,
    # flipping `@test_broken` → `@test` is the only change needed here.
    @testset "Adjoint through Prob (Enzyme)" begin
        function enzyme_loss(t, prob_)
            _, repack_, _ = SS.canonicalize(
                SS.Tunable(), parameter_values(prob_),
            )
            new_prob = remake(prob_; p = repack_(t))
            sensealg = SciMLSensitivity.GaussAdjoint(
                autojacvec = SciMLSensitivity.EnzymeVJP(),
            )
            sol = solve(new_prob; sensealg)
            return sum(sol)
        end
        @test begin
            dprob = Enzyme.make_zero(prob)
            dtunables = zero(tunables)
            Enzyme.autodiff(
                Enzyme.set_runtime_activity(Enzyme.Reverse),
                Enzyme.Const(enzyme_loss),
                Enzyme.Active,
                Enzyme.Duplicated(copy(tunables), dtunables),
                Enzyme.Duplicated(prob, dprob),
            )
            any(!iszero, dtunables)
        end
    end
end

# https://github.com/SciML/SciMLSensitivity.jl/issues/1582
# `ForwardDiffSensitivity` (the default here: length(u0) + length(tunables) <= 100)
# seeds `u0` duals, but a problem carrying initialization data re-derives `u0`
# from `p` during `remake`. If the seed is discarded the whole `du0` is zero, and
# every `Initial(x)` gradient silently vanishes.
@testset "Gradients w.r.t. `Initial` parameters (#1582)" begin
    @parameters m2 = 1.5 d2 = 9.0
    @variables s2(t) = 1.0 v2(t) = 1.0
    initsys = mtkcompile(
        System([D(s2) ~ v2, m2 * D(v2) ~ 1 - d2 * v2], t; name = :initmodel)
    )
    initprob = ODEProblem(initsys, [], (0.0, 1.0))
    set_v0 = SII.setp_oop(initprob, [Initial(initsys.v2)])
    get_s = SII.getsym(initprob, initsys.s2)

    initloss = function (x)
        sol = solve(remake(initprob; p = set_v0(initprob, x)), Tsit5(); saveat = 0.02)
        return sum(abs2, get_s(sol))
    end

    x0 = [1.1]
    @test SciMLSensitivity.automatic_sensealg_choice(
        initprob, state_values(initprob), parameter_values(initprob), false, nothing
    ) isa ForwardDiffSensitivity

    reference = ForwardDiff.gradient(initloss, x0)
    @test !iszero(only(reference))
    @test Zygote.gradient(initloss, x0)[1] ≈ reference rtol = 1.0e-5

    # The true adjoints reach the same value through `du0`.
    for sensealg in (InterpolatingAdjoint(), GaussAdjoint(), QuadratureAdjoint())
        g = Zygote.gradient(
            x -> sum(
                abs2,
                get_s(
                    solve(
                        remake(initprob; p = set_v0(initprob, x)), Tsit5();
                        saveat = 0.02, sensealg
                    )
                )
            ),
            x0
        )[1]
        @test g ≈ reference rtol = 1.0e-5
    end
end
