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
    # added alongside this testset. Annotations follow the documented
    # user-side pattern: `Const(loss)` for the closure that captures the
    # mutable `ODEProblem`, and `set_runtime_activity(Reverse)` so Enzyme's
    # activity analysis tolerates the runtime-activity transitions through
    # MTK's `remake` path. With these in place the activity layer is
    # handled; the remaining blocker is a `MixedDuplicated` /
    # `Core.SimpleVector` MethodError further down in Enzyme's
    # runtime-activity wrapping for MTK-System / NonlinearSolution
    # types — tracked in SciMLSensitivity.jl#1359. When that lifts,
    # flipping `@test_broken` → `@test` is the only change needed here.
    @testset "Adjoint through Prob (Enzyme)" begin
        sensealg = SciMLSensitivity.GaussAdjoint(
            autojacvec = SciMLSensitivity.EnzymeVJP(),
        )
        loss = let prob = prob, repack = repack, sensealg = sensealg
            function (tunables)
                new_prob = remake(prob; p = repack(tunables))
                sol = solve(new_prob; sensealg)
                return sum(sol)
            end
        end
        @test_broken begin
            g = Enzyme.gradient(
                Enzyme.set_runtime_activity(Enzyme.Reverse),
                Enzyme.Const(loss), copy(tunables),
            )[1]
            any(!iszero, g)
        end
    end
end
