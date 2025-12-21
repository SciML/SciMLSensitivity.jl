using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using SymbolicIndexingInterface
using SciMLStructures
import SymbolicIndexingInterface as SII
import SciMLStructures as SS
import ModelingToolkit as MTK
using SciMLSensitivity
using SCCNonlinearSolve  # Needed to load ChainRulesCore extension for SCCNonlinearProblem
using OrdinaryDiffEq
using Test
using ForwardDiff

@parameters σ ρ β
@variables x(t) y(t) z(t) w(t)

eqs = [D(D(x)) ~ σ * (y - x),
    D(y) ~ x * (ρ - z) - y,
    D(z) ~ x * y - β * z,
    w ~ x + y + z + 2 * β]

@mtkbuild sys = ODESystem(eqs, t, defaults = [ρ => missing], guesses = [ρ => 10.0])

u0 = [D(x) => 2.0,
    x => 1.0,
    y => 0.0,
    z => 0.0]

p = [σ => 28.0,
    ρ => 10.0,
    β => 8 / 3]

tspan = (0.0, 100.0)
prob = ODEProblem(sys, u0, tspan, p, jac = true)
sol = solve(prob, Tsit5())

tunables, repack, _ = SS.canonicalize(SS.Tunable(), parameter_values(prob))

@testset "Adjoint through Parameter Initialization" begin
    # For NonlinearProblem initialization with no state variables (empty u0),
    # the gradient flows through observed functions.
    # ForwardDiff works directly with isol[w], while Zygote requires the
    # Zygote.ignore() pattern for observed values due to pullback chaining limitations.

    fn = function (tunables)
        new_prob = remake(prob; p = repack(tunables))
        initdata = new_prob.f.initialization_data
        iprob = initdata.initializeprob
        iprob = if initdata.is_update_oop === Val(true)
            initdata.update_initializeprob!(iprob, new_prob)
        else
            initdata.update_initializeprob!(iprob, new_prob)
            iprob
        end
        isol = solve(iprob)
        isol[w]
    end

    @testset "Forward Mode (ForwardDiff)" begin
        # ForwardDiff works directly with isol[w]
        gs_fwd = ForwardDiff.gradient(fn, tunables)
        @test any(!iszero, gs_fwd)
    end

    # Zygote tests are skipped on Julia 1.12+ due to compatibility issues
    if VERSION >= v"1.12"
        @testset "Reverse Mode (skipped on Julia 1.12+)" begin
            @test_skip false
        end
    else
        using Zygote

        @testset "Reverse Mode" begin
            # For reverse mode with Zygote, use the Zygote.ignore() pattern
            # to access observed values when u0 is empty
            sensealg = SciMLSensitivity.SteadyStateAdjoint(autojacvec = SciMLSensitivity.ZygoteVJP())
            gs_obs, = Zygote.gradient(tunables) do tunables
                new_prob = remake(prob; p = repack(tunables))
                initdata = new_prob.f.initialization_data
                iprob = initdata.initializeprob
                iprob = if initdata.is_update_oop === Val(true)
                    initdata.update_initializeprob!(iprob, new_prob)
                else
                    initdata.update_initializeprob!(iprob, new_prob)
                    iprob
                end
                isol = solve(iprob; sensealg)
                obsfn = Zygote.ignore() do
                    SII.observed(isol.prob.f.sys, w).f_oop
                end
                obsfn(iprob.u0, iprob.p)
            end
            @test any(!iszero, gs_obs)

            # Verify ForwardDiff and Zygote produce the same gradient
            gs_fwd = ForwardDiff.gradient(fn, tunables)
            @test gs_fwd ≈ gs_obs
        end

        @testset "Adjoint through Prob" begin
            sensealg = SciMLSensitivity.GaussAdjoint(autojacvec = SciMLSensitivity.ZygoteVJP())
            gs_prob, = Zygote.gradient(tunables) do tunables
                new_prob = remake(prob; p = repack(tunables))
                sol = solve(new_prob; sensealg)
                sum(sol)
            end
            @test any(!iszero, gs_prob)

            gs_prob, = Zygote.gradient(tunables) do tunables
                new_prob = remake(prob; p = repack(tunables))
                sol = solve(new_prob; sensealg, initializealg = SciMLBase.OverrideInit(), abstol = 1.0e-6)
                o = sol[w]
                o[2]
            end
            @test any(!iszero, gs_prob)
        end
    end
end
