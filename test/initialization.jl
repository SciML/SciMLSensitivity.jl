using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using SymbolicIndexingInterface
using SciMLStructures
import SymbolicIndexingInterface as SII
import SciMLStructures as SS
import ModelingToolkit as MTK
using SciMLSensitivity
using OrdinaryDiffEq
using Zygote
using Test

@parameters σ ρ β
@variables x(t) y(t) z(t) w(t)

eqs = [D(D(x)) ~ σ * (y - x), 
    D(y) ~ x * (ρ - z) - y,
    D(z) ~ x * y - β * z,
    w ~ x + y + z + 2 * β]

@mtkbuild sys = ODESystem(eqs, t)

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

@testset "Adjoint through Initialization" begin
    @testset "Forward Mode" begin
        gs_fwd, = Zygote.gradient(tunables) do tunables
            new_prob = remake(prob; p = repack(tunables))
            iprob = new_prob.f.initialization_data.initializeprob
            isol = solve(iprob)
            isol[w]
        end
        @test any(!iszero, gs_fwd)
    end

    @testset "Reverse Mode" begin
        sensealg = SciMLSensitivity.SteadyStateAdjoint(autojacvec = SciMLSensitivity.ReverseDiffVJP())
        gs_reverse, = Zygote.gradient(tunables) do tunables
            new_prob = remake(prob; p = repack(tunables))
            iprob = new_prob.f.initialization_data.initializeprob
            isol = solve(iprob; sensealg)
            isol[w]
        end
        @test any(!iszero, gs_reverse)

        sensealg = SciMLSensitivity.SteadyStateAdjoint(autojacvec = SciMLSensitivity.ZygoteVJP())
        gs_zyg, = Zygote.gradient(tunables) do tunables
            new_prob = remake(prob; p = repack(tunables))
            iprob = new_prob.f.initialization_data.initializeprob
            isol = solve(iprob; sensealg)
            isol[w]
        end
        @test any(!iszero, gs_zyg)

        sensealg = SciMLSensitivity.SteadyStateAdjoint(autojacvec = SciMLSensitivity.ZygoteVJP())
        gs_obs, = Zygote.gradient(tunables) do tunables
            new_prob = remake(prob; p = repack(tunables))
            iprob = new_prob.f.initialization_data.initializeprob
            isol = solve(iprob; sensealg)
            fo = Zygote.ignore() do
                SII.observed(isol.prob.f.sys, w).f_oop
            end
            fo(iprob.u0, iprob.p)
        end
        @test gs_zyg ≈ gs_obs
    end

    @testset "Adjoint through Prob" begin
        gs_prob, = Zygote.gradient(tunables) do tunables
            new_prob = remake(prob; p = repack(tunables))
            sol = solve(prob; sensealg)
            sum(sol)
        end
        @test any(!iszero, gs_prob)

        gs_prob, = Zygote.gradient(tunables) do tunables
            new_prob = remake(prob; p = repack(tunables))
            sol = solve(new_prob, sensealg = sensealg)
            o = sol[w]
            o[1]
        end
        @test any(!iszero, gs_prob)
    end

end
