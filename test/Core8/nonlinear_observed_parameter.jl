using ModelingToolkit, NonlinearSolve, SciMLSensitivity, Zygote, ForwardDiff, Test
using ModelingToolkit: t_nounits as t
using SymbolicIndexingInterface: setp_oop

# Observed-function parameter cotangents on NonlinearProblem: `sol[sym]` puts the
# parameter cotangent in `Δ.prob.p` and leaves `Δ.u === nothing`. MTK-generated
# nonlinear problems additionally have `prob.u0 === nothing` (the initial guess
# lives in the initialization data), which the adjoint cache previously could not
# size from.
@variables xz(t) yz(t)
@parameters γ δ
@mtkcompile nsys = System([0 ~ -γ * xz + δ, yz ~ γ * xz], t)
nlprob = NonlinearProblem(nsys, [xz => 0.5, γ => 1.2, δ => 2.1])
set_np = setp_oop(nlprob, [γ, δ])
np0 = [1.2, 2.1]

function nl_solve_at(ps; kwargs...)
    return solve(
        remake(nlprob; p = set_np(nlprob, ps)), NewtonRaphson(); kwargs...
    )
end

nl_losses = [sol -> sol[yz], sol -> sol[xz], sol -> sol[yz] + sol[xz]]

# xz = δ/γ and yz = γ·xz = δ, so in [γ, δ] order:
#   ∂xz = [-δ/γ², 1/γ] = [-1.4583, 0.8333], ∂yz = [0, 1].
nl_expected = map(
    loss -> ForwardDiff.gradient(ps -> loss(nl_solve_at(ps)), np0), nl_losses
)

@testset "NonlinearProblem observed parameter cotangent" begin
    # The backpass must consume the structural `NonlinearSolution` cotangent:
    # `Δ.u === nothing` contributes zero state cotangent and the parameter part
    # arrives through `Δ.prob.p`.
    nlsol = nl_solve_at(np0)
    _, sym_pb = Zygote.pullback(s -> s[yz], nlsol)
    Δyz = only(sym_pb(1.0))
    _, sym_pb2 = Zygote.pullback(s -> s[xz], nlsol)
    Δxz = only(sym_pb2(1.0))
    for sensealg in (SteadyStateAdjoint(), nothing)
        prob2 = remake(nlprob; p = set_np(nlprob, np0))
        _, bp = SciMLBase._concrete_solve_adjoint(
            prob2, NewtonRaphson(), sensealg,
            nothing, prob2.p, SciMLBase.ChainRulesOriginator(); verbose = false
        )
        # ∂yz = {0, 1}, ∂xz = {1/γ, -δ/γ²}; tunable ordering differs across
        # Julia versions, so compare unordered.
        @test sort(bp(Δyz)[5].tunable) ≈ [0.0, 1.0] atol = 1.0e-8
        @test sort(bp(Δxz)[5].tunable) ≈ sort([1.0 / 1.2, -2.1 / 1.2^2]) atol = 1.0e-8
    end

    for sensealg in (nothing, SteadyStateAdjoint())
        kw = sensealg === nothing ? (;) : (; sensealg)
        for (loss, g) in zip(nl_losses, nl_expected)
            @test only(Zygote.gradient(ps -> loss(nl_solve_at(ps; kw...)), np0)) ≈
                g atol = 1.0e-8
        end
    end
end

@testset "NonlinearLeastSquaresProblem parameter cotangent" begin
    # For f(u, p) = u - p the solution is u* = p, so a state cotangent Δu gives
    # dp = Δu; a `Δ.prob.p` cotangent must additionally accumulate onto it.
    nlls_f(u, p) = u .- p
    for sensealg in (nothing, SteadyStateAdjoint())
        nlls_prob = NonlinearLeastSquaresProblem(nlls_f, [0.5, 0.5], [1.0, 2.0])
        _, bp = SciMLBase._concrete_solve_adjoint(
            nlls_prob, LevenbergMarquardt(), sensealg,
            nlls_prob.u0, nlls_prob.p, SciMLBase.ChainRulesOriginator()
        )
        @test bp((u = nothing, prob = (p = [0.5, 0.5],)))[5] ≈ [0.5, 0.5]
        @test bp((u = [1.0, 1.0], prob = (p = [0.5, 0.5],)))[5] ≈ [1.5, 1.5]
        @test bp([1.0, 1.0])[5] ≈ [1.0, 1.0]
    end
end
