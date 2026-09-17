using ModelingToolkit, OrdinaryDiffEq, SciMLSensitivity, Zygote, ForwardDiff, Test
using ModelingToolkit: t_nounits as t, D_nounits as D
using SymbolicIndexingInterface: setp_oop

@variables x(t) y(t)
@parameters α β
@mtkcompile sys = System([D(x) ~ -β * x, y ~ α * x], t)
prob = ODEProblem(sys, [x => 1.0, α => 1.2, β => 2.1], (0.0, 1.0))
set_p = setp_oop(prob, [α, β])
p0 = [1.2, 2.1]
function solve_at(ps; kwargs...)
    return solve(
        remake(prob; p = set_p(prob, ps)), Tsit5(); saveat = 0.1, abstol = 1.0e-8,
        reltol = 1.0e-8, kwargs...
    )
end

# α enters only through the observed variable
observed_loss(ps; kwargs...) = sum(solve_at(ps; kwargs...)[y])
state_loss(ps; kwargs...) = sum(solve_at(ps; kwargs...)[x])
# α through the observed variable, β through the state, in one gradient. The two
# solves are needed: `sol[x]` and `sol[y]` pullbacks return differently-shaped
# cotangents for the same solution object, which `Zygote.accum` cannot combine.
function mixed_loss(ps; kwargs...)
    return sum(solve_at(ps; kwargs...)[x]) + sum(solve_at(ps; kwargs...)[y])
end
# multi-symbol indexing also routes the observed part through `Δ.prob.p`
pair_loss(ps; kwargs...) = sum(sum, solve_at(ps; kwargs...)[[x, y]])
pair_loss_colon(ps; kwargs...) = sum(sum, solve_at(ps; kwargs...)[[x, y], :])

losses = (observed_loss, state_loss, mixed_loss, pair_loss, pair_loss_colon)
expected = map(loss -> ForwardDiff.gradient(loss, p0), losses)

@testset "$(nameof(typeof(sensealg)))" for sensealg in (
        nothing,
        InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)),
        QuadratureAdjoint(autojacvec = ReverseDiffVJP(true)),
        GaussAdjoint(autojacvec = ReverseDiffVJP(true)),
        ForwardDiffSensitivity(),
    )
    kw = sensealg === nothing ? (;) : (; sensealg)
    for (loss, g) in zip(losses, expected)
        @test Zygote.gradient(ps -> loss(ps; kw...), p0)[1] ≈ g rtol = 1.0e-6
    end
end
