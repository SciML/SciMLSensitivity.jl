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
g_observed = ForwardDiff.gradient(observed_loss, p0)
g_state = ForwardDiff.gradient(state_loss, p0)

@testset "$(nameof(typeof(sensealg)))" for sensealg in (
        nothing,
        InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)),
        QuadratureAdjoint(autojacvec = ReverseDiffVJP(true)),
        GaussAdjoint(autojacvec = ReverseDiffVJP(true)),
        ForwardDiffSensitivity(),
    )
    kw = sensealg === nothing ? (;) : (; sensealg)
    @test Zygote.gradient(ps -> observed_loss(ps; kw...), p0)[1] ≈ g_observed rtol = 1.0e-6
    @test Zygote.gradient(ps -> state_loss(ps; kw...), p0)[1] ≈ g_state rtol = 1.0e-6
end
