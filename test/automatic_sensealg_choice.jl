using Lux, ComponentArrays, OrdinaryDiffEq, SciMLSensitivity, Random, Test
using SciMLStructures

# Only import Zygote on Julia <= 1.11
if VERSION <= v"1.11"
    using Zygote
end

rng = Random.default_rng()
tspan = (0.0f0, 8.0f0)

ann = Chain(Dense(1, 32, tanh), Dense(32, 32, tanh), Dense(32, 1))
ps, st = Lux.setup(rng, ann)
p = ComponentArray(ps)

θ, ax = getdata(p), getaxes(p)

function dxdt_(dx, x, p, t)
    ps = ComponentArray(p, ax)
    x1, x2 = x
    dx[1] = x[2]
    return dx[2] = first(ann([t], ps, st))[1]^3
end
x0 = [-4.0f0, 0.0f0]
ts = Float32.(collect(0.0:0.01:tspan[2]))
prob = ODEProblem(dxdt_, x0, tspan, θ)
_, repack, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), p)
sensealg = SciMLSensitivity.automatic_sensealg_choice(prob, x0, θ, true, repack)

# On Julia 1.12+, Enzyme is not yet fully supported, so the automatic choice
# will fall back to a different VJP. Mark this as broken until Enzyme v1.12 support lands.
# See: https://github.com/EnzymeAD/Enzyme.jl/issues/2699
if VERSION >= v"1.12"
    @test sensealg isa GaussAdjoint
    @test_broken sensealg.autojacvec isa EnzymeVJP
else
    @test sensealg isa GaussAdjoint && sensealg.autojacvec isa EnzymeVJP
end
