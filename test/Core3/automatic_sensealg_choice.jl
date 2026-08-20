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
    dx[2] = first(ann([t], ps, st))[1]^3
    return nothing
end
x0 = [-4.0f0, 0.0f0]
ts = Float32.(collect(0.0:0.01:tspan[2]))
prob = ODEProblem(dxdt_, x0, tspan, θ)
_, repack, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), p)
sensealg = SciMLSensitivity.automatic_sensealg_choice(prob, x0, θ, true, repack)

@test sensealg isa GaussAdjoint
@test sensealg.autojacvec isa EnzymeVJP

# Regression test: an in-place RHS that is a struct holding `SparseMatrixCSC` fields must
# still select EnzymeVJP. The availability probe used to capture the RHS in a closure, which
# Enzyme cannot prove read-only when it holds sparse arrays, so the probe threw
# EnzymeMutabilityException and silently fell back to ReverseDiff. See SciMLOperators #319.
using LinearAlgebra, SparseArrays

struct SparseStructRHS{M}
    A1::M
    A2::M
end
function (r::SparseStructRHS)(du, u, p, t)
    mul!(du, r.A1, u, -p[1], zero(eltype(du)))
    mul!(du, r.A2, u, p[2], one(eltype(du)))
    return nothing
end

let N = 150
    rhs = SparseStructRHS(spdiagm(1 => ones(N - 1)), spdiagm(-1 => ones(N - 1)))
    u0 = ones(N)
    psparse = [1.0, 2.0]
    probsparse = ODEProblem{true}(rhs, u0, (0.0, 1.0), psparse)
    _, repack_sparse, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), psparse)
    @test SciMLSensitivity.inplace_vjp(probsparse, u0, psparse, true, repack_sparse) isa
        EnzymeVJP
    sensealg_sparse = SciMLSensitivity.automatic_sensealg_choice(
        probsparse, u0, psparse, true, repack_sparse
    )
    @test sensealg_sparse isa GaussAdjoint
    @test sensealg_sparse.autojacvec isa EnzymeVJP
end

# Regression test for #1605: `automatic_sensealg_choice` used to re-canonicalize the
# already-canonical tunables vector, overwriting the caller's repack (e.g. for
# `MTKParameters`) with an identity-like `ArrayRepack`. Every VJP probe then called the
# RHS with the bare tunables vector; any RHS reading a non-tunable portion threw in the
# probe primal, so the chooser silently fell back to numerical VJPs
# `GaussAdjoint(autodiff = false, autojacvec = false)`. More than 100 tunables are needed
# to get past the `ForwardDiffSensitivity` short-circuit and reach the probes.
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D

let
    @parameters k[1:128] = 0.01 .* ones(128)
    @parameters c = 3.0 [tunable = false]
    @variables x(t) = 1.0
    @named sys = System([D(x) ~ -c * sum(k) * x], t)
    sys = mtkcompile(sys)
    prob = ODEProblem(sys, [], (0.0, 1.0))

    tunables, repack, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), prob.p)
    sensealg_mtk = SciMLSensitivity.automatic_sensealg_choice(
        prob, prob.u0, tunables, true, repack, prob.p
    )
    @test sensealg_mtk isa GaussAdjoint
    @test !(sensealg_mtk.autojacvec isa Bool)
end
