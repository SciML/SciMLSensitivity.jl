using OrdinaryDiffEq, Zygote, SciMLSensitivity, ComponentArrays, Lux, ForwardDiff
using Test, Random, LinearAlgebra

const T = 10.0;
const ω = π / T;
const id = Matrix{Complex{Float64}}(I, 2, 2);
const u0 = id;
const utarget = Matrix{Complex{Float64}}([im 0; 0 -im]);

ann = Lux.Chain(Lux.Dense(1, 32), Lux.Dense(32, 32, tanh), Lux.Dense(32, 1));
rng = Random.default_rng();
ip, st = Lux.setup(rng, ann);

function f_nn(u, p, t)
    local a, _ = ann([t / T], p, st)
    local A = [a[1] 0.0; 0.0 -a[1]]
    return -(im * A) * u
end

tspan = (0.0, T)
prob_ode = ODEProblem(f_nn, u0, tspan, ComponentArray(ip));

function loss_adjoint(p; sensealg = nothing)
    local prediction = solve(
        prob_ode, BS5(), p = p, abstol = 1e-13, reltol = 1e-13, sensealg = sensealg)
    local usol = last(prediction)
    local loss = abs(1.0 - abs(tr(usol * utarget') / 2))
    return loss
end

dp1 = Zygote.gradient(loss_adjoint, ComponentArray(ip))
dp2 = ForwardDiff.gradient(loss_adjoint, ComponentArray(ip))
dp3 = Zygote.gradient(
    x -> loss_adjoint(
        x, sensealg = InterpolatingAdjoint(autodiff = false, autojacvec = false)),
    ComponentArray(ip))

@test dp1[1]≈dp2 atol=1e-2
@test dp1[1]≈dp3[1] atol=5e-2
