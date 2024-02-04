using Lux, ComponentArrays, OrdinaryDiffEq, SciMLSensitivity, Zygote, Random

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
end
x0 = [-4.0f0, 0.0f0]
ts = Float32.(collect(0.0:0.01:tspan[2]))
prob = ODEProblem(dxdt_, x0, tspan, θ)
SciMLSensitivity.automatic_sensealg_choice(prob, x0, θ, true) == InterpolatingAdjoint{0, true, Val{:central}, EnzymeVJP}(EnzymeVJP(0), false, false)