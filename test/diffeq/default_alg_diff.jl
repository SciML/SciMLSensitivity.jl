using ComponentArrays, DifferentialEquations, Lux, Random, SciMLSensitivity, Zygote

function f(du, u, p, t)
    du .= first(nn(u, p, st))
    nothing
end

nn = Dense(8, 8, tanh)
ps, st = Lux.setup(Random.default_rng(), nn)
ps = ComponentArray(ps)

r = rand(Float32, 8, 64)

function f2(x)
    prob = ODEProblem(f, r, (0.0f0, 1.0f0), x)
    sol = solve(prob; sensealg = InterpolatingAdjoint(; autodiff = true, autojacvec = true))
    sum(last(sol.u))
end

f2(ps)
Zygote.gradient(f2, ps)
