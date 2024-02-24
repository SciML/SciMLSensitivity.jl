using OrdinaryDiffEq, ComponentArrays, Random,
      SciMLSensitivity, LinearAlgebra, Optimization, OptimizationOptimisers, Lux
nn = Chain(Dense(1, 16), Dense(16, 16, tanh), Dense(16, 2)) |> f64
ps, st = Lux.setup(Random.default_rng(), nn)
ps = ComponentArray(ps)

function ode2!(u, p, t)
    f1, f2 = first(nn([t], p, st)) .+ im
    [-f1^2; f2]
end

tspan = (0.0, 10.0)
prob = ODEProblem(ode2!, Complex{Float64}[0; 0], tspan, ps)

loss = function (p)
    sol = last(solve(prob, Tsit5(), p = p,
        sensealg = BacksolveAdjoint(autojacvec = ZygoteVJP(allow_nothing = true))))
    return norm(sol)
end

optf = Optimization.OptimizationFunction((x, p) -> loss(x), Optimization.AutoZygote())
optprob = Optimization.OptimizationProblem(optf, ps)
res = Optimization.solve(optprob, Adam(0.1), maxiters = 100)
