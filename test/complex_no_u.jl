using OrdinaryDiffEq, DiffEqSensitivity, LinearAlgebra, Optimization, OptimizationFlux, Flux
nn = Chain(Dense(1,16),Dense(16,16,tanh),Dense(16,2))
initial,re = Flux.destructure(nn)

function ode2!(u, p, t)
    f1, f2 = re(p)([t]) .+ im
    [-f1^2; f2]
end

tspan = (0.0, 10.0)
prob = ODEProblem(ode2!, Complex{Float64}[0;0], tspan, initial)

function loss(p)
  sol = last(solve(prob, Tsit5(), p=p, sensealg=BacksolveAdjoint(autojacvec=ZygoteVJP(allow_nothing=true))))
  return norm(sol)
end

optf = Optimization.OptimizationFunction((x,p) -> loss(x), Optimization.AutoZygote())
optprob = Optimization.OptimizationProblem(optf, initial)
res = Optimization.solve(optprob, ADAM(0.1), maxiters = 100)