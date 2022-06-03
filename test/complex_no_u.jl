using OrdinaryDiffEq, DiffEqSensitivity, DiffEqFlux, LinearAlgebra, Flux
nn = FastChain(FastDense(1,16),FastDense(16,16,tanh),FastDense(16,2))
initial = initial_params(nn)

function ode2!(u, p, t)
    f1, f2 = nn([t],p)
    [-f1^2; f2]
end

tspan = (0.0, 10.0)
prob = ODEProblem(ode2!, Complex{Float64}[0;0], tspan, initial)

function loss(p)
  sol = last(solve(prob, Tsit5(), p=p, sensealg=BacksolveAdjoint(autojacvec=ZygoteVJP())))
  return norm(sol)
end

optf = Optimization.OptimizationFunction((x,p) -> loss(x), Optimization.AutoZygote())
optprob = Optimization.OptimizationProblem(optfunc, initial)
res = Optimization.solve(optprob, ADAM(0.1), maxiters = 100)
