using DiffEqFlux, OrdinaryDiffEq, DiffEqSensitivity
using CUDA, Test
CUDA.allowscalar(false)

ann = FastChain(FastDense(1, 4, tanh))
p = initial_params(ann)

function func(x, p, t)
    ann([t],p)[1]*H*x
end

x0 = gpu(rand(Float32, 2))
x1 = gpu(rand(Float32, 2))

prob = ODEProblem(func, x0, (0.0f0, 1.0f0))

function evolve(p)
    solve(prob, Tsit5(), p=p, save_start=false,
          save_everystep=false, abstol=1e-4, reltol=1e-4,
          sensealg=QuadratureAdjoint(autojacvec=ZygoteVJP())).u[1]
end

function cost(p)
    x = evolve(p)
    c = sum(abs,x - x1)
    #println(c)
    c
end

grad = Zygote.gradient(cost,p)[1]
@test !iszero(grad[1])
@test iszero(grad[2:4])
@test !iszero(grad[5])
@test iszero(grad[6:end])
