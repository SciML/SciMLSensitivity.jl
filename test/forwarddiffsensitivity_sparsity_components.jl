using OrdinaryDiffEq, SciMLSensitivity
using ComponentArrays, LinearAlgebra, Optimization, OptimizationOptimisers, Test

const nknots = 10
const h = 1.0 / (nknots + 1)
x = range(0, step = h, length = nknots)
u0 = sin.(π * x)

@inline function f(du, u, p, t)
    du .= zero(eltype(u))
    u₃ = @view u[3:end]
    u₂ = @view u[2:(end - 1)]
    u₁ = @view u[1:(end - 2)]
    @. du[2:(end - 1)] = p.k * ((u₃ - 2 * u₂ + u₁) / (h^2.0))
    nothing
end

p_true = ComponentArray(k = 0.42)
jac_proto = Tridiagonal(similar(u0, nknots - 1), similar(u0), similar(u0, nknots - 1))
prob = ODEProblem(ODEFunction(f, jac_prototype = jac_proto), u0, (0.0, 1.0), p_true)
@time sol_true = solve(prob, Rodas4P(), saveat = 0.1)

function loss(p)
    _prob = remake(prob, p = p)
    sol = solve(_prob, Rodas4P(autodiff = false), saveat = 0.1,
        sensealg = ForwardDiffSensitivity())
    sum((sol .- sol_true) .^ 2)
end

p0 = ComponentArray(k = 1.0)

optf = Optimization.OptimizationFunction((x, p) -> loss(x), Optimization.AutoZygote())
optprob = Optimization.OptimizationProblem(optf, p0)
res = Optimization.solve(optprob, Adam(0.01), maxiters = 100)

@test res.u.k≈0.42461977305259074 rtol=1e-1
