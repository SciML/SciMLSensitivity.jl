using SciMLSensitivity
using OrdinaryDiffEq

using FiniteDiff
using Zygote
using ForwardDiff

u0 = [1.0, 1.0]
p = [1.5, 1.0, 3.0, 1.0]
function fiip(du, u, p, t)
    du[1] = dx = p[1] * u[1] - p[2] * u[1] * u[2]
    du[2] = dy = -p[3] * u[2] + p[4] * u[1] * u[2]
end

prob = ODEProblem(fiip, u0, (0.0, 10.0), [1.5, 1.0, 3.0, 1.0], reltol = 1e-14,
    abstol = 1e-14)
function cost(p1)
    _prob = remake(prob, p = vcat(p1, p[2:end]))
    sol = solve(_prob, Tsit5(), sensealg = ForwardDiffSensitivity(),
        saveat = 0.1, abstol = 1e-12, reltol = 1e-12)
    sum(sol)
end
res = FiniteDiff.finite_difference_derivative(cost, p[1]) #  8.305557728239275
res2 = ForwardDiff.derivative(cost, p[1]) # 8.305305252400714 # only 1 dual number
res3 = Zygote.gradient(cost, p[1])[1] # (8.305266428305409,) # 4 dual numbers

function cost(p1)
    _prob = remake(prob, p = vcat(p1, p[2:end]))
    sol = solve(_prob, Tsit5(), sensealg = ForwardSensitivity(),
        saveat = 0.1, abstol = 1e-12, reltol = 1e-12)
    sum(sol)
end
res4 = Zygote.gradient(cost, p[1])[1] # (7.720368430265481,)

@test res ≈ res2
@test res2 ≈ res3
@test res2 ≈ res4
