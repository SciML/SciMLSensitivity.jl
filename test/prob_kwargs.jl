using OrdinaryDiffEq, SciMLSensitivity

function growth(du, u, p, t)
    @. du = p * u * (1 - u)
end
u0 = [0.1]
tspan = (0.0, 2.0)
prob = ODEProblem(growth, u0, tspan, [1.0])
sol = solve(prob, Tsit5(), reltol = 1e-8, abstol = 1e-8)

savetimes = [0.0, 1.0, 1.9]

function f(a)
    _prob = remake(prob, p = [a[1]], saveat = savetimes)
    predicted = solve(_prob, Tsit5(), sensealg = InterpolatingAdjoint(), abstol = 1e-12,
                      reltol = 1e-12)
    sum(predicted[end])
end

function f2(a)
    _prob = remake(prob, p = [a[1]], saveat = savetimes)
    predicted = solve(_prob, Tsit5(), sensealg = ForwardDiffSensitivity(), abstol = 1e-12,
                      reltol = 1e-12)
    sum(predicted[end])
end

using Zygote
a = ones(3)
@test Zygote.gradient(f, a)[1][1] ≈ Zygote.gradient(f2, a)[1][1]
@test Zygote.gradient(f, a)[1][2] == Zygote.gradient(f2, a)[1][2] == 0
@test Zygote.gradient(f, a)[1][3] == Zygote.gradient(f2, a)[1][3] == 0
