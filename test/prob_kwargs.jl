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
    sum(predicted.u[end])
end

function f2(a)
    _prob = remake(prob, p = [a[1]], saveat = savetimes)
    predicted = solve(_prob, Tsit5(), sensealg = ForwardDiffSensitivity(), abstol = 1e-12,
        reltol = 1e-12)
    sum(predicted.u[end])
end

using Zygote
a = ones(3)
@test Zygote.gradient(f, a)[1][1] ≈ Zygote.gradient(f2, a)[1][1]
@test Zygote.gradient(f, a)[1][2] == Zygote.gradient(f2, a)[1][2] == 0
@test Zygote.gradient(f, a)[1][3] == Zygote.gradient(f2, a)[1][3] == 0

# callback in problem construction or in solve call should give same result
odef(du, u, p, t) = du .= u .* p
prob = ODEProblem(odef, [2.0], (0.0, 1.0), [3.0])

function f1(u0p)
    condition(u, t, integrator) = t == 0.5
    affect!(integrator) = integrator.p[1] += 0.1
    cb = DiscreteCallback(condition, affect!)
    prob = ODEProblem{true}(odef, u0p[1:1], (0.0, 1.0), u0p[2:2]; callback = cb)
    sum(solve(prob, Tsit5(), tstops = [0.5], sensealg = ForwardDiffSensitivity()))
end

function f2(u0p)
    condition(u, t, integrator) = t == 0.5
    affect!(integrator) = integrator.p[1] += 0.1
    cb = DiscreteCallback(condition, affect!)
    prob = ODEProblem{true}(odef, u0p[1:1], (0.0, 1.0), u0p[2:2])
    sum(solve(
        prob, Tsit5(), tstops = [0.5], callback = cb, sensealg = ForwardDiffSensitivity()))
end
u0p = [2.0, 3.0]

@test f1(u0p) == f2(u0p)
@test Zygote.gradient(f1, u0p)[1] == Zygote.gradient(f2, u0p)[1]

