using OrdinaryDiffEq, SciMLSensitivity, Zygote, ForwardDiff, Test

function neural_ode(u, p, t)
    return u * p[1]
end

p = [2.0]
u0 = rand(1)[1]
tspan = (0.0, 10.0)
t = Array(range(0, 0.1, length = 100))
prob_neuralode = ODEProblem(neural_ode, u0, tspan)

function loss_neuralode(p)
    trial = Array(
        solve(
            prob_neuralode, AutoTsit5(Rosenbrock23()), u0 = u0, p = p,
            saveat = t, abstol = 1.0e-6, reltol = 1.0e-6
        )
    )
    loss = sum(abs2, trial)
    return loss
end

dp1 = Zygote.gradient(loss_neuralode, p)[1]
dp2 = ForwardDiff.gradient(loss_neuralode, p)

@test dp1 ≈ dp2 atol = 1.0e-8
