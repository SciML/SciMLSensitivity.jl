using OrdinaryDiffEq, SciMLSensitivity, Zygote, Test

function lotka_volterra(u, p, t)
    du1 = p[1] * u[1] - p[2] * u[1] * u[2]
    du2 = -p[3] * u[2] + p[4] * u[1] * u[2]
    return [du1, du2]
end
u0 = [1.0, 1.0]
tspan = (0.0, 10.0)
p = [1.5, 1.0, 3.0, 1.0]
prob = ODEProblem(lotka_volterra, u0, tspan, p)

function loss(u0; kwargs...)
    solve(remake(prob, u0 = u0), Tsit5(); reltol = 1e-10, abstol = 1e-10, kwargs...).u |>
    last |> sum
end

dp1 = Zygote.gradient(loss, u0)[1]
dp2 = Zygote.gradient(u0 -> loss(u0; sensealg = TrackerAdjoint()), u0)[1]
dp3 = Zygote.gradient(u0 -> loss(u0; sensealg = ReverseDiffAdjoint()), u0)[1]
@test dp1 ≈ dp2
@test dp1 ≈ dp3
