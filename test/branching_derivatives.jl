using SciMLSensitivity, OrdinaryDiffEq, Zygote, Test

function get_param(breakpoints, values, t)
    for (i, tᵢ) in enumerate(breakpoints)
        if t <= tᵢ
            return values[i]
        end
    end
    return values[end]
end

function fiip(du, u, p, t)
    a = get_param([1.0, 2.0, 3.0], p[1:4], t)
    du[1] = dx = a * u[1] - u[1] * u[2]
    du[2] = dy = -a * u[2] + u[1] * u[2]
end

p = [1.0, 1.0, 1.0, 1.0];
u0 = [1.0; 1.0];

prob = ODEProblem(fiip, u0, (0.0, 4.0), p);

dp1 = Zygote.gradient(
    p -> sum(solve(prob, Tsit5(), u0 = u0, p = p,
        sensealg = ForwardDiffSensitivity(), saveat = 0.1,
        abstol = 1e-12, reltol = 1e-12)),
    p)
dp2 = Zygote.gradient(
    p -> sum(solve(prob, Tsit5(), u0 = u0, p = p,
        sensealg = ForwardDiffSensitivity(convert_tspan = true),
        saveat = 0.1, abstol = 1e-12, reltol = 1e-12)),
    p)
dp3 = Zygote.gradient(
    p -> sum(solve(prob, Tsit5(), u0 = u0, p = p,
        sensealg = ForwardSensitivity(), saveat = 0.1,
        abstol = 1e-12, reltol = 1e-12)),
    p)
dp4 = Zygote.gradient(
    p -> sum(solve(prob, Tsit5(), u0 = u0, p = p, saveat = 0.1,
        abstol = 1e-12, reltol = 1e-12)),
    p)
dp5 = Zygote.gradient(
    p -> sum(solve(prob, Tsit5(), u0 = u0, p = p, saveat = 0.1,
        abstol = 1e-12, reltol = 1e-12,
        sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP()))),
    p)
dp6 = Zygote.gradient(
    p -> sum(solve(prob, Tsit5(), u0 = u0, p = p, saveat = 0.1,
        abstol = 1e-12, reltol = 1e-12,
        sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)))),
    p)

@test dp1[1] ≈ dp2[1]
@test dp1[1] ≈ dp3[1]
@test dp1[1] ≈ dp4[1]
@test dp1[1] ≈ dp5[1]
@test sum(dp5[1]) ≈ sum(dp6[1])
@test all(dp6[1][1:3] .== 0)
