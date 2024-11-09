using SciMLSensitivity, ForwardDiff, Distributions, OrdinaryDiffEq, LinearAlgebra, Test

function fiip(du, u, p, t)
    du[1] = dx = p[1] * u[1] - p[2] * u[1] * u[2]
    du[2] = dy = -p[3] * u[2] + p[4] * u[1] * u[2]
end
function g(sol)
    J = extract_local_sensitivities(sol, true)[2]
    det(J' * J)
end

u0 = [1.0, 1.0]
p = [1.5, 1.0, 3.0, 1.0]
prob = ODEForwardSensitivityProblem(fiip, u0, (0.0, 10.0), p, saveat = 0:10)
sol = solve(prob, Tsit5())
u0_dist = [Uniform(0.9, 1.1), 1.0]
p_dist = [1.5, truncated(Normal(1.5, 0.1), 1.1, 1.9), 3.0, 1.0]
u0_dist_extended = vcat(u0_dist, zeros(length(p) * length(u0)))

function fiip_expe_SciML_forw_sen_SciML()
    prob = ODEForwardSensitivityProblem(fiip, u0, (0.0, 10.0), p, saveat = 0:10)
    prob_func = function (prob, i, repeat)
        _prob = remake(
            prob, u0 = [isa(ui, Distribution) ? rand(ui) : ui for ui in u0_dist],
            p = [isa(pj, Distribution) ? rand(pj) : pj for pj in p_dist])
        _prob
    end
    output_func = function (sol, i)
        (g(sol), false)
    end
    monte_prob = EnsembleProblem(prob; output_func = output_func, prob_func = prob_func)
    sol = solve(monte_prob, Tsit5(), EnsembleSerial(), trajectories = 100_000)
    mean(sol.u)
end

@test fiip_expe_SciML_forw_sen_SciML()≈3.56e6 rtol=4e-2

# `remake`: https://github.com/SciML/SciMLSensitivity.jl/issues/1137

function ff3(du, u, p, t)
    du[1] = dx = p[1] * u[1] - p[2] * u[1] * u[2]
    du[2] = dy = -p[3] * u[2] + u[1] * u[2]
end

p = [1.5, 1.0, 3.0]
ts = (0, 10)
prob = ODEForwardSensitivityProblem(ff3, [1.0; 1.0], ts, p, sensealg=ForwardDiffSensitivity())
sol = solve(prob, Tsit5())

# https://github.com/SciML/SciMLSensitivity.jl/issues/1143

prob1 = ODEForwardSensitivityProblem(ff3, [1.0, 1.0], (0.0,10.0), p,
                                     sensealg = ForwardSensitivity())
prob2 = remake(prob1, tspan = (0.0, 10.0))
@test length(prob1.u0) == length(prob2.u0) == 8