using SciMLSensitivity, OrdinaryDiffEq, Zygote
using Test, ForwardDiff, SteadyStateDiffEq
import Tracker, ReverseDiff, ChainRulesCore

function fiip(du, u, p, t)
    du[1] = dx = p[1] * u[1] - p[2] * u[1] * u[2]
    du[2] = dy = -p[3] * u[2] + p[4] * u[1] * u[2]
end
function foop(u, p, t)
    dx = p[1] * u[1] - p[2] * u[1] * u[2]
    dy = -p[3] * u[2] + p[4] * u[1] * u[2]
    [dx, dy]
end

p = [1.5, 1.0, 3.0, 1.0];
u0 = [1.0; 1.0];
prob = ODEProblem(fiip, u0, (0.0, 10.0), p)
proboop = ODEProblem(foop, u0, (0.0, 10.0), p)

probsteady = SteadyStateProblem(proboop)

@test_throws SciMLSensitivity.AdjointSteadyProblemPairingError Zygote.gradient(
    (u0, p) -> sum(solve(probsteady,
        DynamicSS(Tsit5()),
        u0 = u0,
        p = p,
        abstol = 1e-14,
        reltol = 1e-14,
        saveat = 0.1,
        sensealg = QuadratureAdjoint())),
    u0, p)
