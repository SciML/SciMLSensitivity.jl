using SciMLSensitivity
using Test

function precompile_workload_rhs!(du, u, p, t)
    du[1] = p[1] * u[1]
    return nothing
end

prob = ODEForwardSensitivityProblem(
    precompile_workload_rhs!, [1.0], (0.0, 1.0), [0.5]
)
du = similar(prob.u0)
prob.f(du, prob.u0, prob.p, first(prob.tspan))

@test du == [0.5, 1.0]
