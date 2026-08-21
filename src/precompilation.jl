function _precompile_sensitivity_rhs!(du, u, p, t)
    du[1] = p[1] * u[1]
    return nothing
end

@compile_workload begin
    p = [0.5]
    prob = ODEForwardSensitivityProblem(
        _precompile_sensitivity_rhs!, [1.0], (0.0, 1.0), p
    )
    du = similar(prob.u0)
    prob.f(du, prob.u0, prob.p, first(prob.tspan))
end
