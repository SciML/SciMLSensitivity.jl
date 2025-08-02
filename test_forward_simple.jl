using SciMLSensitivity, OrdinaryDiffEq
using ADTypes: AutoFiniteDiff

function simple_ode!(du, u, p, t)
    du[1] = p[1] * u[1]
end

# Test the exact same constructor call that's failing
try
    # This should fail with the same error as in CI
    alg = ForwardSensitivity(autodiff = AutoFiniteDiff())
    println("ERROR: This should have failed but didn't!")
catch e
    println("Expected error caught: ", e)
end