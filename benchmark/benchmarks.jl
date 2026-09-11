using SciMLSensitivity, OrdinaryDiffEq, ForwardDiff, ADTypes, BenchmarkTools

const SUITE = BenchmarkGroup()

# Lotka-Volterra
function lotka!(du, u, p, t)
    du[1] = p[1] * u[1] - u[1] * u[2]
    du[2] = -p[2] * u[2] + u[1] * u[2]
    return nothing
end
u0 = [1.0, 1.0]
p = [1.5, 1.0]
tspan = (0.0, 10.0)
prob = ODEProblem(lotka!, u0, tspan, p)

# =============================================================================
# Forward sensitivity
# =============================================================================

SUITE["forward"] = BenchmarkGroup()

fs_prob = ODEForwardSensitivityProblem(ODEFunction(lotka!), u0, tspan, p)
SUITE["forward"]["solve"] = @benchmarkable solve($fs_prob, Tsit5())
SUITE["forward"]["construct"] = @benchmarkable ODEForwardSensitivityProblem(
    $(ODEFunction(lotka!)), $u0, $tspan, $p
)

# =============================================================================
# Parameter gradient via ForwardDiff through the solver
# =============================================================================

SUITE["gradient"] = BenchmarkGroup()

function loss(pvec)
    _prob = remake(prob; p = pvec)
    return sum(Array(solve(_prob, Tsit5(); saveat = 0.1)))
end

SUITE["gradient"]["forwarddiff"] = @benchmarkable ForwardDiff.gradient($loss, $p)
SUITE["gradient"]["loss_eval"] = @benchmarkable $loss($p)

# =============================================================================
# Adjoint sensitivity solve
# =============================================================================

SUITE["adjoint"] = BenchmarkGroup()

sol = solve(prob, Tsit5(); saveat = 0.1)
dg(out, u, p, t, i) = (out .= 1.0)

SUITE["adjoint"]["interpolating"] = @benchmarkable adjoint_sensitivities(
    $sol, Tsit5(); t = $(sol.t), dgdu_discrete = $dg,
    sensealg = InterpolatingAdjoint(autodiff = AutoFiniteDiff(), autojacvec = false)
)
SUITE["adjoint"]["quadrature"] = @benchmarkable adjoint_sensitivities(
    $sol, Tsit5(); t = $(sol.t), dgdu_discrete = $dg,
    sensealg = QuadratureAdjoint(autodiff = AutoForwardDiff(), autojacvec = false)
)
