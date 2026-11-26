using SciMLSensitivity
using ADTypes: AutoFiniteDiff, AutoForwardDiff
using OrdinaryDiffEq
using Test

# Construction maps ADTypes onto the Bool type parameter used internally
a_fd = InterpolatingAdjoint(autodiff = AutoFiniteDiff(), autojacvec = false)
a_ad = InterpolatingAdjoint(autodiff = AutoForwardDiff(), autojacvec = false)
a_true = InterpolatingAdjoint(autodiff = true, autojacvec = false)
a_false = InterpolatingAdjoint(autodiff = false, autojacvec = false)
@test SciMLSensitivity.alg_autodiff(a_fd) === false
@test SciMLSensitivity.alg_autodiff(a_ad) === true
@test SciMLSensitivity.alg_autodiff(a_true) === true
@test SciMLSensitivity.alg_autodiff(a_false) === false

f_fd = ForwardSensitivity(autodiff = AutoFiniteDiff(), autojacvec = false)
f_ad = ForwardSensitivity(autodiff = AutoForwardDiff(), autojacvec = false)
@test SciMLSensitivity.alg_autodiff(f_fd) === false
@test SciMLSensitivity.alg_autodiff(f_ad) === true
# default autojacvec follows the normalized autodiff Bool
f_def = ForwardSensitivity(autodiff = AutoFiniteDiff())
@test f_def.autojacvec === false

# End-to-end: ADTypes sensealg no longer TypeErrors in boolean context
f(u, p, t) = p[1] * u
u0 = [1.0]
tspan = (0.0, 1.0)
p = [1.5]
prob = ODEProblem(f, u0, tspan, p)
sol = solve(prob, Tsit5(), saveat = 0.1, abstol = 1.0e-6, reltol = 1.0e-6)
dg(out, u, p, t, i) = (out .= 1.0)
for sensealg in (
        InterpolatingAdjoint(autodiff = AutoFiniteDiff(), autojacvec = false),
        InterpolatingAdjoint(autodiff = AutoForwardDiff(), autojacvec = false),
        QuadratureAdjoint(autodiff = AutoForwardDiff(), autojacvec = false),
        GaussAdjoint(autodiff = AutoForwardDiff(), autojacvec = false),
    )
    du0, dp = adjoint_sensitivities(
        sol, Tsit5(); t = sol.t, dgdu_discrete = dg,
        sensealg = sensealg, abstol = 1.0e-6, reltol = 1.0e-6
    )
    @test length(dp) == length(p)
    @test all(isfinite, dp)
end
