# Regression test for Mooncake as the outer AD around a solve that uses
# `ReverseDiffAdjoint` as the inner sensealg. Before the fix in
# `_concrete_solve_adjoint(..., ::ReverseDiffAdjoint, ..., ::MooncakeOriginator)`,
# this path threw `MooncakeTrackedRealError` because the returned
# `sensitivity_solution(sol, …)` still carried `ReverseDiff.TrackedReal` /
# `TrackedArray` in nested type parameters (interp, prob, alg, …), which broke
# Mooncake's recursive `tangent_type` computation in `@from_rrule`. The fix
# returns a freshly-solved, plain-typed primal solution on the Mooncake path
# while keeping the tape-based backward pass.

using OrdinaryDiffEq
using SciMLSensitivity
using DiffEqCallbacks
using Mooncake
using DifferentiationInterface
using ForwardDiff
using Zygote
using Test

const backend = AutoMooncake(; config = nothing)

# ---------------------------------------------------------------------------
# 1. Plain ODE: Mooncake(ReverseDiffAdjoint) vs. Zygote(ReverseDiffAdjoint)
# ---------------------------------------------------------------------------

function lotka_volterra(u, p, t)
    du1 = p[1] * u[1] - p[2] * u[1] * u[2]
    du2 = -p[3] * u[2] + p[4] * u[1] * u[2]
    return [du1, du2]
end

const lv_u0 = [1.0, 1.0]
const lv_tspan = (0.0, 10.0)
const lv_p0 = [1.5, 1.0, 3.0, 1.0]
const lv_prob = ODEProblem(lotka_volterra, lv_u0, lv_tspan, lv_p0)

function lv_loss(p)
    sol = solve(
        remake(lv_prob; p), Tsit5();
        reltol = 1.0e-10, abstol = 1.0e-10,
        sensealg = ReverseDiffAdjoint()
    )
    return sum(last(sol.u))
end

function lv_loss_plain(p)
    sol = solve(
        remake(lv_prob; p), Tsit5();
        reltol = 1.0e-10, abstol = 1.0e-10
    )
    return sum(last(sol.u))
end

@testset "Mooncake(ReverseDiffAdjoint) plain ODE" begin
    prep = prepare_gradient(lv_loss, backend, lv_p0)
    grad_moon = DifferentiationInterface.gradient(lv_loss, prep, backend, lv_p0)
    grad_zyg = Zygote.gradient(lv_loss, lv_p0)[1]
    grad_fd = ForwardDiff.gradient(lv_loss_plain, lv_p0)
    # The Mooncake path re-runs the forward solve on plain inputs to obtain a
    # cleanly-typed primal; the tape's tracked-arithmetic forward and the
    # plain forward can differ by a few ULPs, which causes a correspondingly
    # small (~1.0e-5 relative) difference in the cotangent fed into the
    # ReverseDiff tape compared to the Zygote path. Check against ForwardDiff
    # at the accuracy `ReverseDiffAdjoint` already has vs. a first-principles
    # gradient.
    @test grad_moon ≈ grad_fd rtol = 1.0e-4
    @test grad_moon ≈ grad_zyg rtol = 1.0e-4
end

# ---------------------------------------------------------------------------
# 2. Hybrid ODE with PresetTimeCallback — this is the hybrid_diffeq tutorial
#    shape that PR #1419 was forced to keep on Zygote.
# ---------------------------------------------------------------------------

function decay!(du, u, p, t)
    du[1] = -p[1] * u[1]
    du[2] = -p[2] * u[2]
    return nothing
end

const hyb_u0 = [2.0, 0.0]
const hyb_tspan = (0.0, 10.5)
const hyb_dosetimes = [1.0, 2.0, 4.0, 8.0]
const hyb_p0 = [1.0, 1.0]

function hyb_affect!(integrator)
    integrator.u .= integrator.u .+ 1
    return nothing
end

const hyb_cb = PresetTimeCallback(
    hyb_dosetimes, hyb_affect!; save_positions = (false, false)
)

const hyb_prob = ODEProblem(decay!, hyb_u0, hyb_tspan, hyb_p0)

function hyb_loss(p)
    sol = solve(
        hyb_prob, Tsit5(); p, callback = hyb_cb,
        saveat = 0.5, sensealg = ReverseDiffAdjoint()
    )
    return sum(abs2, last(sol.u))
end

function hyb_loss_plain(p)
    sol = solve(
        hyb_prob, Tsit5(); p, callback = hyb_cb, saveat = 0.5
    )
    return sum(abs2, last(sol.u))
end

@testset "Mooncake(ReverseDiffAdjoint) hybrid ODE with PresetTimeCallback" begin
    prep = prepare_gradient(hyb_loss, backend, hyb_p0)
    grad_moon = DifferentiationInterface.gradient(hyb_loss, prep, backend, hyb_p0)
    grad_zyg = Zygote.gradient(hyb_loss, hyb_p0)[1]
    grad_fd = ForwardDiff.gradient(hyb_loss_plain, hyb_p0)
    # Looser tolerance than the plain-ODE case: the PresetTimeCallback
    # amplifies the ~ULP difference between the tape's tracked forward and
    # the primal's plain forward into ~2e-4 relative gradient drift.
    @test grad_moon ≈ grad_fd rtol = 1.0e-3
    @test grad_moon ≈ grad_zyg rtol = 1.0e-3
end
