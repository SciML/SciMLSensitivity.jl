using ModelingToolkit, OrdinaryDiffEq
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEq
using OrdinaryDiffEqCore
using OrdinaryDiffEqNonlinearSolve: BrownFullBasicInit
using SciMLBase: NoInit
using SciMLSensitivity
using Enzyme
using Mooncake
using Tracker
import SciMLStructures as SS
using SymbolicIndexingInterface
using Test

@parameters σ ρ β A[1:3]
@variables x(t) y(t) z(t) w(t) w2(t)

eqs = [
    D(D(x)) ~ σ * (y - x),
    D(y) ~ x * (ρ - z) - y,
    D(z) ~ x * y - β * z,
    w ~ x + y + z + 2 * β,
]

@mtkbuild sys = ODESystem(eqs, t)

u0 = [
    D(x) => 2.0,
    x => 1.0,
    y => 0.0,
    z => 0.0,
]

p = [
    σ => 28.0,
    ρ => 10.0,
    β => 8 / 3,
]
# A => ones(3),]

tspan = (0.0, 100.0)
prob = ODEProblem(sys, u0, tspan, p, jac = true)
sol = solve(prob, Tsit5())

# Extract tunables as Vector{Float64} for AD compatibility
tunables, repack, _ = SS.canonicalize(SS.Tunable(), parameter_values(prob))

sensealg = GaussAdjoint(; autojacvec = SciMLSensitivity.EnzymeVJP())

dmtk = Tracker.gradient(tunables) do tunables
    new_sol = solve(prob, Tsit5(); p = repack(tunables), sensealg)
    sum(new_sol)
end

# Verify observed function identity holds for the solution
mtkparams = SciMLSensitivity.parameter_values(sol)
test_sol = solve(prob, Tsit5(); p = mtkparams)
@test all(isapprox.(test_sol[x + y + z + 2 * β - w], 0, atol = 1.0e-12))

# Force DAE handling for Initialization

eqs = [
    D(D(x)) ~ σ * (y - x),
    D(y) ~ x * (ρ - z) - y,
    D(z) ~ x * y - β * z,
    w ~ x + y + z + 2 * β,
    0 ~ x^2 + y^2 - w2^2,
]

@mtkbuild sys = ODESystem(eqs, t)

u0_incorrect = [
    D(x) => 2.0,
    x => 1.0,
    y => 0.0,
    z => 0.0,
]

p = [
    σ => 28.0,
    ρ => 10.0,
    β => 8 / 3,
]

tspan = (0.0, 100.0)

# Check that the gradients for the `solve` are the same both for an initialization
# (for the algebraic variables) initialized poorly (therefore needs correction with BrownBasicInit)
# and with the initialization corrected to satisfy the algebraic equation
prob_incorrectu0 = ODEProblem(
    sys, u0_incorrect, tspan, p, jac = true, guesses = [w2 => 0.0],
)
tunables_incorrectu0, repack_incorrectu0, _ = SS.canonicalize(
    SS.Tunable(), parameter_values(prob_incorrectu0),
)
test_sol = solve(prob_incorrectu0, Rodas5P(), abstol = 1.0e-6, reltol = 1.0e-3)

u0_timedep = [
    D(x) => 2.0,
    x => 1.0,
    y => t,
    z => 0.0,
]
# this ensures that `y => t` is not applied in the adjoint equation
# If the MTK init is called for the reverse, then `y0` in the backwards
# pass will be extremely far off and cause an incorrect gradient
prob_timedepu0 = ODEProblem(
    sys, u0_timedep, tspan, p, jac = true, guesses = [w2 => 0.0],
)
tunables_timedepu0, repack_timedepu0, _ = SS.canonicalize(
    SS.Tunable(), parameter_values(prob_incorrectu0),
)
test_sol = solve(prob_timedepu0, Rodas5P(), abstol = 1.0e-6, reltol = 1.0e-3)

u0_correct = [
    D(x) => 2.0,
    x => 1.0,
    y => 0.0,
    z => 0.0,
]
prob_correctu0 = ODEProblem(
    sys, u0_correct, tspan, p, jac = true, guesses = [w2 => -1.0],
)
tunables_correctu0, repack_correctu0, _ = SS.canonicalize(
    SS.Tunable(), parameter_values(prob_correctu0),
)
test_sol = solve(prob_correctu0, Rodas5P(), abstol = 1.0e-6, reltol = 1.0e-3)
u0_overdetermined = [
    D(x) => 2.0,
    x => 1.0,
    y => 0.0,
    z => 0.0,
    w2 => -1.0,
]
prob_overdetermined = ODEProblem(sys, u0_overdetermined, tspan, p, jac = true)
tunables_overdetermined, repack_overdetermined, _ = SS.canonicalize(
    SS.Tunable(), parameter_values(prob_overdetermined),
)
test_sol = solve(prob_overdetermined, Rodas5P(), abstol = 1.0e-6, reltol = 1.0e-3)

setups = [
    (prob_correctu0, tunables_correctu0, repack_correctu0, CheckInit()), # Source of truth first
    (prob_incorrectu0, tunables_incorrectu0, repack_incorrectu0, BrownFullBasicInit()),
    (prob_incorrectu0, tunables_incorrectu0, repack_incorrectu0, OrdinaryDiffEqCore.DefaultInit()),
    (prob_incorrectu0, tunables_incorrectu0, repack_incorrectu0, nothing),
    (prob_timedepu0, tunables_timedepu0, repack_timedepu0, BrownFullBasicInit()),
    (prob_timedepu0, tunables_timedepu0, repack_timedepu0, OrdinaryDiffEqCore.DefaultInit()),
    (prob_timedepu0, tunables_timedepu0, repack_timedepu0, nothing),
    (prob_correctu0, tunables_correctu0, repack_correctu0, BrownFullBasicInit()),
    (prob_correctu0, tunables_correctu0, repack_correctu0, OrdinaryDiffEqCore.DefaultInit()),
    (prob_correctu0, tunables_correctu0, repack_correctu0, NoInit()),
    (prob_correctu0, tunables_correctu0, repack_correctu0, nothing),
    (prob_overdetermined, tunables_overdetermined, repack_overdetermined, BrownFullBasicInit()),
    (
        prob_overdetermined, tunables_overdetermined, repack_overdetermined,
        OrdinaryDiffEq.OrdinaryDiffEqCore.DefaultInit(),
    ),
    (prob_overdetermined, tunables_overdetermined, repack_overdetermined, NoInit()),
    (prob_overdetermined, tunables_overdetermined, repack_overdetermined, nothing),
]

# Reverse-mode AD through DAE initialization with SCCNonlinearProblem mutation.
# Use the same `Duplicated(prob, diprob)` pattern as the SCC init test in
# `desauty_dae_mwe.jl` (#1454): a plain function `enzyme_solve_loss` (no
# captured-mutable closure) with the `ODEProblem` passed explicitly as
# `Duplicated`. `sensealg` is referenced via a module-level `const` so that
# Enzyme sees it as a `Const`-typed input — passing it as a function argument
# under `set_runtime_activity(Reverse)` causes Enzyme to promote it to
# `Duplicated`, which then doesn't match the `solve_up` Enzyme rule signature
# in `DiffEqBaseEnzymeExt` (which requires `sensealg::Const`). The inner solve
# pins `Rodas5P()` to avoid polyalgorithm Union dispatch.
#
# Status: now `@test` (below). The `MixedDuplicated(::ODESolution)` wall (#1359)
# is fixed upstream in OrdinaryDiffEq#3700; the `solve_up` init-path issues
# (`NonConstantKeywordArgException`, the `FunctionWrappersWrapper` return-type
# mismatch, and the `NonlinearSolvePolyAlgorithm` runtime-activity assertion) are
# fixed in NonlinearSolveBase (`inactive_kwarg`, the `within_autodiff` wrap-skip,
# and `inactive_type`); and the repeated-call state corruption caused by the
# nested `Enzyme.gradient` in `_init_originator_gradient(::EnzymeOriginator)` was
# an upstream Enzyme bug (EnzymeAD/Enzyme.jl#3139, nested reverse-over-BLAS under
# `set_runtime_activity`) fixed as of the `Enzyme` floor in `[compat]`, so that
# init gradient is Enzyme-native again (#1467 routed it through Zygote until the
# upstream fix landed; #1469).
#
# The default `autojacvec` for this in-place MTK DAE resolves to `EnzymeVJP`, but
# its nested `Enzyme.autodiff` corrupts Enzyme state across the repeated calls of
# the `setups` sweep (later setups throw `SingularException` / return zero; #1469).
# `ReverseDiffVJP` is the workaround across the sweep; the `EnzymeVJP` sweep is
# the `@test_broken` at the end of this file.
const _MTK_SENSEALG = GaussAdjoint(; autojacvec = SciMLSensitivity.ReverseDiffVJP())

function _mtk_enzyme_solve_loss_with_init(t, prob_, init_)
    _, repack_, _ = SS.canonicalize(SS.Tunable(), parameter_values(prob_))
    new_prob = remake(prob_; u0 = prob_.u0, p = repack_(t))
    new_sol = solve(
        new_prob, Rodas5P(); initializealg = init_,
        sensealg = _MTK_SENSEALG, abstol = 1.0e-6, reltol = 1.0e-3,
    )
    return sum(new_sol)
end

function _mtk_enzyme_solve_loss_default_init(t, prob_)
    _, repack_, _ = SS.canonicalize(SS.Tunable(), parameter_values(prob_))
    new_prob = remake(prob_; u0 = prob_.u0, p = repack_(t))
    new_sol = solve(
        new_prob, Rodas5P();
        sensealg = _MTK_SENSEALG, abstol = 1.0e-6, reltol = 1.0e-3,
    )
    return sum(new_sol)
end

const _MTK_SENSEALG_ENZYME = GaussAdjoint(; autojacvec = SciMLSensitivity.EnzymeVJP())

function _mtk_enzyme_solve_loss_with_init_enzyme(t, prob_, init_)
    _, repack_, _ = SS.canonicalize(SS.Tunable(), parameter_values(prob_))
    new_prob = remake(prob_; u0 = prob_.u0, p = repack_(t))
    new_sol = solve(
        new_prob, Rodas5P(); initializealg = init_,
        sensealg = _MTK_SENSEALG_ENZYME, abstol = 1.0e-6, reltol = 1.0e-3,
    )
    return sum(new_sol)
end

function _mtk_enzyme_solve_loss_default_init_enzyme(t, prob_)
    _, repack_, _ = SS.canonicalize(SS.Tunable(), parameter_values(prob_))
    new_prob = remake(prob_; u0 = prob_.u0, p = repack_(t))
    new_sol = solve(
        new_prob, Rodas5P();
        sensealg = _MTK_SENSEALG_ENZYME, abstol = 1.0e-6, reltol = 1.0e-3,
    )
    return sum(new_sol)
end

@test begin
    grads = map(setups) do setup
        prob, tunables, repack, init = setup
        diprob = Enzyme.make_zero(prob)
        dtunables = zero(tunables)
        if init === nothing
            Enzyme.autodiff(
                Enzyme.set_runtime_activity(Enzyme.Reverse),
                Enzyme.Const(_mtk_enzyme_solve_loss_default_init), Enzyme.Active,
                Enzyme.Duplicated(tunables, dtunables),
                Enzyme.Duplicated(prob, diprob),
            )
        else
            Enzyme.autodiff(
                Enzyme.set_runtime_activity(Enzyme.Reverse),
                Enzyme.Const(_mtk_enzyme_solve_loss_with_init), Enzyme.Active,
                Enzyme.Duplicated(tunables, dtunables),
                Enzyme.Duplicated(prob, diprob),
                Enzyme.Const(init),
            )
        end
        dtunables
    end
    all(x ≈ grads[1] for x in grads)
end

@test begin
    prob_test, tunables_test, repack_test, init_test = setups[1]
    u0_test = prob_test.u0
    loss = let prob = prob_test, u0 = u0_test, repack = repack_test,
            init = init_test, sensealg = sensealg

        tunables_val -> begin
            new_prob = remake(prob; u0, p = repack(tunables_val))
            new_sol = solve(
                new_prob, Rodas5P(); initializealg = init,
                sensealg, abstol = 1.0e-6, reltol = 1.0e-3,
            )
            sum(new_sol)
        end
    end
    rule = Mooncake.build_rrule(loss, tunables_test)
    _, (_, grad) = Mooncake.value_and_gradient!!(rule, loss, tunables_test)
    any(!iszero, grad)
end

# The first `@test`'s sweep with the default `EnzymeVJP` inner VJP (#1469). Kept
# last so its Enzyme-state corruption cannot affect the tests above.
@test_broken begin
    grads = map(setups) do setup
        prob, tunables, repack, init = setup
        diprob = Enzyme.make_zero(prob)
        dtunables = zero(tunables)
        if init === nothing
            Enzyme.autodiff(
                Enzyme.set_runtime_activity(Enzyme.Reverse),
                Enzyme.Const(_mtk_enzyme_solve_loss_default_init_enzyme), Enzyme.Active,
                Enzyme.Duplicated(tunables, dtunables),
                Enzyme.Duplicated(prob, diprob),
            )
        else
            Enzyme.autodiff(
                Enzyme.set_runtime_activity(Enzyme.Reverse),
                Enzyme.Const(_mtk_enzyme_solve_loss_with_init_enzyme), Enzyme.Active,
                Enzyme.Duplicated(tunables, dtunables),
                Enzyme.Duplicated(prob, diprob),
                Enzyme.Const(init),
            )
        end
        dtunables
    end
    all(x ≈ grads[1] for x in grads)
end
