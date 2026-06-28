using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEqRosenbrock
using OrdinaryDiffEqNonlinearSolve: BrownFullBasicInit
using SciMLBase: CheckInit
using SciMLSensitivity
import SciMLStructures as SS
using SciMLSensitivity: parameter_values
using Enzyme
using Test

# Regression test for #1470. Repeatedly differentiating a `solve` of an MTK DAE
# with `GaussAdjoint(EnzymeVJP())` must not corrupt the problem across calls. The
# EnzymeVJP shadow parameters (`adjoint_common.jl`) used to be built with
# `repack(zero(tunables))`, which aliases the primal's non-tunable arrays
# (`initials`) by reference; `_vecjacobian!` then `remake_zero!`s the shadow,
# zeroing the user's `p.initials`. The first gradient was correct, but the next
# solve's DAE init read the corrupted `Initials` and threw `CheckInitFailureError`.
# Fixed by allocating a disjoint shadow with `make_zero(p)`; every repeated
# differentiation is now correct and identical.

@parameters σ ρ β
@variables x(t) y(t) z(t) w(t) w2(t)
eqs = [
    D(D(x)) ~ σ * (y - x), D(y) ~ x * (ρ - z) - y, D(z) ~ x * y - β * z,
    w ~ x + y + z + 2β, 0 ~ x^2 + y^2 - w2^2,
]
@mtkbuild sys = ODESystem(eqs, t)
prob = ODEProblem(
    sys, [D(x) => 2.0, x => 1.0, y => 0.0, z => 0.0], (0.0, 5.0),
    [σ => 28.0, ρ => 10.0, β => 8 / 3]; jac = true, guesses = [w2 => -1.0]
)
tunables, repack, _ = SS.canonicalize(SS.Tunable(), parameter_values(prob))

const _RA_SENSEALG = GaussAdjoint(; autojacvec = SciMLSensitivity.EnzymeVJP())

function _ra_loss(tun, prob_)
    _, rp, _ = SS.canonicalize(SS.Tunable(), parameter_values(prob_))
    new_prob = remake(prob_; u0 = prob_.u0, p = rp(tun))
    new_sol = solve(
        new_prob, Rodas5P(); initializealg = CheckInit(), sensealg = _RA_SENSEALG,
        abstol = 1.0e-6, reltol = 1.0e-3,
    )
    return sum(new_sol)
end

function _ra_grad()
    dtun = zero(tunables)
    diprob = Enzyme.make_zero(prob)
    Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        Enzyme.Const(_ra_loss), Enzyme.Active,
        Enzyme.Duplicated(copy(tunables), dtun),
        Enzyme.Duplicated(prob, diprob),
    )
    return dtun
end

g1 = _ra_grad()
@test any(!iszero, g1)
# Without the parameter copy these repeated calls throw `CheckInitFailureError`.
g2 = _ra_grad()
@test g2 ≈ g1
g3 = _ra_grad()
@test g3 ≈ g1
