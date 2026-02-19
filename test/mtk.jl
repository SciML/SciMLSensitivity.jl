using ModelingToolkit, OrdinaryDiffEq
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEq
using OrdinaryDiffEqCore
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
# Marked as broken until Enzyme/Mooncake fully support this pattern.
@test_broken begin
    grads = map(setups) do setup
        prob, tunables, repack, init = setup
        u0 = prob.u0
        loss = let prob = prob, u0 = u0, repack = repack, init = init, sensealg = sensealg
            tunables_val -> begin
                new_prob = remake(prob; u0, p = repack(tunables_val))
                if init === nothing
                    new_sol = solve(
                        new_prob, Rodas5P(); sensealg, abstol = 1.0e-6, reltol = 1.0e-3,
                    )
                else
                    new_sol = solve(
                        new_prob, Rodas5P(); initializealg = init,
                        sensealg, abstol = 1.0e-6, reltol = 1.0e-3,
                    )
                end
                sum(new_sol)
            end
        end
        Enzyme.gradient(Enzyme.Reverse, loss, tunables)
    end
    all(x ≈ grads[1] for x in grads)
end

@test_broken begin
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
