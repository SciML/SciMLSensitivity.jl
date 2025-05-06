using ModelingToolkit, OrdinaryDiffEq
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEq
using OrdinaryDiffEqCore
using SciMLSensitivity
using ForwardDiff
using Zygote
using Statistics
using Test

@parameters σ ρ β A[1:3]
@variables x(t) y(t) z(t) w(t) w2(t)

eqs = [D(D(x)) ~ σ * (y - x),
    D(y) ~ x * (ρ - z) - y,
    D(z) ~ x * y - β * z,
    w ~ x + y + z + 2 * β
]

@mtkbuild sys = ODESystem(eqs, t)

u0 = [D(x) => 2.0,
    x => 1.0,
    y => 0.0,
    z => 0.0]

p = [σ => 28.0,
    ρ => 10.0,
    β => 8 / 3]
# A => ones(3),]

tspan = (0.0, 100.0)
prob = ODEProblem(sys, u0, tspan, p, jac = true)
sol = solve(prob, Tsit5())
mtkparams = SciMLSensitivity.parameter_values(sol)

gt = rand(length(sol.u))
dmtk, = Zygote.gradient(mtkparams) do p
    new_sol = solve(prob, Tsit5(), p = p)
    Zygote.ChainRules.ChainRulesCore.ignore_derivatives() do
        @test all(isapprox.(new_sol[x + y + z + 2 * β - w], 0, atol = 1e-12))
    end
    mean(abs.(new_sol[sys.x] .- gt))
end

# Force DAE handling for Initialization

eqs = [D(D(x)) ~ σ * (y - x),
    D(y) ~ x * (ρ - z) - y,
    D(z) ~ x * y - β * z,
    w ~ x + y + z + 2 * β,
    0 ~ x^2 + y^2 - w2^2
]

@mtkbuild sys = ODESystem(eqs, t)

u0_incorrect = [D(x) => 2.0,
    x => 1.0,
    y => 0.0,
    z => 0.0]

p = [σ => 28.0,
    ρ => 10.0,
    β => 8 / 3]

tspan = (0.0, 100.0)

# Check that the gradients for the `solve` are the same both for an initialization
# (for the algebraic variables) initialized poorly (therefore needs correction with BrownBasicInit)
# and with the initialization corrected to satisfy the algebraic equation
prob_incorrectu0 = ODEProblem(sys, u0_incorrect, tspan, p, jac = true, guesses = [w2 => 0.0])
mtkparams_incorrectu0 = SciMLSensitivity.parameter_values(prob_incorrectu0)
test_sol = solve(prob_incorrectu0, Rodas5P(), abstol = 1e-6, reltol = 1e-3)

u0_timedep = [D(x) => 2.0,
    x => 1.0,
    y => t,
    z => 0.0]
# this ensures that `y => t` is not applied in the adjoint equation
# If the MTK init is called for the reverse, then `y0` in the backwards
# pass will be extremely far off and cause an incorrect gradient
prob_timedepu0 = ODEProblem(sys, u0_timedep, tspan, p, jac = true, guesses = [w2 => 0.0])
mtkparams_timedepu0 = SciMLSensitivity.parameter_values(prob_incorrectu0)
test_sol = solve(prob_timedepu0, Rodas5P(), abstol = 1e-6, reltol = 1e-3)

u0_correct = [D(x) => 2.0,
    x => 1.0,
    y => 0.0,
    z => 0.0,]
prob_correctu0 = ODEProblem(sys, u0_correct, tspan, p, jac = true, guesses = [w2 => -1.0])
mtkparams_correctu0 = SciMLSensitivity.parameter_values(prob_correctu0)
test_sol = solve(prob_correctu0, Rodas5P(), abstol = 1e-6, reltol = 1e-3)

u0_gt = [D(x) => 2.0,
    x => 1.0,
    y => 0.0,
    z => 0.0,
    w2 => -1.0,]
# The initialization might be over determined, but that is necessary as we
# will force DAE initialization to not run via CheckInit. CheckInit will
# still need to check that the algebraic equations are satisfied, so we need to
# make sure that the initialization is correct
prob_gtu0 = ODEProblem(sys, u0_gt, tspan, p, jac = true)
mtkparams_gtu0 = SciMLSensitivity.parameter_values(prob_gtu0)
test_sol = solve(prob_gtu0, Rodas5P(), abstol = 1e-6, reltol = 1e-3)

u0_overdetermined = [D(x) => 2.0,
    x => 1.0,
    y => 0.0,
    z => 0.0,
    w2 => -1.0,]
prob_overdetermined = ODEProblem(sys, u0_overdetermined, tspan, p, jac = true)
mtkparams_overdetermined = SciMLSensitivity.parameter_values(prob_overdetermined)
test_sol = solve(prob_overdetermined, Rodas5P(), abstol = 1e-6, reltol = 1e-3)

sensealg = GaussAdjoint(; autojacvec = SciMLSensitivity.ZygoteVJP())

setups = [
          (prob_gtu0, mtkparams_gtu0, CheckInit()), # Source of truth first
    
          (prob_incorrectu0, mtkparams_incorrectu0, BrownFullBasicInit()),
          (prob_incorrectu0, mtkparams_incorrectu0, OrdinaryDiffEqCore.DefaultInit()),
          (prob_incorrectu0, mtkparams_incorrectu0, nothing), 

          (prob_timedepu0, mtkparams_timedepu0, BrownFullBasicInit()),
          (prob_timedepu0, mtkparams_timedepu0, OrdinaryDiffEqCore.DefaultInit()),
          (prob_timedepu0, mtkparams_timedepu0, nothing), 

          (prob_correctu0, mtkparams_correctu0, BrownFullBasicInit()),
          (prob_correctu0, mtkparams_correctu0, OrdinaryDiffEqCore.DefaultInit()),
          
        (prob_correctu0, mtkparams_correctu0, NoInit()),
          (prob_correctu0, mtkparams_correctu0, nothing),

          (prob_overdetermined, mtkparams_overdetermined, BrownFullBasicInit()),
          (prob_overdetermined, mtkparams_overdetermined, OrdinaryDiffEq.OrdinaryDiffEqCore.DefaultInit()),

          (prob_overdetermined, mtkparams_overdetermined, NoInit()),
          (prob_overdetermined, mtkparams_overdetermined, nothing),
];

grads = map(setups) do setup
    prob, ps, init = setup
    @show init
    u0 = prob.u0
    Zygote.gradient(u0, ps) do u0,p
        new_prob = remake(prob, u0 = u0, p = p)
        if init === nothing
            new_sol = solve(new_prob, Rodas5P(); sensealg, abstol = 1e-6, reltol = 1e-3)
        else
            new_sol = solve(new_prob, Rodas5P(); initializealg = init, sensealg, abstol = 1e-6, reltol = 1e-3)
        end
        gt = Zygote.ChainRules.ChainRulesCore.ignore_derivatives() do
            @test new_sol.retcode == SciMLBase.ReturnCode.Success
            # Test that beginning of forward pass init'd correctly
            @test all(isapprox.(new_sol[x + y + z + 2 * β - w], 0, atol = 1e-12))
            @test all(isapprox.(new_sol[x^2 + y^2 - w2^2], 0, atol = 1e-5, rtol = 1e0))
            zeros(size(new_sol, 2))
        end
        mean(abs.(new_sol[sys.x] .- gt))
    end
end

u0grads = getindex.(grads,1)
pgrads = getproperty.(getindex.(grads, 2), (:tunable,))
@test all(x ≈ u0grads[1] for x in u0grads)
@test all(x ≈ pgrads[1] for x in pgrads)
