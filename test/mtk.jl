using ModelingToolkit, OrdinaryDiffEq
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEq
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

gt = rand(1495)
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
    z => 0.0,
    w2 => 0.0,]

p = [σ => 28.0,
    ρ => 10.0,
    β => 8 / 3]

tspan = (0.0, 100.0)

# Check that the gradients for the `solve` are the same both for an initialization
# (for the algebraic variables) initialized poorly (therefore needs correction with BrownBasicInit)
# and with the initialization corrected to satisfy the algebraic equation
prob_incorrectu0 = ODEProblem(sys, u0_incorrect, tspan, p, jac = true)
mtkparams_incorrectu0 = SciMLSensitivity.parameter_values(prob_incorrectu0)

gt = rand(3609)

sensealg = GaussAdjoint(; autojacvec = SciMLSensitivity.ZygoteVJP())
dmtk_incorrectu0, = Zygote.gradient(mtkparams_incorrectu0) do p
    new_sol = solve(prob_incorrectu0, Rodas5P(); p = p, initializealg = BrownFullBasicInit(), sensealg, abstol = 1e-6, reltol = 1e-3)
    Zygote.ChainRules.ChainRulesCore.ignore_derivatives() do
        @test new_sol.retcode == SciMLBase.ReturnCode.Success
        @test all(isapprox.(new_sol[x + y + z + 2 * β - w], 0, atol = 1e-12))
        @test all(isapprox.(new_sol[x^2 + y^2 - w2^2], 0, atol = 1e-5, rtol = 1e0))
    end
    mean(abs.(new_sol[sys.x] .- gt))
end

u0_correct = [D(x) => 2.0,
    x => 1.0,
    y => 0.0,
    z => 0.0,
    w2 => -1.0,]
prob_correctu0 = remake(prob_incorrectu0, u0 = u0_correct)
mtkparams_correctu0 = SciMLSensitivity.parameter_values(prob_correctu0)

dmtk_correctu0, = Zygote.gradient(mtkparams_correctu0) do p
    new_sol = solve(prob_correctu0, Rodas5P(); p = p, initializealg = BrownFullBasicInit(), sensealg, abstol = 1e-6, reltol = 1e-3)
    Zygote.ChainRules.ChainRulesCore.ignore_derivatives() do
        @test new_sol.retcode == SciMLBase.ReturnCode.Success
        @test all(isapprox.(new_sol[x + y + z + 2 * β - w], 0, atol = 1e-12))
        @test all(isapprox.(new_sol[x^2 + y^2 - w2^2], 0, atol = 1e-5, rtol = 1e0))
    end
    mean(abs.(new_sol[sys.x] .- gt))
end

@test dmtk_incorrectu0.tunable ≈ dmtk_correctu0.tunable
