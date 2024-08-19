using ModelingToolkit, OrdinaryDiffEq
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEq
using SciMLSensitivity
using ForwardDiff
using Zygote
using Statistics

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

gt = rand(5501)
dmtk, = Zygote.gradient(mtkparams) do p
    new_sol = solve(prob, Rosenbrock23(), p = p)
    mean(abs.(new_sol[sys.x] .- gt))
end
