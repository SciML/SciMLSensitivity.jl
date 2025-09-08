using Pkg
Pkg.activate(".")

using OrdinaryDiffEq
using SciMLSensitivity
using Enzyme

# Simple test case
function f(du, u, p, t)
    du[1] = -p[1] * u[1]
    du[2] = p[2] * u[2]
end

function my_fun_works(p)
    u0 = [1.0, 2.0]
    prob = ODEProblem(f, u0, (0.0, 1.0), p)
    sol = solve(prob, Tsit5(), sensealg=BacksolveAdjoint(autojacvec=EnzymeVJP()))
    return sol.u[end][1] + sol.u[end][2]
end

function my_fun_fails(p)
    u0 = [1.0, 2.0]
    prob = ODEProblem(f, u0, (0.0, 1.0), p, sensealg=BacksolveAdjoint(autojacvec=EnzymeVJP()))
    sol = solve(prob, Tsit5())
    return sol.u[end][1] + sol.u[end][2]
end

p = [0.5, 1.5]

println("Testing working case (sensealg in solve)...")
dp_works = Enzyme.make_zero(p)
try
    Enzyme.autodiff(Enzyme.Reverse, my_fun_works, Enzyme.Active, Enzyme.Duplicated(p, dp_works))
    println("SUCCESS: dp_works = ", dp_works)
catch e
    println("ERROR in working case: ", e)
end

println("\nTesting failing case (sensealg in ODEProblem)...")
dp_fails = Enzyme.make_zero(p)
try
    Enzyme.autodiff(Enzyme.Reverse, my_fun_fails, Enzyme.Active, Enzyme.Duplicated(p, dp_fails))
    println("SUCCESS: dp_fails = ", dp_fails)
    println("Fix worked!")
catch e
    println("ERROR in failing case: ", e)
end