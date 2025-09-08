using LinearAlgebra
using SparseArrays
using OrdinaryDiffEq
using SciMLSensitivity
using SciMLOperators
using Enzyme

const T = ComplexF64

coef1(a, u, p, t) = -p[1]
coef2(a, u, p, t) = p[2]

A1_data = sparse(T[0.0 1.0; 0.0 0.0])
A2_data = sparse(T[0.0 0.0; 1.0 0.0])
c1 = ScalarOperator(one(T), coef1)
c2 = ScalarOperator(one(T), coef2)

const U = c1 * MatrixOperator(A1_data) + c2 * MatrixOperator(A2_data)

function my_fun_works(p)
    x = T[3.0, 4.0]

    prob = ODEProblem{true}(U, x, (0.0, 1.0), p)

    sol = solve(prob, Tsit5(), save_everystep=false, 
        sensealg=BacksolveAdjoint(autojacvec=EnzymeVJP()) # This works
    )

    return real(sol.u[end][end])
end

function my_fun_fails(p)
    x = T[3.0, 4.0]

    prob = ODEProblem{true}(U, x, (0.0, 1.0), p, 
        sensealg=BacksolveAdjoint(autojacvec=EnzymeVJP()) # This fails 
    )

    sol = solve(prob, Tsit5(), save_everystep=false)

    return real(sol.u[end][end])
end

p = [1.0, 2.0]

# Test the working case
println("Testing working case (sensealg in solve)...")
dp_works = Enzyme.make_zero(p)
try
    Enzyme.autodiff(Enzyme.set_runtime_activity(Enzyme.Reverse), my_fun_works, Active, Duplicated(p, dp_works))
    println("Success: dp_works = ", dp_works)
catch e
    println("Error in working case: ", e)
end

# Test the failing case
println("\nTesting failing case (sensealg in ODEProblem)...")
dp_fails = Enzyme.make_zero(p)
try
    Enzyme.autodiff(Enzyme.set_runtime_activity(Enzyme.Reverse), my_fun_fails, Active, Duplicated(p, dp_fails))
    println("Success: dp_fails = ", dp_fails)
catch e
    println("Error in failing case: ", e)
end