# Regression test for https://github.com/SciML/SciMLSensitivity.jl/issues/1365
# ForwardSensitivity with sparse jac_prototype and sparse linear solver
using OrdinaryDiffEq, SciMLSensitivity
using SparseArrays, LinearSolve, Zygote
using Test

function rhs!(du, u, p, t)
    du .= -p .* u
    return nothing
end

params = [4.0, 5.5, 3.0]
tspan = (0.0, 10.0)
u0 = ones(Float64, 3)

# Sparse jac_prototype
jac0 = spzeros(length(u0), length(u0))
jac0[1:3, 1:3] .= 1.0

fun_sparse = ODEFunction(rhs!; jac_prototype = jac0)
prob_sparse = ODEProblem(fun_sparse, u0, tspan)

# Dense reference
prob_dense = ODEProblem(rhs!, u0, tspan)

sense_alg = ForwardSensitivity(autodiff = true, autojacvec = true)

function loss_sparse(p)
    new_prob = remake(prob_sparse, p = p)
    sol = solve(new_prob, FBDF(autodiff = false, linsolve = UMFPACKFactorization()),
        abstol = 1e-8, reltol = 1e-6, saveat = 1.0, sensealg = sense_alg)
    return sum(sum.(sol.u))
end

function loss_dense(p)
    new_prob = remake(prob_dense, p = p)
    sol = solve(new_prob, FBDF(autodiff = false),
        abstol = 1e-8, reltol = 1e-6, saveat = 1.0, sensealg = sense_alg)
    return sum(sum.(sol.u))
end

grad_sparse = Zygote.gradient(loss_sparse, params)[1]
grad_dense = Zygote.gradient(loss_dense, params)[1]

@test grad_sparse≈grad_dense rtol=1e-6
