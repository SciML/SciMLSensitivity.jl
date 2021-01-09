using DiffEqSensitivity, OrdinaryDiffEq, LinearAlgebra, SparseArrays, Zygote
using AlgebraicMultigrid: AlgebraicMultigrid
using Test

foop(u, p, t) = jac(u, p, t) * u
jac(u, p, t) = spdiagm(0=>p)
paramjac(u, p, t) = SparseArrays.spdiagm(0=>u)
@Zygote.adjoint foop(u, p, t) = foop(u, p, t), delta->(jac(u, p, t)' * delta, paramjac(u, p, t)' * delta, zeros(length(u)))

function linsolve_amg!(::Type{Val{:init}}, f, u0, kwargs...)
    function _linsolve!(x, A, b, update_matrix=false, args...; kwargs...)
        rs = AlgebraicMultigrid.ruge_stuben(A)
        copy!(x, AlgebraicMultigrid.solve(rs, b))
    end
end

linsolve_lu = LinSolveFactorize(lu)

n = 2
p = collect(1.0:n)
u0 = ones(n)
tspan = [0.0, 1]
odef = ODEFunction(foop; jac=jac, jac_prototype=jac(u0, p, 0.0), paramjac=paramjac)
function g_helper(p; alg=Rosenbrock23(linsolve=linsolve_lu))
    prob = ODEProblem(odef, u0, tspan, p)
    soln = Array(solve(prob, alg; u0=prob.u0, p=prob.p, abstol=1e-4, reltol=1e-4))[:, end]
    return soln
end
function g(p; kwargs...)
    soln = g_helper(p; kwargs...)
    return sum(soln)
end
@test isapprox(exp.(p), g_helper(p); atol=1e-3, rtol=1e-3)
@test isapprox(exp.(p), Zygote.gradient(g, p)[1]; atol=1e-3, rtol=1e-3)#passes but does not use sparse matrices on the backward pass
@test isapprox(exp.(p), g_helper(p; alg=Rosenbrock23(linsolve=linsolve_amg!)); atol=1e-3, rtol=1e-3)
@test isapprox(exp.(p), Zygote.gradient(p->g(p; alg=Rosenbrock23(linsolve=linsolve_amg!)), p)[1]; atol=1e-3, rtol=1e-3)
@test isapprox(exp.(p), g_helper(p; alg=ImplicitEuler(linsolve=linsolve_lu)); atol=1e-1, rtol=1e-1)
@test isapprox(exp.(p), Zygote.gradient(p->g(p; alg=ImplicitEuler(linsolve=linsolve_lu)), p)[1]; atol=1e-1, rtol=1e-1)
@test isapprox(exp.(p), g_helper(p; alg=ImplicitEuler(linsolve=linsolve_amg!)); atol=1e-1, rtol=1e-1)
@test isapprox(exp.(p), Zygote.gradient(p->g(p; alg=ImplicitEuler(linsolve=linsolve_amg!)), p)[1]; atol=1e-1, rtol=1e-1)
