using SciMLSensitivity, OrdinaryDiffEq, LinearAlgebra, SparseArrays, Zygote, LinearSolve
using AlgebraicMultigrid
using Test

foop(u, p, t) = jac(u, p, t) * u
jac(u, p, t) = spdiagm(0 => p)
paramjac(u, p, t) = SparseArrays.spdiagm(0 => u)
Zygote.@adjoint function foop(u, p, t)
    foop(u, p, t),
    delta -> (jac(u, p, t)' * delta, paramjac(u, p, t)' * delta, zeros(length(u)))
end

n = 2
p = collect(1.0:n)
u0 = ones(n)
tspan = [0.0, 1]
odef = ODEFunction(foop; jac = jac, jac_prototype = jac(u0, p, 0.0), paramjac = paramjac)
function g_helper(p; alg = Rosenbrock23(linsolve = LUFactorization()))
    prob = ODEProblem(odef, u0, tspan, p)
    soln = Array(solve(prob, alg; u0 = prob.u0, p = prob.p, abstol = 1e-4, reltol = 1e-4,
        sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP())))[:, end]
    return soln
end
function g(p; kwargs...)
    soln = g_helper(p; kwargs...)
    return sum(soln)
end
@test isapprox(exp.(p), g_helper(p); atol = 1e-3, rtol = 1e-3)
@test isapprox(exp.(p), Zygote.gradient(g, p)[1]; atol = 1e-3, rtol = 1e-3)
@test isapprox(exp.(p), g_helper(p; alg = Rosenbrock23(linsolve = KLUFactorization()));
    atol = 1e-3, rtol = 1e-3)
@test isapprox(exp.(p),
    Zygote.gradient(p -> g(p; alg = Rosenbrock23(linsolve = KLUFactorization())),
        p)[1]; atol = 1e-3, rtol = 1e-3)
@test isapprox(exp.(p), g_helper(p; alg = ImplicitEuler(linsolve = LUFactorization()));
    atol = 1e-1, rtol = 1e-1)
@test isapprox(exp.(p),
    Zygote.gradient(p -> g(p; alg = ImplicitEuler(linsolve = LUFactorization())),
        p)[1]; atol = 1e-1, rtol = 1e-1)
@test isapprox(
    exp.(p), g_helper(p; alg = ImplicitEuler(linsolve = UMFPACKFactorization()));
    atol = 1e-1, rtol = 1e-1)
@test isapprox(exp.(p),
    Zygote.gradient(p -> g(p;
            alg = ImplicitEuler(linsolve = UMFPACKFactorization())),
        p)[1]; atol = 1e-1, rtol = 1e-1)
@test isapprox(exp.(p), g_helper(p; alg = ImplicitEuler(linsolve = KrylovJL_GMRES()));
    atol = 1e-1, rtol = 1e-1)
@test isapprox(exp.(p),
    Zygote.gradient(p -> g(p; alg = ImplicitEuler(linsolve = KrylovJL_GMRES())),
        p)[1]; atol = 1e-1, rtol = 1e-1)
