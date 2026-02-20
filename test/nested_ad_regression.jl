using OrdinaryDiffEq, SciMLSensitivity, Test, ADTypes, Reactant
function f!(du, u::AbstractArray{T}, p, x) where {T}
    du[1] = -p[1] * exp((x - 8)) * u[1]
    return nothing
end

# primal calculation
p = [1.0]
u0 = [1.0]
xspan = 0.0, 20.0
prob = ODEProblem{true}(f!, u0, xspan, p)
sol = solve(prob, KenCarp4(), abstol = 1.0e-6, reltol = 1.0e-6)

# sensitivity
g(u, p, t) = (sum(u) .^ 2) ./ 2
dg(out, u, p, t) = (out[1] = u[1])

adj_prob = ODEAdjointProblem(
    sol,
    QuadratureAdjoint(autojacvec = EnzymeVJP()),
    KenCarp4(),
    nothing, nothing, nothing, dg, nothing, g
)
adj_sol = solve(adj_prob, KenCarp4())
@test length(adj_sol.t) < 300
adj_sol2 = solve(adj_prob, KenCarp4(autodiff = AutoFiniteDiff()))
@test abs(length(adj_sol.t) - length(adj_sol2.t)) < 20

adj_prob2 = ODEAdjointProblem(
    sol,
    QuadratureAdjoint(autojacvec = ReverseDiffVJP(true)),
    KenCarp4(),
    nothing, nothing, nothing, dg, nothing, g
)
adj_sol3 = solve(adj_prob, KenCarp4(autodiff = AutoFiniteDiff()))
@test abs(length(adj_sol.t) - length(adj_sol3.t)) < 20

res2 = adjoint_sensitivities(
    sol, KenCarp4(); dgdu_continuous = dg, g,
    abstol = 1.0e-6, reltol = 1.0e-6, sensealg = QuadratureAdjoint(autojacvec = ReverseDiffVJP(true))
);

res1 = adjoint_sensitivities(
    sol, KenCarp4(); dgdu_continuous = dg, g,
    abstol = 1.0e-6, reltol = 1.0e-6, sensealg = QuadratureAdjoint(autojacvec = EnzymeVJP())
);

res3 = adjoint_sensitivities(
    sol, KenCarp4(); dgdu_continuous = dg, g,
    abstol = 1.0e-6, reltol = 1.0e-6, sensealg = QuadratureAdjoint(autojacvec = ReactantVJP())
);

@test res1[1] ≈ res2[1]
@test res1[2] ≈ res2[2]
@test res1[1] ≈ res3[1]
@test res1[2] ≈ res3[2]
