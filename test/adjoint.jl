using SciMLSensitivity, OrdinaryDiffEq, RecursiveArrayTools, DiffEqBase,
    ForwardDiff, Calculus, QuadGK, LinearAlgebra, Zygote, Mooncake, ADTypes
using Test

function fb(du, u, p, t)
    du[1] = dx = p[1] * u[1] - p[2] * u[1] * u[2] * t
    du[2] = dy = -p[3] * u[2] + t * p[4] * u[1] * u[2]
    return nothing
end
function foop(u, p, t)
    dx = p[1] * u[1] - p[2] * u[1] * u[2] * t
    dy = -p[3] * u[2] + t * p[4] * u[1] * u[2]
    return [dx, dy]
end
function jac(J, u, p, t)
    (x, y, a, b, c, d) = (u[1], u[2], p[1], p[2], p[3], p[4])
    J[1, 1] = a + y * b * -1 * t
    J[2, 1] = t * y * d
    J[1, 2] = b * x * -1 * t
    J[2, 2] = c * -1 + t * x * d
    return nothing
end

f = ODEFunction(fb; jac)
p = [1.5, 1.0, 3.0, 1.0];
u0 = [1.0; 1.0];
prob = ODEProblem(f, u0, (0.0, 10.0), p)
sol = solve(prob, Tsit5(), abstol = 1.0e-14, reltol = 1.0e-14)
probb = ODEProblem(fb, u0, (0.0, 10.0), p)
proboop = ODEProblem(foop, u0, (0.0, 10.0), p)

solb = solve(probb, Tsit5(), abstol = 1.0e-14, reltol = 1.0e-14)
sol_end = solve(
    probb, Tsit5(), abstol = 1.0e-14, reltol = 1.0e-14,
    save_everystep = false, save_start = false
)

sol_nodense = solve(probb, Tsit5(), abstol = 1.0e-14, reltol = 1.0e-14, dense = false)
soloop = solve(proboop, Tsit5(), abstol = 1.0e-14, reltol = 1.0e-14)
soloop_nodense = solve(proboop, Tsit5(), abstol = 1.0e-14, reltol = 1.0e-14, dense = false)

# Do a discrete adjoint problem
println("Calculate discrete adjoint sensitivities")
t = 0.0:0.5:10.0
# g(t,u,i) = (1-u)^2/2, L2 away from 1
function dg(out, u, p, t, i)
    return (out .= -2.0 .+ u)
end

_,
    easy_res = adjoint_sensitivities(
    sol, Tsit5(); t, dgdu_discrete = dg, abstol = 1.0e-14,
    reltol = 1.0e-14
)
_,
    easy_res2 = adjoint_sensitivities(
    solb, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = QuadratureAdjoint(abstol = 1.0e-14, reltol = 1.0e-14)
)
_,
    easy_res22 = adjoint_sensitivities(
    solb, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = QuadratureAdjoint(
        autojacvec = false,
        abstol = 1.0e-14,
        reltol = 1.0e-14
    )
)
_,
    easy_res23 = adjoint_sensitivities(
    solb, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = QuadratureAdjoint(
        abstol = 1.0e-14,
        reltol = 1.0e-14,
        autojacvec = ReverseDiffVJP(true)
    )
)
_,
    easy_res3 = adjoint_sensitivities(
    solb, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = InterpolatingAdjoint()
)
_,
    easy_res32 = adjoint_sensitivities(
    solb, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = InterpolatingAdjoint(autojacvec = false)
)
_,
    easy_res4 = adjoint_sensitivities(
    solb, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = BacksolveAdjoint()
)
_,
    easy_res42 = adjoint_sensitivities(
    solb, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = BacksolveAdjoint(autojacvec = false)
)
_,
    easy_res43 = adjoint_sensitivities(
    solb, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = BacksolveAdjoint(autojacvec = false, checkpointing = false)
)
_,
    easy_res5 = adjoint_sensitivities(
    sol,
    Kvaerno5(nlsolve = NLAnderson(), smooth_est = false);
    t, dgdu_discrete = dg, abstol = 1.0e-12,
    reltol = 1.0e-10,
    sensealg = BacksolveAdjoint()
)
_,
    easy_res6 = adjoint_sensitivities(
    sol_nodense, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = InterpolatingAdjoint(checkpointing = true),
    checkpoints = sol.t[1:10:end]
)
_,
    easy_res62 = adjoint_sensitivities(
    sol_nodense, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = InterpolatingAdjoint(checkpointing = true, autojacvec = false),
    checkpoints = sol.t[1:500:end]
)
# It should automatically be checkpointing since the solution isn't dense
_,
    easy_res7 = adjoint_sensitivities(
    sol_nodense, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = InterpolatingAdjoint(),
    checkpoints = sol.t[1:500:end]
)
_,
    easy_res8 = adjoint_sensitivities(
    solb, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = InterpolatingAdjoint(autojacvec = SciMLSensitivity.TrackerVJP())
)
_,
    easy_res9 = adjoint_sensitivities(
    solb, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = InterpolatingAdjoint(autojacvec = SciMLSensitivity.ZygoteVJP())
)
_,
    easy_res10 = adjoint_sensitivities(
    solb, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = InterpolatingAdjoint(autojacvec = SciMLSensitivity.ReverseDiffVJP())
)
_,
    easy_res11 = adjoint_sensitivities(
    solb, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = InterpolatingAdjoint(autojacvec = SciMLSensitivity.ReverseDiffVJP(true))
)
_,
    easy_res12 = adjoint_sensitivities(
    solb, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = InterpolatingAdjoint(autojacvec = SciMLSensitivity.EnzymeVJP())
)
_,
    easy_res13 = adjoint_sensitivities(
    solb, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = QuadratureAdjoint(autojacvec = SciMLSensitivity.EnzymeVJP())
)
_,
    easy_res14 = adjoint_sensitivities(
    solb, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = GaussAdjoint()
)
_,
    easy_res14k = adjoint_sensitivities(
    solb, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = GaussKronrodAdjoint()
)
_,
    easy_res15 = adjoint_sensitivities(
    solb, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = InterpolatingAdjoint(autojacvec = SciMLSensitivity.MooncakeVJP())
)
_,
    easy_res16 = adjoint_sensitivities(
    solb, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = QuadratureAdjoint(autojacvec = SciMLSensitivity.MooncakeVJP())
)
_,
    easy_res142 = adjoint_sensitivities(
    solb, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = GaussAdjoint(autojacvec = false)
)
_,
    easy_res143 = adjoint_sensitivities(
    solb, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = GaussAdjoint(autojacvec = ReverseDiffVJP(true))
)
_,
    easy_res144 = adjoint_sensitivities(
    solb, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = GaussAdjoint(autojacvec = SciMLSensitivity.EnzymeVJP())
)
_,
    easy_res145 = adjoint_sensitivities(
    sol_nodense, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = GaussAdjoint(checkpointing = true),
    checkpoints = sol.t[1:500:end]
)
_,
    easy_res146 = adjoint_sensitivities(
    sol_nodense, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = GaussAdjoint(
        checkpointing = true,
        autojacvec = false
    ),
    checkpoints = sol.t[1:500:end]
)
_,
    easy_res147 = adjoint_sensitivities(
    solb, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = GaussAdjoint(autojacvec = SciMLSensitivity.MooncakeVJP())
)
_,
    easy_res142k = adjoint_sensitivities(
    solb, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = GaussKronrodAdjoint(autojacvec = false)
)
_,
    easy_res143k = adjoint_sensitivities(
    solb, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = GaussKronrodAdjoint(autojacvec = ReverseDiffVJP(true))
)
_,
    easy_res144k = adjoint_sensitivities(
    solb, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = GaussKronrodAdjoint(autojacvec = SciMLSensitivity.EnzymeVJP())
)
_,
    easy_res145k = adjoint_sensitivities(
    sol_nodense, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = GaussKronrodAdjoint(checkpointing = true),
    checkpoints = sol.t[1:500:end]
)
_,
    easy_res146k = adjoint_sensitivities(
    sol_nodense, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = GaussKronrodAdjoint(
        checkpointing = true,
        autojacvec = false
    ),
    checkpoints = sol.t[1:500:end]
)
_,
    easy_res147k = adjoint_sensitivities(
    solb, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = GaussKronrodAdjoint(autojacvec = SciMLSensitivity.MooncakeVJP())
)
adj_prob = ODEAdjointProblem(
    sol,
    QuadratureAdjoint(
        abstol = 1.0e-14, reltol = 1.0e-14,
        autojacvec = SciMLSensitivity.ReverseDiffVJP()
    ),
    Tsit5(),
    t, dg
)
adj_sol = solve(adj_prob, Tsit5(), abstol = 1.0e-14, reltol = 1.0e-14)
integrand = AdjointSensitivityIntegrand(
    sol, adj_sol,
    QuadratureAdjoint(
        abstol = 1.0e-14, reltol = 1.0e-14,
        autojacvec = SciMLSensitivity.ReverseDiffVJP()
    )
)
res, err = quadgk(integrand, 0.0, 10.0, atol = 1.0e-14, rtol = 1.0e-12)

@test isapprox(res, easy_res, rtol = 1.0e-10)
@test isapprox(res, easy_res2, rtol = 1.0e-10)
@test isapprox(res, easy_res22, rtol = 1.0e-10)
@test isapprox(res, easy_res23, rtol = 1.0e-10)
@test isapprox(res, easy_res3, rtol = 1.0e-10)
@test isapprox(res, easy_res32, rtol = 1.0e-10)
@test isapprox(res, easy_res4, rtol = 1.0e-10)
@test isapprox(res, easy_res42, rtol = 1.0e-10)
@test isapprox(res, easy_res43, rtol = 1.0e-10)
@test isapprox(res, easy_res5, rtol = 1.0e-7)
@test isapprox(res, easy_res6, rtol = 1.0e-9)
@test isapprox(res, easy_res62, rtol = 1.0e-9)
@test isapprox(easy_res6, easy_res7, rtol = 1.0e-9)
@test isapprox(res, easy_res8, rtol = 1.0e-9)
@test isapprox(res, easy_res9, rtol = 1.0e-9)
@test isapprox(res, easy_res10, rtol = 1.0e-9)
@test isapprox(res, easy_res11, rtol = 1.0e-9)
@test isapprox(res, easy_res12, rtol = 1.0e-9)
@test isapprox(res, easy_res13, rtol = 1.0e-9)
@test isapprox(res, easy_res14, rtol = 1.0e-9)
@test isapprox(res, easy_res14k, rtol = 1.0e-9)
@test isapprox(res, easy_res15, rtol = 1.0e-9)
@test isapprox(res, easy_res16, rtol = 1.0e-9)
@test isapprox(res, easy_res142, rtol = 1.0e-9)
@test isapprox(res, easy_res143, rtol = 1.0e-9)
@test isapprox(res, easy_res144, rtol = 1.0e-9)
@test isapprox(res, easy_res145, rtol = 1.0e-9)
@test isapprox(res, easy_res146, rtol = 1.0e-9)
@test isapprox(res, easy_res147, rtol = 1.0e-9)
@test isapprox(res, easy_res142k, rtol = 1.0e-9)
@test isapprox(res, easy_res143k, rtol = 1.0e-9)
@test isapprox(res, easy_res144k, rtol = 1.0e-9)
@test isapprox(res, easy_res145k, rtol = 1.0e-9)
@test isapprox(res, easy_res146k, rtol = 1.0e-9)
@test isapprox(res, easy_res147k, rtol = 1.0e-9)

println("OOP adjoint sensitivities ")

_,
    easy_res = adjoint_sensitivities(
    soloop, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14
)
_,
    easy_res2 = adjoint_sensitivities(
    soloop, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = QuadratureAdjoint(abstol = 1.0e-14, reltol = 1.0e-14)
)
_,
    easy_res22 = adjoint_sensitivities(
    soloop, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = QuadratureAdjoint(autojacvec = false, abstol = 1.0e-14, reltol = 1.0e-14)
)
_,
    easy_res2 = adjoint_sensitivities(
    soloop, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = QuadratureAdjoint(
        abstol = 1.0e-14,
        reltol = 1.0e-14,
        autojacvec = ReverseDiffVJP(true)
    )
)
_,
    easy_res3 = adjoint_sensitivities(
    soloop, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = InterpolatingAdjoint()
)
@test easy_res32 = adjoint_sensitivities(
    soloop, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = InterpolatingAdjoint(autojacvec = false)
)[1] isa AbstractArray
_,
    easy_res4 = adjoint_sensitivities(
    soloop, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = BacksolveAdjoint()
)
@test easy_res42 = adjoint_sensitivities(
    soloop, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = BacksolveAdjoint(autojacvec = false)
)[1] isa AbstractArray
_,
    easy_res5 = adjoint_sensitivities(
    soloop,
    Kvaerno5(nlsolve = NLAnderson(), smooth_est = false);
    t, dgdu_discrete = dg,
    abstol = 1.0e-12,
    reltol = 1.0e-10,
    sensealg = BacksolveAdjoint()
)
_,
    easy_res6 = adjoint_sensitivities(
    soloop_nodense, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = InterpolatingAdjoint(checkpointing = true),
    checkpoints = soloop_nodense.t[1:5:end]
)
_,
    easy_res62 = adjoint_sensitivities(
    soloop_nodense, Tsit5(); t,
    dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = InterpolatingAdjoint(checkpointing = true, autojacvec = false),
    checkpoints = soloop_nodense.t[1:5:end]
)

_,
    easy_res8 = adjoint_sensitivities(
    soloop_nodense, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = InterpolatingAdjoint(autojacvec = SciMLSensitivity.TrackerVJP())
)
_,
    easy_res9 = adjoint_sensitivities(
    soloop_nodense, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = InterpolatingAdjoint(autojacvec = SciMLSensitivity.ZygoteVJP())
)
_,
    easy_res10 = adjoint_sensitivities(
    soloop_nodense, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = InterpolatingAdjoint(autojacvec = SciMLSensitivity.ReverseDiffVJP())
)
_,
    easy_res11 = adjoint_sensitivities(
    soloop_nodense, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = InterpolatingAdjoint(autojacvec = SciMLSensitivity.ReverseDiffVJP(true))
)
#@test_broken _,easy_res12 = adjoint_sensitivities(soloop_nodense,Tsit5();t,dg_discrete=dg,
#                                     abstol=1e-14,reltol=1e-14,
#                                     sensealg=InterpolatingAdjoint(autojacvec=SciMLSensitivity.EnzymeVJP())
#                                     ) isa Tuple
#@test_broken _,easy_res13 = adjoint_sensitivities(soloop_nodense,Tsit5();t,dg_discrete=dg,
#                                     abstol=1e-14,reltol=1e-14,
#                                     sensealg=QuadratureAdjoint(autojacvec=SciMLSensitivity.EnzymeVJP())
#                                     ) isa Tuple
_,
    easy_res12 = adjoint_sensitivities(
    soloop, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = GaussAdjoint()
)
_,
    easy_res122 = adjoint_sensitivities(
    soloop, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = GaussAdjoint(autojacvec = ReverseDiffVJP(true))
)
_,
    easy_res123 = adjoint_sensitivities(
    soloop_nodense, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = GaussAdjoint(checkpointing = true),
    checkpoints = soloop_nodense.t[1:5:end]
)
_,
    easy_res12k = adjoint_sensitivities(
    soloop, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = GaussKronrodAdjoint()
)
_,
    easy_res122k = adjoint_sensitivities(
    soloop, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = GaussKronrodAdjoint(autojacvec = ReverseDiffVJP(true))
)
_,
    easy_res123k = adjoint_sensitivities(
    soloop_nodense, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = GaussKronrodAdjoint(checkpointing = true),
    checkpoints = soloop_nodense.t[1:5:end]
)

_,
    easy_res2_mc_quad = adjoint_sensitivities(
    soloop, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = QuadratureAdjoint(
        abstol = 1.0e-14, reltol = 1.0e-14, autojacvec = SciMLSensitivity.MooncakeVJP()
    )
)
_,
    easy_res2_mc_interp = adjoint_sensitivities(
    soloop, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = InterpolatingAdjoint(autojacvec = SciMLSensitivity.MooncakeVJP())
)
_,
    easy_res2_mc_back = adjoint_sensitivities(
    soloop, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = BacksolveAdjoint(autojacvec = SciMLSensitivity.MooncakeVJP())
)
_,
    easy_res6_mc_quad = adjoint_sensitivities(
    soloop_nodense, Tsit5(); t,
    dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = QuadratureAdjoint(
        abstol = 1.0e-14, reltol = 1.0e-14, autojacvec = SciMLSensitivity.MooncakeVJP()
    )
)
_,
    easy_res6_mc_interp = adjoint_sensitivities(
    soloop_nodense, Tsit5(); t,
    dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = InterpolatingAdjoint(
        checkpointing = true,
        autojacvec = SciMLSensitivity.MooncakeVJP()
    ),
    checkpoints = soloop_nodense.t[1:5:end]
)
_,
    easy_res6_mc_back = adjoint_sensitivities(
    soloop_nodense, Tsit5(); t,
    dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    sensealg = BacksolveAdjoint(autojacvec = SciMLSensitivity.MooncakeVJP())
)

@test isapprox(res, easy_res, rtol = 1.0e-10)
@test isapprox(res, easy_res2, rtol = 1.0e-10)
@test isapprox(res, easy_res22, rtol = 1.0e-10)
@test isapprox(res, easy_res23, rtol = 1.0e-10)
@test isapprox(res, easy_res3, rtol = 1.0e-10)
@test isapprox(res, easy_res32, rtol = 1.0e-10)
@test isapprox(res, easy_res4, rtol = 1.0e-10)
@test isapprox(res, easy_res42, rtol = 1.0e-10)
@test isapprox(res, easy_res5, rtol = 1.0e-9)
@test isapprox(res, easy_res6, rtol = 1.0e-10)
@test isapprox(res, easy_res62, rtol = 1.0e-9)
@test isapprox(res, easy_res8, rtol = 1.0e-9)
@test isapprox(res, easy_res9, rtol = 1.0e-9)
@test isapprox(res, easy_res10, rtol = 1.0e-9)
@test isapprox(res, easy_res11, rtol = 1.0e-9)
#@test isapprox(res, easy_res12, rtol = 1e-9)
#@test isapprox(res, easy_res13, rtol = 1e-9)
@test isapprox(res, easy_res12, rtol = 1.0e-9)
@test isapprox(res, easy_res122, rtol = 1.0e-9)
@test isapprox(res, easy_res123, rtol = 1.0e-4)
@test isapprox(res, easy_res12k, rtol = 1.0e-9)
@test isapprox(res, easy_res122k, rtol = 1.0e-9)
@test isapprox(res, easy_res123k, rtol = 1.0e-4)
@test isapprox(res, easy_res2_mc_quad, rtol = 1.0e-9)
@test isapprox(res, easy_res2_mc_interp, rtol = 1.0e-9)
@test isapprox(res, easy_res2_mc_back, rtol = 1.0e-9)
@test isapprox(res, easy_res6_mc_quad, rtol = 1.0e-4)
@test isapprox(res, easy_res6_mc_interp, rtol = 1.0e-9)
@test isapprox(res, easy_res6_mc_back, rtol = 1.0e-9)

println("Calculate adjoint sensitivities ")

_,
    easy_res8 = adjoint_sensitivities(
    solb, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    save_everystep = false, save_start = false,
    sensealg = BacksolveAdjoint()
)
_,
    easy_res82 = adjoint_sensitivities(
    solb, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    save_everystep = false, save_start = false,
    sensealg = BacksolveAdjoint(checkpointing = false)
)

@test isapprox(res, easy_res8, rtol = 1.0e-9)
@test isapprox(res, easy_res82, rtol = 1.0e-9)

_,
    end_only_res = adjoint_sensitivities(
    sol_end, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14,
    save_everystep = false, save_start = false,
    sensealg = BacksolveAdjoint()
)

@test isapprox(res, end_only_res, rtol = 1.0e-9)

println("Calculate adjoint sensitivities from autodiff & numerical diff")
function G(p)
    tmp_prob = remake(prob; u0 = convert.(eltype(p), prob.u0), p)
    sol = solve(
        tmp_prob, Tsit5(), abstol = 1.0e-14, reltol = 1.0e-14,
        sensealg = DiffEqBase.SensitivityADPassThrough(), saveat = t
    )
    A = Array(sol)
    return sum(((2 .- A) .^ 2) ./ 2)
end
G([1.5, 1.0, 3.0, 1.0])
res2 = ForwardDiff.gradient(G, [1.5, 1.0, 3.0, 1.0])
res3 = Calculus.gradient(G, [1.5, 1.0, 3.0, 1.0])

@test norm(res' .- res2) < 1.0e-7
@test norm(res' .- res3) < 1.0e-5

# check other t handling

t2 = [0.5, 1.0]
t3 = [0.0, 0.5, 1.0]
t4 = [0.5, 1.0, 10.0]

_,
    easy_res2 = adjoint_sensitivities(
    sol, Tsit5(), t = t2, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14
)
_,
    easy_res3 = adjoint_sensitivities(
    sol, Tsit5(), t = t3, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14
)
_,
    easy_res4 = adjoint_sensitivities(
    sol, Tsit5(), t = t4, dgdu_discrete = dg,
    abstol = 1.0e-14,
    reltol = 1.0e-14
)

function G(p, ts)
    tmp_prob = remake(prob; u0 = convert.(eltype(p), prob.u0), p)
    sol = solve(
        tmp_prob, Tsit5(), abstol = 1.0e-10, reltol = 1.0e-10,
        sensealg = DiffEqBase.SensitivityADPassThrough(), saveat = ts
    )
    A = convert(Array, sol)
    return sum(((2 .- A) .^ 2) ./ 2)
end
res2 = ForwardDiff.gradient(p -> G(p, t2), [1.5, 1.0, 3.0, 1.0])
res3 = ForwardDiff.gradient(p -> G(p, t3), [1.5, 1.0, 3.0, 1.0])
res4 = ForwardDiff.gradient(p -> G(p, t4), [1.5, 1.0, 3.0, 1.0])

@test easy_res2' ≈ res2 rtol = 1.0e-10
@test easy_res3' ≈ res3 rtol = 1.0e-10
@test easy_res4' ≈ res4 rtol = 1.0e-10

println("Adjoints of u0")

function dg(out, u, p, t, i)
    return out .= -1 .+ u
end

ū0, adj = adjoint_sensitivities(
    sol, Tsit5(); t, dgdu_discrete = dg, abstol = 1.0e-14, reltol = 1.0e-14
)

_, adjnou0 = adjoint_sensitivities(
    sol, Tsit5(); t, dgdu_discrete = dg, abstol = 1.0e-14, reltol = 1.0e-14
)

ū02, adj2 = adjoint_sensitivities(
    sol, Tsit5(); t, dgdu_discrete = dg, abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = BacksolveAdjoint()
)

ū022, adj22 = adjoint_sensitivities(
    sol, Tsit5(); t, dgdu_discrete = dg, abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = BacksolveAdjoint(autojacvec = false)
)

ū023, adj23 = adjoint_sensitivities(
    sol, Tsit5(); t, dgdu_discrete = dg, abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = BacksolveAdjoint(autojacvec = false, checkpointing = false)
)

ū03, adj3 = adjoint_sensitivities(
    sol, Tsit5(); t, dgdu_discrete = dg, abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = InterpolatingAdjoint()
)

ū032, adj32 = adjoint_sensitivities(
    sol, Tsit5(); t, dgdu_discrete = dg, abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = InterpolatingAdjoint(autojacvec = false)
)

ū04, adj4 = adjoint_sensitivities(
    sol, Tsit5(); t, dgdu_discrete = dg, abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = InterpolatingAdjoint(checkpointing = true),
    checkpoints = sol.t[1:500:end]
)

@test_nowarn adjoint_sensitivities(
    sol, Tsit5(); t, dgdu_discrete = dg, abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = InterpolatingAdjoint(checkpointing = true),
    checkpoints = sol.t[1:5:end]
)

ū042, adj42 = adjoint_sensitivities(
    sol, Tsit5(); t, dgdu_discrete = dg, abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = InterpolatingAdjoint(checkpointing = true, autojacvec = false),
    checkpoints = sol.t[1:500:end]
)

ū05, adj5 = adjoint_sensitivities(
    sol, Tsit5(); t, dgdu_discrete = dg, abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = QuadratureAdjoint(abstol = 1.0e-14, reltol = 1.0e-14)
)

ū052, adj52 = adjoint_sensitivities(
    sol, Tsit5(); t, dgdu_discrete = dg, abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = QuadratureAdjoint(
        autojacvec = false, abstol = 1.0e-14, reltol = 1.0e-14
    )
)

ū05, adj53 = adjoint_sensitivities(
    sol, Tsit5(); t, dgdu_discrete = dg, abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = QuadratureAdjoint(
        abstol = 1.0e-14,
        reltol = 1.0e-14,
        autojacvec = ReverseDiffVJP(true)
    )
)

ū06, adj6 = adjoint_sensitivities(
    sol, Tsit5(); t, dgdu_discrete = dg, abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = GaussAdjoint()
)
ū062, adj62 = adjoint_sensitivities(
    sol, Tsit5(); t, dgdu_discrete = dg, abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = GaussAdjoint(autojacvec = false)
)
ū06, adj63 = adjoint_sensitivities(
    sol, Tsit5(); t, dgdu_discrete = dg, abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = GaussAdjoint(autojacvec = ReverseDiffVJP(true))
)

ū06k, adj6k = adjoint_sensitivities(
    sol, Tsit5(); t, dgdu_discrete = dg, abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = GaussKronrodAdjoint()
)
ū062k, adj62k = adjoint_sensitivities(
    sol, Tsit5(); t, dgdu_discrete = dg, abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = GaussKronrodAdjoint(autojacvec = false)
)
ū06k, adj63k = adjoint_sensitivities(
    sol, Tsit5(); t, dgdu_discrete = dg, abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = GaussKronrodAdjoint(autojacvec = ReverseDiffVJP(true))
)

ū0args, adjargs = adjoint_sensitivities(
    sol, Tsit5(); t, dgdu_discrete = dg, abstol = 1.0e-14, reltol = 1.0e-14,
    save_everystep = false, save_start = false,
    sensealg = BacksolveAdjoint()
)

ū0args2, adjargs2 = adjoint_sensitivities(
    sol, Tsit5(); t, dgdu_discrete = dg, abstol = 1.0e-14, reltol = 1.0e-14,
    save_everystep = false, save_start = false,
    sensealg = InterpolatingAdjoint()
)

res = ForwardDiff.gradient(prob.u0) do u0
    tmp_prob = remake(prob; u0)
    sol = solve(tmp_prob, Tsit5(), abstol = 1.0e-14, reltol = 1.0e-14, saveat = t)
    A = convert(Array, sol)
    sum(((1 .- A) .^ 2) ./ 2)
end

@test ū0 ≈ res rtol = 1.0e-10
@test ū02 ≈ res rtol = 1.0e-10
@test ū022 ≈ res rtol = 1.0e-10
@test ū023 ≈ res rtol = 1.0e-10
@test ū03 ≈ res rtol = 1.0e-10
@test ū032 ≈ res rtol = 1.0e-10
@test ū04 ≈ res rtol = 1.0e-10
@test ū042 ≈ res rtol = 1.0e-10
@test ū05 ≈ res rtol = 1.0e-10
@test ū052 ≈ res rtol = 1.0e-10
@test ū06 ≈ res rtol = 1.0e-10
@test ū062 ≈ res rtol = 1.0e-10
@test ū06k ≈ res rtol = 1.0e-10
@test ū062k ≈ res rtol = 1.0e-10
@test adj ≈ adjnou0 rtol = 1.0e-10
@test adj ≈ adj2 rtol = 1.0e-10
@test adj ≈ adj22 rtol = 1.0e-10
@test adj ≈ adj23 rtol = 1.0e-10
@test adj ≈ adj3 rtol = 1.0e-10
@test adj ≈ adj32 rtol = 1.0e-10
@test adj ≈ adj4 rtol = 1.0e-10
@test adj ≈ adj42 rtol = 1.0e-10
@test adj ≈ adj5 rtol = 1.0e-10
@test adj ≈ adj52 rtol = 1.0e-10
@test adj ≈ adj53 rtol = 1.0e-10
@test adj ≈ adj6 rtol = 1.0e-10
@test adj ≈ adj62 rtol = 1.0e-10
@test adj ≈ adj63 rtol = 1.0e-10

@test adj ≈ adj6k rtol = 1.0e-10
@test adj ≈ adj62k rtol = 1.0e-10
@test adj ≈ adj63k rtol = 1.0e-10

@test ū0args ≈ res rtol = 1.0e-10
@test adjargs ≈ adj rtol = 1.0e-10
@test ū0args2 ≈ res rtol = 1.0e-10
@test adjargs2 ≈ adj rtol = 1.0e-10

println("Do a continuous adjoint problem")

# Energy calculation
g(u, p, t) = (sum(u) .^ 2) ./ 2
# Gradient of (u1 + u2)^2 / 2
function dg(out, u, p, t)
    out[1] = u[1] + u[2]
    out[2] = u[1] + u[2]
    return nothing
end

adj_prob = ODEAdjointProblem(
    sol,
    QuadratureAdjoint(
        abstol = 1.0e-14, reltol = 1.0e-14,
        autojacvec = SciMLSensitivity.ReverseDiffVJP()
    ),
    Tsit5(),
    nothing, nothing, nothing, dg, nothing, g
)
adj_sol = solve(adj_prob, Tsit5(), abstol = 1.0e-14, reltol = 1.0e-10)
integrand = AdjointSensitivityIntegrand(
    sol, adj_sol,
    QuadratureAdjoint(
        abstol = 1.0e-14, reltol = 1.0e-14,
        autojacvec = SciMLSensitivity.ReverseDiffVJP()
    )
)
res, err = quadgk(integrand, 0.0, 10.0, atol = 1.0e-14, rtol = 1.0e-10)

println("Test the `adjoint_sensitivities` utility function")
_, easy_res = adjoint_sensitivities(
    sol, Tsit5(); dgdu_continuous = dg, g,
    abstol = 1.0e-14, reltol = 1.0e-14
)
println("2")
_, easy_res2 = adjoint_sensitivities(
    sol, Tsit5(); dgdu_continuous = dg, g,
    abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = InterpolatingAdjoint()
)
_, easy_res22 = adjoint_sensitivities(
    sol, Tsit5(); dgdu_continuous = dg, g,
    abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = InterpolatingAdjoint(autojacvec = false)
)
println("23")
_, easy_res23 = adjoint_sensitivities(
    sol, Tsit5(); dgdu_continuous = dg, g,
    abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = QuadratureAdjoint(abstol = 1.0e-14, reltol = 1.0e-14)
)
_, easy_res232 = adjoint_sensitivities(
    sol, Tsit5(); dgdu_continuous = dg, g,
    abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = QuadratureAdjoint(
        abstol = 1.0e-14,
        reltol = 1.0e-14,
        autojacvec = ReverseDiffVJP(false)
    )
)
_, easy_res24 = adjoint_sensitivities(
    sol, Tsit5(); dgdu_continuous = dg, g,
    abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = QuadratureAdjoint(
        autojacvec = false, abstol = 1.0e-14, reltol = 1.0e-14
    )
)
println("25")
_, easy_res25 = adjoint_sensitivities(
    sol, Tsit5(); dgdu_continuous = dg, g,
    abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = BacksolveAdjoint()
)
_, easy_res26 = adjoint_sensitivities(
    sol, Tsit5(); dgdu_continuous = dg, g,
    abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = BacksolveAdjoint(autojacvec = false)
)
_, easy_res262 = adjoint_sensitivities(
    sol, Tsit5(); dgdu_continuous = dg, g,
    abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = BacksolveAdjoint(autojacvec = false, checkpointing = false)
)
println("27")
_, easy_res27 = adjoint_sensitivities(
    sol, Tsit5(); dgdu_continuous = dg, g,
    abstol = 1.0e-14, reltol = 1.0e-14,
    checkpoints = sol.t[1:500:end],
    sensealg = InterpolatingAdjoint(checkpointing = true)
)
_, easy_res28 = adjoint_sensitivities(
    sol, Tsit5(); dgdu_continuous = dg, g,
    abstol = 1.0e-14, reltol = 1.0e-14,
    checkpoints = sol.t[1:500:end],
    sensealg = InterpolatingAdjoint(checkpointing = true, autojacvec = false)
)
println("3")
_, easy_res3 = adjoint_sensitivities(
    sol, Tsit5(); g, abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = InterpolatingAdjoint()
)
_, easy_res32 = adjoint_sensitivities(
    sol, Tsit5(); g, abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = InterpolatingAdjoint(autojacvec = false)
)
println("33")
_, easy_res33 = adjoint_sensitivities(
    sol, Tsit5(); g, abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = QuadratureAdjoint(abstol = 1.0e-14, reltol = 1.0e-14)
)
_, easy_res34 = adjoint_sensitivities(
    sol, Tsit5(); g, abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = QuadratureAdjoint(
        autojacvec = false,
        abstol = 1.0e-14,
        reltol = 1.0e-14
    )
)
println("35")
_, easy_res35 = adjoint_sensitivities(
    sol, Tsit5(); g, abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = BacksolveAdjoint()
)
_, easy_res36 = adjoint_sensitivities(
    sol, Tsit5(); g, abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = BacksolveAdjoint(autojacvec = false)
)
println("37")
_, easy_res37 = adjoint_sensitivities(
    sol, Tsit5(); g, abstol = 1.0e-14, reltol = 1.0e-14,
    checkpoints = sol.t[1:500:end],
    sensealg = InterpolatingAdjoint(checkpointing = true)
)
_, easy_res38 = adjoint_sensitivities(
    sol, Tsit5(); g, abstol = 1.0e-14, reltol = 1.0e-14,
    checkpoints = sol.t[1:500:end],
    sensealg = InterpolatingAdjoint(checkpointing = true, autojacvec = false)
)
println("40")
_, easy_res40 = adjoint_sensitivities(
    sol, Tsit5(); dgdu_continuous = dg, g,
    abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = GaussAdjoint()
)
_, easy_res41 = adjoint_sensitivities(
    sol, Tsit5(); dgdu_continuous = dg, g,
    abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = GaussAdjoint(autojacvec = ReverseDiffVJP(false))
)
_, easy_res42 = adjoint_sensitivities(
    sol, Tsit5(); dgdu_continuous = dg, g,
    abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = GaussAdjoint(autojacvec = false)
)
println("43")
_, easy_res43 = adjoint_sensitivities(
    sol, Tsit5(); g, abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = GaussAdjoint()
)
_, easy_res44 = adjoint_sensitivities(
    sol, Tsit5(); g, abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = GaussAdjoint(autojacvec = false)
)

_, easy_res40k = adjoint_sensitivities(
    sol, Tsit5(); dgdu_continuous = dg, g,
    abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = GaussKronrodAdjoint()
)
_, easy_res41k = adjoint_sensitivities(
    sol, Tsit5(); dgdu_continuous = dg, g,
    abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = GaussKronrodAdjoint(autojacvec = ReverseDiffVJP(false))
)
_, easy_res42k = adjoint_sensitivities(
    sol, Tsit5(); dgdu_continuous = dg, g,
    abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = GaussKronrodAdjoint(autojacvec = false)
)
_, easy_res43k = adjoint_sensitivities(
    sol, Tsit5(); g, abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = GaussKronrodAdjoint()
)
_, easy_res44k = adjoint_sensitivities(
    sol, Tsit5(); g, abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = GaussKronrodAdjoint(autojacvec = false)
)

@test norm(easy_res .- res) < 1.0e-8
@test norm(easy_res2 .- res) < 1.0e-8
@test norm(easy_res22 .- res) < 1.0e-8
@test norm(easy_res23 .- res) < 1.0e-8
@test norm(easy_res232 .- res) < 1.0e-8
@test norm(easy_res24 .- res) < 1.0e-8
@test norm(easy_res25 .- res) < 1.0e-8
@test norm(easy_res26 .- res) < 1.0e-8
@test norm(easy_res262 .- res) < 1.0e-8
@test norm(easy_res27 .- res) < 1.0e-8
@test norm(easy_res28 .- res) < 1.0e-8
@test norm(easy_res3 .- res) < 1.0e-8
@test norm(easy_res32 .- res) < 1.0e-8
@test norm(easy_res33 .- res) < 1.0e-8
@test norm(easy_res34 .- res) < 1.0e-8
@test norm(easy_res35 .- res) < 1.0e-8
@test norm(easy_res36 .- res) < 1.0e-8
@test norm(easy_res37 .- res) < 1.0e-8
@test norm(easy_res38 .- res) < 1.0e-8
@test norm(easy_res40 .- res) < 1.0e-8
@test norm(easy_res41 .- res) < 1.0e-8
@test norm(easy_res42 .- res) < 1.0e-8
@test norm(easy_res43 .- res) < 1.0e-8
@test norm(easy_res44 .- res) < 1.0e-8
@test norm(easy_res40k .- res) < 1.0e-8
@test norm(easy_res41k .- res) < 1.0e-8
@test norm(easy_res42k .- res) < 1.0e-8
@test norm(easy_res43k .- res) < 1.0e-8
@test norm(easy_res44k .- res) < 1.0e-8

println("Calculate adjoint sensitivities from autodiff & numerical diff")
function G(p)
    tmp_prob = remake(
        prob; u0 = eltype(p).(prob.u0), p,
        tspan = eltype(p).(prob.tspan)
    )
    sol = solve(tmp_prob, Tsit5(), abstol = 1.0e-14, reltol = 1.0e-14)
    integrand = (t) -> (sum(sol(t)) .^ 2) ./ 2
    res, err = quadgk(integrand, 0.0, 10.0, atol = 1.0e-14, rtol = 1.0e-10)
    return res
end
res2 = ForwardDiff.gradient(G, [1.5, 1.0, 3.0, 1.0])
res3 = Calculus.gradient(G, [1.5, 1.0, 3.0, 1.0])

@test norm(res' .- res2) < 1.0e-8
@test norm(res' .- res3) < 1.0e-6

# Buffer length test
f = (du, u, p, t) -> du .= 0
p = zeros(3);
u = zeros(50);
prob = ODEProblem(f, u, (0.0, 10.0), p)
sol = solve(prob, Tsit5(), abstol = 1.0e-14, reltol = 1.0e-14)
@test_nowarn _, res = adjoint_sensitivities(
    sol, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-14, reltol = 1.0e-14
)

@info "Checkpointed backsolve"
using SciMLSensitivity, OrdinaryDiffEq
tf = 10.0
function lorenz(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
    return nothing
end
prob_lorenz = ODEProblem(lorenz, [1.0, 0.0, 0.0], (0, tf), [10, 28, 8 / 3])
sol_lorenz = solve(prob_lorenz, Tsit5(), reltol = 1.0e-6, abstol = 1.0e-9)
function dg(out, u, p, t, i)
    return (out .= -2.0 .+ u)
end
t = 0:0.1:tf
_, easy_res1 = adjoint_sensitivities(
    sol_lorenz, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-6, reltol = 1.0e-9,
    sensealg = BacksolveAdjoint()
)
_, easy_res2 = adjoint_sensitivities(
    sol_lorenz, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-6, reltol = 1.0e-9,
    sensealg = InterpolatingAdjoint()
)
_, easy_res3 = adjoint_sensitivities(
    sol_lorenz, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-6, reltol = 1.0e-9,
    sensealg = BacksolveAdjoint(),
    checkpoints = sol_lorenz.t[1:10:end]
)
_, easy_res4 = adjoint_sensitivities(
    sol_lorenz, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-6, reltol = 1.0e-9,
    sensealg = BacksolveAdjoint(),
    checkpoints = sol_lorenz.t[1:20:end]
)
# cannot finish in a reasonable amount of time
@test_skip adjoint_sensitivities(
    sol_lorenz, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-6, reltol = 1.0e-9,
    sensealg = BacksolveAdjoint(checkpointing = false)
)
@test easy_res2 ≈ easy_res1 rtol = 1.0e-5
@test easy_res2 ≈ easy_res3 rtol = 1.0e-5
@test easy_res2 ≈ easy_res4 rtol = 1.0e-4

ū1, adj1 = adjoint_sensitivities(
    sol_lorenz, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-6, reltol = 1.0e-9,
    sensealg = BacksolveAdjoint()
)
ū2, adj2 = adjoint_sensitivities(
    sol_lorenz, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-6,
    reltol = 1.0e-9,
    sensealg = InterpolatingAdjoint()
)
ū3, adj3 = adjoint_sensitivities(
    sol_lorenz, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-6,
    reltol = 1.0e-9,
    sensealg = BacksolveAdjoint(),
    checkpoints = sol_lorenz.t[1:10:end]
)
ū4, adj4 = adjoint_sensitivities(
    sol_lorenz, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-6,
    reltol = 1.0e-9,
    sensealg = BacksolveAdjoint(),
    checkpoints = sol_lorenz.t[1:20:end]
)
# cannot finish in a reasonable amount of time
@test_skip adjoint_sensitivities(
    sol_lorenz, Tsit5(); t, dgdu_discrete = dg,
    abstol = 1.0e-6, reltol = 1.0e-9,
    sensealg = BacksolveAdjoint(checkpointing = false)
)
@test ū2 ≈ ū1 rtol = 1.0e-5
@test adj2 ≈ adj1 rtol = 1.0e-5
@test ū2 ≈ ū3 rtol = 1.0e-5
@test adj2 ≈ adj3 rtol = 1.0e-5
@test ū2 ≈ ū4 rtol = 1.0e-4
@test adj2 ≈ adj4 rtol = 1.0e-4

# LQR Tests from issue https://github.com/SciML/SciMLSensitivity.jl/issues/300
x_dim = 2
T = 40.0

cost = (x, u) -> x' * x
params = [-0.4142135623730951, 0.0, -0.0, -0.4142135623730951, 0.0, 0.0]

function dynamics!(du, u, p, t)
    du[1] = -u[1] + tanh(p[1] * u[1] + p[2] * u[2])
    du[2] = -u[2] + tanh(p[3] * u[1] + p[4] * u[2])
    return nothing
end

function backsolve_grad(sol, lqr_params, checkpointing)
    bwd_sol = solve(
        ODEAdjointProblem(
            sol,
            BacksolveAdjoint(; autojacvec = EnzymeVJP(), checkpointing),
            Tsit5(),
            nothing, nothing, nothing, nothing, nothing,
            (x, lqr_params, t) -> cost(x, lqr_params)
        ),
        Tsit5(),
        dense = false,
        save_everystep = false
    )

    return bwd_sol.u[end][1:(end - x_dim)]
    #fwd_sol, bwd_sol
end

x0 = ones(x_dim)
fwd_sol = solve(
    ODEProblem(dynamics!, x0, (0, T), params),
    Tsit5(), abstol = 1.0e-9, reltol = 1.0e-9,
    u0 = x0,
    p = params,
    dense = false,
    save_everystep = true
)

backsolve_results = backsolve_grad(fwd_sol, params, false)
backsolve_checkpointing_results = backsolve_grad(fwd_sol, params, true)

@test backsolve_results != backsolve_checkpointing_results

int_u0,
    int_p = adjoint_sensitivities(
    fwd_sol, Tsit5(),
    g = (x, params, t) -> cost(x, params),
    sensealg = InterpolatingAdjoint()
)

@test isapprox(backsolve_checkpointing_results[1:length(x0)], int_u0, rtol = 1.0e-10)
@test isapprox(
    backsolve_checkpointing_results[(1:length(params)) .+ length(x0)],
    int_p', rtol = 1.0e-10
)

@info "Mass matrix tests"
using Test
using LinearAlgebra, SciMLSensitivity, OrdinaryDiffEq, ForwardDiff, QuadGK
function G(p, prob, ts, cost)
    tmp_prob_mm = remake(prob; u0 = convert.(eltype(p), prob.u0), p)
    sol = solve(
        tmp_prob_mm, Rodas4(autodiff = AutoFiniteDiff()), abstol = 1.0e-14, reltol = 1.0e-14,
        saveat = ts
    )
    return cost(sol)
end
alg = Rodas4(autodiff = AutoFiniteDiff())
@info "discrete cost"
A = [1 2 3; 4 5 6; 7 8 9]
function foo(du, u, p, t)
    mul!(du, A, u)
    du .= du .+ p
    du[2] += sum(p)
    return nothing
end
mm = -[1 2 4; 2 3 7; 1 3 41]
u0 = [1, 2.0, 3]
p = [1.0, 2.0, 3]
prob_mm = ODEProblem(ODEFunction(foo, mass_matrix = mm), u0, (0, 1.0), p)
sol_mm = solve(prob_mm, Rodas4(), reltol = 1.0e-14, abstol = 1.0e-14)

ts = 0:0.01:1
dg(out, u, p, t, i) = out .= 1
_,
    res = adjoint_sensitivities(
    sol_mm, alg, t = ts, dgdu_discrete = dg,
    abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = QuadratureAdjoint()
)
reference_sol = ForwardDiff.gradient(p -> G(p, prob_mm, ts, sum), vec(p))
@test res' ≈ reference_sol rtol = 1.0e-11

_,
    res_gauss = adjoint_sensitivities(
    sol_mm, alg, t = ts, dgdu_discrete = dg,
    abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = GaussAdjoint()
)
@test res_gauss ≈ res rtol = 1.0e-11

_,
    res_gausskron = adjoint_sensitivities(
    sol_mm, alg, t = ts, dgdu_discrete = dg,
    abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = GaussKronrodAdjoint()
)
@test res_gausskron ≈ res rtol = 1.0e-11

_,
    res_interp = adjoint_sensitivities(
    sol_mm, alg, t = ts, dgdu_discrete = dg,
    abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = InterpolatingAdjoint()
)
@test res_interp ≈ res rtol = 1.0e-11
_,
    res_interp2 = adjoint_sensitivities(
    sol_mm, alg, t = ts, dgdu_discrete = dg,
    abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = InterpolatingAdjoint(checkpointing = true),
    checkpoints = sol_mm.t[1:10:end]
)
@test res_interp2 ≈ res rtol = 1.0e-11

_,
    res_bs = adjoint_sensitivities(
    sol_mm, alg, t = ts, dgdu_discrete = dg,
    abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = BacksolveAdjoint(checkpointing = false)
)
@test res_bs ≈ res rtol = 1.0e-11
_,
    res_bs2 = adjoint_sensitivities(
    sol_mm, alg, t = ts, dgdu_discrete = dg,
    abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = BacksolveAdjoint(checkpointing = true),
    checkpoints = sol_mm.t
)
@test res_bs2 ≈ res rtol = 1.0e-11

@info "continuous cost"
g_cont(u, p, t) = (sum(u) .^ 2) ./ 2
dg_cont(out, u, p, t) = out .= sum(u)
_,
    easy_res_cont = adjoint_sensitivities(
    sol_mm, alg, dgdu_continuous = dg_cont,
    g = g_cont,
    abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = QuadratureAdjoint()
)

_,
    easy_res_cont_gauss = adjoint_sensitivities(
    sol_mm, alg, dgdu_continuous = dg_cont,
    g = g_cont,
    abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = GaussAdjoint()
)
_,
    easy_res_cont_gauss_kron = adjoint_sensitivities(
    sol_mm, alg, dgdu_continuous = dg_cont,
    g = g_cont,
    abstol = 1.0e-14, reltol = 1.0e-14,
    sensealg = GaussKronrodAdjoint()
)

function G_cont(p)
    tmp_prob_mm = remake(
        prob_mm; u0 = eltype(p).(prob_mm.u0), p,
        tspan = eltype(p).(prob_mm.tspan)
    )
    sol = solve(
        tmp_prob_mm, Rodas4(autodiff = AutoFiniteDiff()),
        abstol = 1.0e-14, reltol = 1.0e-14
    )
    res, err = quadgk(
        (t) -> (sum(sol(t)) .^ 2) ./ 2, prob_mm.tspan...,
        atol = 1.0e-14, rtol = 1.0e-10
    )
    return res
end
reference_sol_cont = ForwardDiff.gradient(G_cont, p)
@test easy_res_cont' ≈ reference_sol_cont rtol = 1.0e-11
@test easy_res_cont_gauss' ≈ reference_sol_cont rtol = 1.0e-11
@test easy_res_cont_gauss_kron' ≈ reference_sol_cont rtol = 1.0e-11

@info "Singular mass matrix"
function rober(du, u, p, t)
    y₁, y₂, y₃ = u
    k₁, k₂, k₃ = p
    du[1] = -k₁ * y₁ + k₃ * y₂ * y₃
    du[2] = k₁ * y₁ - k₂ * y₂^2 - k₃ * y₂ * y₃
    du[3] = y₁ + y₂ + y₃ - 1
    return nothing
end
function rober(u, p, t)
    y₁, y₂, y₃ = u
    k₁, k₂, k₃ = p
    return [
        -k₁ * y₁ + k₃ * y₂ * y₃,
        k₁ * y₁ - k₂ * y₂^2 - k₃ * y₂ * y₃,
        y₁ + y₂ + y₃ - 1,
    ]
end
M = [
    1.0 0 0
    0 1.0 0
    0 0 0
]
for iip in [true, false]
    f = ODEFunction{iip}(rober, mass_matrix = M)
    p = [0.04, 3.0e7, 1.0e4]

    prob_singular_mm = ODEProblem(f, [1.0, 0.0, 1.0], (0.0, 100), p)
    sol_singular_mm = solve(
        prob_singular_mm, FBDF(autodiff = AutoFiniteDiff()),
        reltol = 1.0e-12, abstol = 1.0e-12, initializealg = BrownFullBasicInit()
    )
    ts = [50, sol_singular_mm.t[end]]
    dg_singular(out, u, p, t, i) = (fill!(out, 0); out[end] = 1)
    _,
        res = adjoint_sensitivities(
        sol_singular_mm, alg, t = ts,
        dgdu_discrete = dg_singular, abstol = 1.0e-8,
        reltol = 1.0e-8, sensealg = QuadratureAdjoint(),
        maxiters = Int(1.0e6)
    )
    reference_sol = ForwardDiff.gradient(
        p -> G(
            p, prob_singular_mm, ts,
            sol -> sum(last, sol.u)
        ), vec(p)
    )
    @test res' ≈ reference_sol rtol = 1.0e-5

    _,
        res_gauss = adjoint_sensitivities(
        sol_singular_mm, alg, t = ts,
        dgdu_discrete = dg_singular, abstol = 1.0e-8,
        reltol = 1.0e-8, sensealg = GaussAdjoint(),
        maxiters = Int(1.0e6)
    )
    @test res_gauss ≈ res rtol = 1.0e-5

    _,
        res_gausskron = adjoint_sensitivities(
        sol_singular_mm, alg, t = ts,
        dgdu_discrete = dg_singular, abstol = 1.0e-8,
        reltol = 1.0e-8, sensealg = GaussKronrodAdjoint(),
        maxiters = Int(1.0e6)
    )
    @test res_gausskron ≈ res rtol = 1.0e-5

    _,
        res_interp = adjoint_sensitivities(
        sol_singular_mm, alg, t = ts,
        dgdu_discrete = dg_singular,
        abstol = 1.0e-8,
        reltol = 1.0e-8,
        sensealg = InterpolatingAdjoint(),
        maxiters = Int(1.0e6)
    )
    @test res_interp ≈ res rtol = 1.0e-5
    _,
        res_interp2 = adjoint_sensitivities(
        sol_singular_mm, alg, t = ts,
        dgdu_discrete = dg_singular,
        abstol = 1.0e-8,
        reltol = 1.0e-8,
        sensealg = InterpolatingAdjoint(checkpointing = true),
        checkpoints = sol_singular_mm.t[1:10:end]
    )
    @test res_interp2 ≈ res rtol = 1.0e-5

    # backsolve doesn't work
    _,
        res_bs = adjoint_sensitivities(
        sol_singular_mm, alg, t = ts,
        dgdu_discrete = dg_singular, abstol = 1.0e-8,
        reltol = 1.0e-8,
        sensealg = BacksolveAdjoint(checkpointing = false)
    )
    @test_broken res_bs ≈ res rtol = 1.0e-5
    _,
        res_bs2 = adjoint_sensitivities(
        sol_singular_mm, alg, t = ts,
        dgdu_discrete = dg_singular, abstol = 1.0e-8,
        reltol = 1.0e-8,
        sensealg = BacksolveAdjoint(checkpointing = true),
        checkpoints = sol_singular_mm.t
    )
    @test_broken res_bs2 ≈ res rtol = 1.0e-5
end

# u' = x = p * u
function simple_linear_dae(du, u, p, t)
    du[1] = u[2]
    du[2] = u[2] - p[1] * u[1]
    return nothing
end
p = [0.5]
prob_singular_mm = ODEProblem(
    ODEFunction(
        simple_linear_dae,
        mass_matrix = Diagonal([1, 0])
    ),
    [2.2, 1.1], (0.0, 1.5), p
)
sol_singular_mm = solve(
    prob_singular_mm, Rodas4(autodiff = AutoFiniteDiff()),
    reltol = 1.0e-14, abstol = 1.0e-14
)
ts = [0.01, 0.25, 0.5, 1.0, 1.5]
dg_singular(out, u, p, t, i) = fill!(out, 1)
reference_sol = ForwardDiff.gradient(
    p -> G(
        p, prob_singular_mm, ts,
        sol -> sum(sum, sol.u)
    ), vec(p)
)
for salg in [
        QuadratureAdjoint(),
        InterpolatingAdjoint(),
        BacksolveAdjoint(),
        GaussAdjoint(),
        GaussKronrodAdjoint(),
    ]
    _,
        res = adjoint_sensitivities(
        sol_singular_mm, alg, t = ts,
        dgdu_discrete = dg_singular, abstol = 1.0e-14,
        reltol = 1.0e-14, sensealg = salg,
        maxiters = Int(1.0e6)
    )
    @test res' ≈ reference_sol rtol = 1.0e-5
end

# u' = x = p * u^2
function simple_nonlinear_dae(du, u, p, t)
    du[1] = u[2]
    du[2] = u[2] - p[1] * u[1]^2
    return nothing
end
p = [0.5]
prob_singular_mm = ODEProblem(
    ODEFunction(
        simple_nonlinear_dae,
        mass_matrix = Diagonal([1, 0])
    ),
    [1.0, 1.0], (0.0, 1), p
)
sol_singular_mm = solve(
    prob_singular_mm, Rodas4(autodiff = AutoFiniteDiff()),
    reltol = 1.0e-12, abstol = 1.0e-12
)
ts = [0.5, 1.0]
_,
    res = adjoint_sensitivities(
    sol_singular_mm, alg, t = ts,
    dgdu_discrete = dg_singular, abstol = 1.0e-8,
    reltol = 1.0e-8, sensealg = QuadratureAdjoint(),
    maxiters = Int(1.0e6)
)
reference_sol = ForwardDiff.gradient(
    p -> G(
        p, prob_singular_mm, ts,
        sol -> sum(sum, sol.u)
    ), vec(p)
)
for salg in [
        QuadratureAdjoint(),
        InterpolatingAdjoint(),
        BacksolveAdjoint(),
        GaussAdjoint(),
        GaussKronrodAdjoint(),
    ]
    _,
        res = adjoint_sensitivities(
        sol_singular_mm, alg, t = ts,
        dgdu_discrete = dg_singular, abstol = 1.0e-14,
        reltol = 1.0e-14, sensealg = salg,
        maxiters = Int(1.0e6)
    )
    @test res' ≈ reference_sol rtol = 1.0e-7
end

function pend(du, u, p, t)
    x, dx, y, dy, T = u
    g, = p
    du[1] = dx
    du[2] = T * x
    du[3] = dy
    du[4] = T * y - g
    du[5] = 2 * (dx^2 + dy^2 + y * (y * T - g) + T * x^2)
    return nothing
end

x0 = [1.0, 0, 0, 0, 0]
tspan = (0.0, 1.0)
p = [9.8]
f_singular_mm = ODEFunction{true}(pend, mass_matrix = Diagonal([1, 1, 1, 1, 0]))
prob_singular_mm = ODEProblem{true}(f_singular_mm, x0, tspan, p)
sol_singular_mm = solve(
    prob_singular_mm, Rodas5P(),
    reltol = 1.0e-12, abstol = 1.0e-12
)
ts = 0:0.1:1.0
dg_singular(out, u, p, t, i) = (fill!(out, 0); out[end] = 1)
_,
    res = adjoint_sensitivities(
    sol_singular_mm, alg, t = ts,
    dgdu_discrete = dg_singular, abstol = 1.0e-8,
    reltol = 1.0e-8, sensealg = QuadratureAdjoint(),
    maxiters = Int(1.0e6)
)
reference_sol = ForwardDiff.gradient(
    p -> G(
        p, prob_singular_mm, ts,
        sol -> sum(last, sol.u)
    ), vec(p)
)
@test res' ≈ reference_sol rtol = 1.0e-6

for salg in [
        QuadratureAdjoint(),
        InterpolatingAdjoint(),
        BacksolveAdjoint(),
        GaussAdjoint(),
        GaussKronrodAdjoint(),
    ]
    sol_singular_mm = solve(
        prob_singular_mm, Rodas5P(),
        reltol = 1.0e-12, abstol = 1.0e-12
    )
    _,
        res = adjoint_sensitivities(
        sol_singular_mm, alg, t = ts,
        dgdu_discrete = dg_singular, abstol = 1.0e-8,
        reltol = 1.0e-8, sensealg = salg,
        maxiters = Int(1.0e6)
    )
    @show salg
    @test res' ≈ reference_sol rtol = 1.0e-6
end
