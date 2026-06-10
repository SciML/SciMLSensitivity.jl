using SciMLSensitivity, OrdinaryDiffEq, StaticArrays, QuadGK, ForwardDiff, Zygote
using Test

##StaticArrays rrule
u0 = @SVector rand(2)
p = @SVector rand(4)

function lotka(u, p, svec = true)
    du1 = p[1] * u[1] - p[2] * u[1] * u[2]
    du2 = -p[3] * u[2] + p[4] * u[1] * u[2]
    return if svec
        @SVector [du1, du2]
    else
        @SMatrix [du1 du2 du1; du2 du1 du1]
    end
end

#SVector constructor adjoint
function loss(p)
    u = lotka(u0, p)
    return sum(1 .- u)
end

grad = Zygote.gradient(loss, p)
@test grad[1] isa SArray
grad2 = ForwardDiff.gradient(loss, p)
@test grad[1] ≈ grad2 rtol = 1.0e-12

#SMatrix constructor adjoint
function loss_mat(p)
    u = lotka(u0, p, false)
    return sum(1 .- u)
end

grad = Zygote.gradient(loss_mat, p)
@test grad[1] isa SArray
grad2 = ForwardDiff.gradient(loss_mat, p)
@test grad[1] ≈ grad2 rtol = 1.0e-12

##Adjoints of StaticArrays ODE

u0 = @SVector [1.0, 1.0]
p = @SVector [1.5, 1.0, 3.0, 1.0]
tspan = (0.0, 5.0)
datasize = 15
tsteps = range(tspan[1], tspan[2], length = datasize)

function lotka(u, p, t)
    du1 = p[1] * u[1] - p[2] * u[1] * u[2]
    du2 = -p[3] * u[2] + p[4] * u[1] * u[2]
    return @SVector [du1, du2]
end

prob = ODEProblem(lotka, u0, tspan, p)
sol = solve(prob, Tsit5(), saveat = tsteps, abstol = 1.0e-14, reltol = 1.0e-14)

## Discrete Case
dg_disc(u, p, t, i; outtype = nothing) = u

du0,
    dp = adjoint_sensitivities(
    sol, Tsit5(); t = tsteps, dgdu_discrete = dg_disc,
    sensealg = QuadratureAdjoint(abstol = 1.0e-14, reltol = 1.0e-14, autojacvec = ZygoteVJP())
)

@test !iszero(du0)
@test !iszero(dp)
#
adj_prob = ODEAdjointProblem(
    sol,
    QuadratureAdjoint(abstol = 1.0e-14, reltol = 1.0e-14, autojacvec = ZygoteVJP()),
    Tsit5(), tsteps, dg_disc
)
adj_sol = solve(adj_prob, Tsit5(), abstol = 1.0e-14, reltol = 1.0e-14)
integrand = AdjointSensitivityIntegrand(
    sol, adj_sol,
    QuadratureAdjoint(abstol = 1.0e-14, reltol = 1.0e-14, autojacvec = ZygoteVJP())
)
res, err = quadgk(integrand, 0.0, 5.0, atol = 1.0e-14, rtol = 1.0e-14)

@test adj_sol.u[end] ≈ du0 rtol = 1.0e-12
@test res ≈ dp rtol = 1.0e-12

###Comparing with gradients of lotka volterra with normal arrays
u2 = [1.0, 1.0]
p2 = [1.5, 1.0, 3.0, 1.0]

function f(u, p, t)
    du1 = p[1] * u[1] - p[2] * u[1] * u[2]
    du2 = -p[3] * u[2] + p[4] * u[1] * u[2]
    return [du1, du2]
end

prob2 = ODEProblem(f, u2, tspan, p2)
sol2 = solve(prob, Tsit5(), saveat = tsteps, abstol = 1.0e-14, reltol = 1.0e-14)

function dg_disc(du, u, p, t, i)
    return du .= u
end

du1,
    dp1 = adjoint_sensitivities(
    sol, Tsit5(); t = tsteps, dgdu_discrete = dg_disc,
    sensealg = QuadratureAdjoint(abstol = 1.0e-14, reltol = 1.0e-14, autojacvec = ZygoteVJP())
)

@test du0 ≈ du1 rtol = 1.0e-12
@test dp ≈ dp1 rtol = 1.0e-12

## with ForwardDiff and Zygote

function G_p(p)
    tmp_prob = remake(prob; u0 = convert.(eltype(p), prob.u0), p)
    sol = solve(
        tmp_prob, Tsit5(), abstol = 1.0e-14, reltol = 1.0e-14,
        sensealg = QuadratureAdjoint(
            abstol = 1.0e-14, reltol = 1.0e-14,
            autojacvec = ZygoteVJP()
        ), saveat = tsteps
    )
    u = Array(sol)
    return sum(((1 .- u) .^ 2) ./ 2)
end

function G_u(u0)
    tmp_prob = remake(prob; u0, prob.p)
    sol = solve(
        tmp_prob, Tsit5(), saveat = tsteps,
        abstol = 1.0e-14, reltol = 1.0e-14,
        sensealg = QuadratureAdjoint(
            abstol = 1.0e-14, reltol = 1.0e-14,
            autojacvec = ZygoteVJP()
        )
    )
    u = Array(sol)

    return sum(((1 .- u) .^ 2) ./ 2)
end

G_p(p)
G_u(u0)
f_dp = ForwardDiff.gradient(G_p, p)
f_du0 = ForwardDiff.gradient(G_u, u0)

z_dp = Zygote.gradient(G_p, p)
z_du0 = Zygote.gradient(G_u, u0)

@test z_du0[1] ≈ f_du0 rtol = 1.0e-12
@test z_dp[1] ≈ f_dp rtol = 1.0e-12

## Continuous Case

g(u, p, t) = sum((u .^ 2) ./ 2)

function dg(u, p, t)
    return u
end

du0, dp = adjoint_sensitivities(
    sol, Tsit5(); dgdu_continuous = dg, g,
    sensealg = QuadratureAdjoint(abstol = 1.0e-14, reltol = 1.0e-14, autojacvec = ZygoteVJP())
)

@test !iszero(du0)
@test !iszero(dp)

adj_prob = ODEAdjointProblem(
    sol,
    QuadratureAdjoint(abstol = 1.0e-14, reltol = 1.0e-14, autojacvec = ZygoteVJP()),
    Tsit5(), nothing, nothing, nothing, dg, nothing, g
)
adj_sol = solve(adj_prob, Tsit5(), abstol = 1.0e-14, reltol = 1.0e-14)
integrand = AdjointSensitivityIntegrand(
    sol, adj_sol,
    QuadratureAdjoint(abstol = 1.0e-14, reltol = 1.0e-14, autojacvec = ZygoteVJP())
)
res, err = quadgk(integrand, 0.0, 5.0, atol = 1.0e-14, rtol = 1.0e-14)

@test adj_sol.u[end] ≈ du0 rtol = 1.0e-12
@test res ≈ dp rtol = 1.0e-12

##ForwardDiff

function G_p(p)
    tmp_prob = remake(prob; p)
    sol = solve(tmp_prob, Tsit5(), abstol = 1.0e-12, reltol = 1.0e-12)
    res, err = quadgk(
        (t) -> (sum((sol(t) .^ 2) ./ 2)), 0.0, 5.0,
        atol = 1.0e-12, rtol = 1.0e-12
    )
    return res
end

function G_u(u0)
    tmp_prob = remake(prob; u0)
    sol = solve(tmp_prob, Tsit5(), abstol = 1.0e-12, reltol = 1.0e-12)
    res, err = quadgk(
        (t) -> (sum((sol(t) .^ 2) ./ 2)), 0.0, 5.0,
        atol = 1.0e-12, rtol = 1.0e-12
    )
    return res
end

f_du0 = ForwardDiff.gradient(G_u, u0)
f_dp = ForwardDiff.gradient(G_p, p)

@test !iszero(f_du0)
@test !iszero(f_dp)

## solve with u0, p

du0, dp = Zygote.gradient(
    (
        u0,
        p,
    ) -> sum(
        solve(
            prob, Tsit5(); u0, p,
            abstol = 1.0e-10, reltol = 1.0e-10, saveat = tsteps,
            sensealg = QuadratureAdjoint(
                abstol = 1.0e-14, reltol = 1.0e-14,
                autojacvec = ZygoteVJP()
            )
        )
    ),
    u0, p
)

@test !iszero(du0)
@test !iszero(dp)

## QuadratureAdjoint with EnzymeVJP on immutable SVector state (issue #1460)
# Out-of-place SVector states must dispatch to the out-of-place `_vecjacobian` and
# `vec_pjac!` EnzymeVJP paths; the result must match the ZygoteVJP path.
du0_cont_z, dp_cont_z = adjoint_sensitivities(
    sol, Tsit5(); dgdu_continuous = dg, g,
    sensealg = QuadratureAdjoint(abstol = 1.0e-14, reltol = 1.0e-14, autojacvec = ZygoteVJP())
)
du0_cont_e, dp_cont_e = adjoint_sensitivities(
    sol, Tsit5(); dgdu_continuous = dg, g,
    sensealg = QuadratureAdjoint(abstol = 1.0e-14, reltol = 1.0e-14, autojacvec = EnzymeVJP())
)
@test !iszero(du0_cont_e)
@test !iszero(dp_cont_e)
@test du0_cont_e ≈ du0_cont_z rtol = 1.0e-10
@test dp_cont_e ≈ dp_cont_z rtol = 1.0e-10

dg_disc_oop(u, p, t, i; outtype = nothing) = u
du0_disc_z, dp_disc_z = adjoint_sensitivities(
    sol, Tsit5(); t = tsteps, dgdu_discrete = dg_disc_oop,
    sensealg = QuadratureAdjoint(abstol = 1.0e-14, reltol = 1.0e-14, autojacvec = ZygoteVJP())
)
du0_disc_e, dp_disc_e = adjoint_sensitivities(
    sol, Tsit5(); t = tsteps, dgdu_discrete = dg_disc_oop,
    sensealg = QuadratureAdjoint(abstol = 1.0e-14, reltol = 1.0e-14, autojacvec = EnzymeVJP())
)
@test du0_disc_e ≈ du0_disc_z rtol = 1.0e-10
@test dp_disc_e ≈ dp_disc_z rtol = 1.0e-10

## GaussAdjoint / GaussKronrodAdjoint on immutable SVector state (issue #1461)
# The immutable-state branch of `ReverseLossCallback` previously asserted
# `sensealg isa QuadratureAdjoint`, the Gauss integrating callbacks passed an
# in-place integrand to the out-of-place adjoint problem, and the out-of-place
# `split_states` ignored checkpointing.

# Discrete cost G(u0, p) = sum over tsteps of sum(u .^ 2) / 2, matching
# dgdu_discrete = dg_disc_oop. ForwardDiff on accurate dense solves is the
# reference.
function G_disc(u0, p)
    tmp_prob = remake(prob; u0 = convert.(promote_type(eltype(u0), eltype(p)), u0), p)
    tmp_sol = solve(tmp_prob, Tsit5(), abstol = 1.0e-14, reltol = 1.0e-14)
    return sum(t -> sum(tmp_sol(t) .^ 2) / 2, tsteps)
end
fd_dp_disc = ForwardDiff.gradient(p -> G_disc(u0, p), p)
fd_du0_disc = ForwardDiff.gradient(u0 -> G_disc(u0, p), u0)

# Dense forward solution: no checkpointing inside GaussAdjoint.
sol_dense = solve(prob, Tsit5(), abstol = 1.0e-14, reltol = 1.0e-14)

for galg in (
        GaussAdjoint(autojacvec = EnzymeVJP()),
        GaussAdjoint(autojacvec = ZygoteVJP()),
        GaussKronrodAdjoint(autojacvec = EnzymeVJP()),
    )
    du0_g, dp_g = adjoint_sensitivities(
        sol_dense, Tsit5(); t = collect(tsteps), dgdu_discrete = dg_disc_oop,
        sensealg = galg, abstol = 1.0e-10, reltol = 1.0e-10
    )
    @test du0_g ≈ fd_du0_disc rtol = 1.0e-6
    @test vec(dp_g) ≈ fd_dp_disc rtol = 1.0e-6
end

# Non-dense (saveat) forward solution: exercises the checkpointing path of the
# out-of-place `split_states` and `GaussIntegrand`.
du0_gc, dp_gc = adjoint_sensitivities(
    sol, Tsit5(); t = collect(tsteps), dgdu_discrete = dg_disc_oop,
    sensealg = GaussAdjoint(autojacvec = EnzymeVJP()),
    abstol = 1.0e-10, reltol = 1.0e-10
)
@test du0_gc ≈ fd_du0_disc rtol = 1.0e-5
@test vec(dp_gc) ≈ fd_dp_disc rtol = 1.0e-5

# Continuous cost, compared against the quadgk + ForwardDiff reference
# (`f_du0`, `f_dp`) computed above.
du0_cont_g, dp_cont_g = adjoint_sensitivities(
    sol_dense, Tsit5(); dgdu_continuous = dg, g,
    sensealg = GaussAdjoint(autojacvec = EnzymeVJP()),
    abstol = 1.0e-10, reltol = 1.0e-10
)
@test du0_cont_g ≈ f_du0 rtol = 1.0e-6
@test vec(dp_cont_g) ≈ f_dp rtol = 1.0e-6

# Through-solve gradient with the default automatic vjp choice.
du0_g, dp_g = Zygote.gradient(
    (u0, p) -> sum(
        solve(
            prob, Tsit5(); u0, p,
            abstol = 1.0e-10, reltol = 1.0e-10, saveat = tsteps,
            sensealg = GaussAdjoint()
        )
    ),
    u0, p
)
du0_q, dp_q = Zygote.gradient(
    (u0, p) -> sum(
        solve(
            prob, Tsit5(); u0, p,
            abstol = 1.0e-10, reltol = 1.0e-10, saveat = tsteps,
            sensealg = QuadratureAdjoint(
                abstol = 1.0e-12, reltol = 1.0e-12, autojacvec = ZygoteVJP()
            )
        )
    ),
    u0, p
)
@test du0_g ≈ du0_q rtol = 1.0e-6
@test dp_g ≈ dp_q rtol = 1.0e-6
