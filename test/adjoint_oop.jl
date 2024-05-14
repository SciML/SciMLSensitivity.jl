using SciMLSensitivity, OrdinaryDiffEq, StaticArrays, QuadGK, ForwardDiff, Zygote
using Test

##StaticArrays rrule
u0 = @SVector rand(2)
p = @SVector rand(4)

function lotka(u, p, svec = true)
    du1 = p[1] * u[1] - p[2] * u[1] * u[2]
    du2 = -p[3] * u[2] + p[4] * u[1] * u[2]
    if svec
        @SVector [du1, du2]
    else
        @SMatrix [du1 du2 du1; du2 du1 du1]
    end
end

#SVector constructor adjoint
function loss(p)
    u = lotka(u0, p)
    sum(1 .- u)
end

grad = Zygote.gradient(loss, p)
@test grad[1] isa SArray
grad2 = ForwardDiff.gradient(loss, p)
@test grad[1]≈grad2 rtol=1e-12

#SMatrix constructor adjoint
function loss_mat(p)
    u = lotka(u0, p, false)
    sum(1 .- u)
end

grad = Zygote.gradient(loss_mat, p)
@test grad[1] isa SArray
grad2 = ForwardDiff.gradient(loss_mat, p)
@test grad[1]≈grad2 rtol=1e-12

##Adjoints of StaticArrays ODE

u0 = @SVector [1.0, 1.0]
p = @SVector [1.5, 1.0, 3.0, 1.0]
tspan = (0.0, 5.0)
datasize = 15
tsteps = range(tspan[1], tspan[2], length = datasize)

function lotka(u, p, t)
    du1 = p[1] * u[1] - p[2] * u[1] * u[2]
    du2 = -p[3] * u[2] + p[4] * u[1] * u[2]
    @SVector [du1, du2]
end

prob = ODEProblem(lotka, u0, tspan, p)
sol = solve(prob, Tsit5(), saveat = tsteps, abstol = 1e-14, reltol = 1e-14)

## Discrete Case
dg_disc(u, p, t, i; outtype = nothing) = u

du0, dp = adjoint_sensitivities(sol, Tsit5(); t = tsteps, dgdu_discrete = dg_disc,
    sensealg = QuadratureAdjoint(abstol = 1e-14, reltol = 1e-14, autojacvec = ZygoteVJP()))

@test !iszero(du0)
@test !iszero(dp)
#
adj_prob = ODEAdjointProblem(sol,
    QuadratureAdjoint(abstol = 1e-14, reltol = 1e-14, autojacvec = ZygoteVJP()),
    Tsit5(), tsteps, dg_disc)
adj_sol = solve(adj_prob, Tsit5(), abstol = 1e-14, reltol = 1e-14)
integrand = AdjointSensitivityIntegrand(sol, adj_sol,
    QuadratureAdjoint(abstol = 1e-14, reltol = 1e-14, autojacvec = ZygoteVJP()))
res, err = quadgk(integrand, 0.0, 5.0, atol = 1e-14, rtol = 1e-14)

@test adj_sol.u[end]≈du0 rtol=1e-12
@test res≈dp rtol=1e-12

###Comparing with gradients of lotka volterra with normal arrays
u2 = [1.0, 1.0]
p2 = [1.5, 1.0, 3.0, 1.0]

function f(u, p, t)
    du1 = p[1] * u[1] - p[2] * u[1] * u[2]
    du2 = -p[3] * u[2] + p[4] * u[1] * u[2]
    [du1, du2]
end

prob2 = ODEProblem(f, u2, tspan, p2)
sol2 = solve(prob, Tsit5(), saveat = tsteps, abstol = 1e-14, reltol = 1e-14)

function dg_disc(du, u, p, t, i)
    du .= u
end

du1, dp1 = adjoint_sensitivities(sol, Tsit5(); t = tsteps, dgdu_discrete = dg_disc,
    sensealg = QuadratureAdjoint(abstol = 1e-14, reltol = 1e-14, autojacvec = ZygoteVJP()))

@test du0≈du1 rtol=1e-12
@test dp≈dp1 rtol=1e-12

## with ForwardDiff and Zygote

function G_p(p)
    tmp_prob = remake(prob, u0 = convert.(eltype(p), prob.u0), p = p)
    sol = solve(tmp_prob, Tsit5(), abstol = 1e-14, reltol = 1e-14,
        sensealg = QuadratureAdjoint(abstol = 1e-14, reltol = 1e-14,
            autojacvec = ZygoteVJP()), saveat = tsteps)
    u = Array(sol)
    return sum(((1 .- u) .^ 2) ./ 2)
end

function G_u(u0)
    tmp_prob = remake(prob, u0 = u0, p = prob.p)
    sol = solve(tmp_prob, Tsit5(), saveat = tsteps,
        sensealg = QuadratureAdjoint(abstol = 1e-14, reltol = 1e-14,
            autojacvec = ZygoteVJP()), abstol = 1e-14,
        reltol = 1e-14)
    u = Array(sol)

    return sum(((1 .- u) .^ 2) ./ 2)
end

G_p(p)
G_u(u0)
f_dp = ForwardDiff.gradient(G_p, p)
f_du0 = ForwardDiff.gradient(G_u, u0)

z_dp = Zygote.gradient(G_p, p)
z_du0 = Zygote.gradient(G_u, u0)

@test z_du0[1]≈f_du0 rtol=1e-12
@test z_dp[1]≈f_dp rtol=1e-12

## Continuous Case

g(u, p, t) = sum((u .^ 2) ./ 2)

function dg(u, p, t)
    u
end

du0, dp = adjoint_sensitivities(sol, Tsit5(); dgdu_continuous = dg, g = g,
    sensealg = QuadratureAdjoint(abstol = 1e-14, reltol = 1e-14, autojacvec = ZygoteVJP()))

@test !iszero(du0)
@test !iszero(dp)

adj_prob = ODEAdjointProblem(sol,
    QuadratureAdjoint(abstol = 1e-14, reltol = 1e-14, autojacvec = ZygoteVJP()),
    Tsit5(), nothing, nothing, nothing, dg, nothing, g)
adj_sol = solve(adj_prob, Tsit5(), abstol = 1e-14, reltol = 1e-14)
integrand = AdjointSensitivityIntegrand(sol, adj_sol,
    QuadratureAdjoint(abstol = 1e-14, reltol = 1e-14, autojacvec = ZygoteVJP()))
res, err = quadgk(integrand, 0.0, 5.0, atol = 1e-14, rtol = 1e-14)

@test adj_sol.u[end]≈du0 rtol=1e-12
@test res≈dp rtol=1e-12

##ForwardDiff

function G_p(p)
    tmp_prob = remake(prob, p = p)
    sol = solve(tmp_prob, Tsit5(), abstol = 1e-12, reltol = 1e-12)
    res, err = quadgk((t) -> (sum((sol(t) .^ 2) ./ 2)), 0.0, 5.0, atol = 1e-12,
        rtol = 1e-12)
    res
end

function G_u(u0)
    tmp_prob = remake(prob, u0 = u0)
    sol = solve(tmp_prob, Tsit5(), abstol = 1e-12, reltol = 1e-12)
    res, err = quadgk((t) -> (sum((sol(t) .^ 2) ./ 2)), 0.0, 5.0, atol = 1e-12,
        rtol = 1e-12)
    res
end

f_du0 = ForwardDiff.gradient(G_u, u0)
f_dp = ForwardDiff.gradient(G_p, p)

@test !iszero(f_du0)
@test !iszero(f_dp)

## concrete solve

du0, dp = Zygote.gradient(
    (u0, p) -> sum(concrete_solve(prob, Tsit5(), u0, p,
        abstol = 1e-10, reltol = 1e-10, saveat = tsteps,
        sensealg = QuadratureAdjoint(abstol = 1e-14, reltol = 1e-14,
            autojacvec = ZygoteVJP()))),
    u0, p)

@test !iszero(du0)
@test !iszero(dp)
