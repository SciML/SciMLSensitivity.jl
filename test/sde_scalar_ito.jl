using Test, LinearAlgebra
using DiffEqSensitivity, StochasticDiffEq
using Random
using DiffEqNoiseProcess
using ForwardDiff
using ReverseDiff

@info "SDE scalar Adjoints"

seed = 100
Random.seed!(seed)

tstart = 0.0
tend = 0.1
trange = (tstart, tend)
t = tstart:0.01:tend
tarray = collect(t)

function g(u,p,t)
  sum(u.^2.0/2.0)
end

function dg!(out,u,p,t,i)
  (out.=u)
end

dt = tend/1e4

# non-exploding initialization.
α = 1/(exp(-randn())+1)
β = -α^2 - 1/(exp(-randn())+1)
p = [α,β]

fIto(u,p,t) = p[1]*u #p[1]*u.+p[2]^2/2*u
fStrat(u,p,t) = p[1]*u.-p[2]^2/2*u #p[1]*u
σ(u,p,t) = p[2]*u

 # Ito sense (Strat sense for commented version)
linear_analytic(u0,p,t,W) = @.(u0*exp(p[1]*t+p[2]*W))
corfunc(u,p,t) = p[2]^2*u

"""
1D oop
"""

# generate noise values
# Z = randn(length(tstart:dt:tend))
# Z1 = cumsum([0;sqrt(dt)*Z[1:end]])
# NG = NoiseGrid(Array(tstart:dt:(tend+dt)),[Z for Z in Z1])

# set initial state
u0 = [1/6]

# define problem in Ito sense
Random.seed!(seed)
probIto = SDEProblem(fIto,
  σ,u0,trange,p,
  #noise=NG
 )

# solve Ito sense
solIto = solve(probIto, EM(), dt=dt, adaptive=false, save_noise=true, saveat=dt)


# define problem in Stratonovich sense
Random.seed!(seed)
probStrat = SDEProblem(SDEFunction(fStrat,σ,),
  σ,u0,trange,p,
  #noise=NG
  )

# solve Strat sense
solStrat = solve(probStrat,RKMil(interpretation=:Stratonovich), dt=dt,
  adaptive=false, save_noise=true,  saveat=dt)

# check that forward solution agrees
@test isapprox(solIto.u, solStrat.u, rtol=1e-3)
@test isapprox(solIto.u, solStrat.u, atol=1e-2)
#@test isapprox(solIto.u, solIto.u_analytic, rtol=1e-3)


"""
solve with continuous adjoint sensitivity tools
"""

# for Ito sense
gs_u0, gs_p = adjoint_sensitivities(solIto,EM(),dg!,Array(t)
  ,dt=dt,adaptive=false,sensealg=BacksolveAdjoint(),corfunc_analytical=corfunc)

@info gs_u0, gs_p

gs_u0a, gs_pa = adjoint_sensitivities(solIto,EM(),dg!,Array(t)
  ,dt=dt,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=DiffEqSensitivity.ReverseDiffVJP()))

@info gs_u0a, gs_pa

@test isapprox(gs_u0, gs_u0a, rtol=1e-8)
@test isapprox(gs_p, gs_pa, rtol=1e-8)

# for Strat sense
res_u0, res_p = adjoint_sensitivities(solStrat,EulerHeun(),dg!,Array(t)
  ,dt=dt,adaptive=false,sensealg=BacksolveAdjoint())

@info res_u0, res_p


"""
Tests with respect to analytical result, forward and reverse mode AD
"""

# tests for parameter gradients

function Gp(p; sensealg = ReverseDiffAdjoint())
  Random.seed!(seed)
  tmp_prob = remake(probStrat,p=p)
  _sol = solve(tmp_prob,EulerHeun(),dt=dt,adaptive=false,saveat=Array(t),sensealg=sensealg)
  A = convert(Array,_sol)
  res = g(A,p,nothing)
end

res_forward = ForwardDiff.gradient(p -> Gp(p,sensealg=ForwardDiffSensitivity()), p)

@info res_forward

Wfix = [solStrat.W(t)[1][1] for t in tarray]
resp1 = sum(@. tarray*u0^2*exp(2*(p[1]-p[2]^2/2)*tarray+2*p[2]*Wfix))
resp2 = sum(@. (Wfix-p[2]*tarray)*u0^2*exp(2*(p[1]-p[2]^2/2)*tarray+2*p[2]*Wfix))
resp = [resp1, resp2]

@show resp


@test isapprox(resp, gs_p', atol=3e-2) # exact vs ito adjoint
@test isapprox(res_p, gs_p, atol=3e-2) # strat vs ito adjoint
@test isapprox(gs_p', res_forward, atol=3e-2) # ito adjoint vs forward
@test isapprox(resp, res_p', rtol=2e-5) # exact vs strat adjoint
@test isapprox(resp, res_forward, rtol=2e-5) # exact vs forward

# tests for initial state gradients

function Gu0(u0; sensealg = ReverseDiffAdjoint())
  Random.seed!(seed)
  tmp_prob = remake(probStrat,u0=u0)
  _sol = solve(tmp_prob,EulerHeun(),dt=dt,adaptive=false,saveat=Array(t),sensealg=sensealg)
  A = convert(Array,_sol)
  res = g(A,p,nothing)
end

res_forward = ForwardDiff.gradient(u0 -> Gu0(u0,sensealg=ForwardDiffSensitivity()), u0)

resu0 = sum(@. u0*exp(2*(p[1]-p[2]^2/2)*tarray+2*p[2]*Wfix))
@show resu0

@test isapprox(resu0, gs_u0[1], rtol=5e-2) # exact vs ito adjoint
@test isapprox(res_u0, gs_u0, rtol=5e-2) # strat vs ito adjoint
@test isapprox(gs_u0, res_forward, rtol=5e-2) # ito adjoint vs forward
@test isapprox(resu0, res_u0[1], rtol=1e-3) # exact vs strat adjoint
@test isapprox(res_u0, res_forward, rtol=1e-3) # strat adjoint vs forward
@test isapprox(resu0, res_forward[1], rtol=1e-3)  # exact vs forward




adj_probStrat = SDEAdjointProblem(solStrat,BacksolveAdjoint(autojacvec=ZygoteVJP()),dg!,t,nothing)
adj_solStrat = solve(adj_probStrat,EulerHeun(), dt=dt)

#@show adj_solStrat[end]

adj_probIto = SDEAdjointProblem(solIto,BacksolveAdjoint(autojacvec=ZygoteVJP()),dg!,t,nothing,
  corfunc_analytical=corfunc)
adj_solIto = solve(adj_probIto,EM(), dt=dt)

@test isapprox(adj_solStrat[4,:], adj_solIto[4,:], rtol=1e-3)


# using Plots
# pl1 = plot(solStrat, label="Strat forward")
# plot!(pl1,solIto, label="Ito forward")
#
# pl1 = plot(adj_solStrat.t, adj_solStrat[4,:], label="Strat reverse")
# plot!(pl1,adj_solIto.t, adj_solIto[4,:], label="Ito reverse")
#
# pl2 = plot(adj_solStrat.t, adj_solStrat[1,:], label="Strat reverse")
# plot!(pl2, adj_solIto.t, adj_solIto[1,:], label="Ito reverse", legend=:bottomright)
#
# pl3 = plot(adj_solStrat.t, adj_solStrat[2,:], label="Strat reverse")
# plot!(pl3, adj_solIto.t, adj_solIto[2,:], label="Ito reverse")
#
# pl4 = plot(adj_solStrat.t, adj_solStrat[3,:], label="Strat reverse")
# plot!(pl4, adj_solIto.t, adj_solIto[3,:], label="Ito reverse")
#
# pl = plot(pl1,pl2,pl3,pl4)
#
# savefig(pl, "plot.png")
