using Test, LinearAlgebra
using DiffEqSensitivity, StochasticDiffEq
using Random

@info "SDE Adjoints"

seed = 100
Random.seed!(seed)

tstart = 0.0
tend = 0.1
dt = 0.005
trange = (tstart, tend)
t = tstart:dt:tend
tarray = collect(t)

function g(u,p,t)
  sum(u.^2.0/2.0)
end

function dg!(out,u,p,t,i)
  (out.=u)
end

p2 = [1.01,0.87]


# scalar noise
@testset "SDE inplace scalar noise tests" begin
  using DiffEqNoiseProcess

  dtscalar = tend/1e3

  f!(du,u,p,t) = (du .= p[1]*u)
  σ!(du,u,p,t) = (du .= p[2]*u)

  @info "scalar SDE"

  Random.seed!(seed)
  W = WienerProcess(0.0,0.0,0.0)
  u0 = rand(2)

  linear_analytic_strat(u0,p,t,W) = @.(u0*exp(p[1]*t+p[2]*W))

  prob = SDEProblem(SDEFunction(f!,σ!,analytic=linear_analytic_strat),σ!,u0,trange,p2,
    noise=W
    )
  sol = solve(prob,EulerHeun(), dt=dtscalar, save_noise=true)

  @test isapprox(sol.u_analytic,sol.u, atol=1e-4)

  res_sde_u0, res_sde_p = adjoint_sensitivities(sol,EulerHeun(),dg!,Array(t)
    ,dt=dtscalar,adaptive=false,sensealg=BacksolveAdjoint())

  @show res_sde_u0, res_sde_p

  res_sde_u02, res_sde_p2 = adjoint_sensitivities(sol,EulerHeun(),dg!,Array(t)
    ,dt=dtscalar,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=false))

  @test isapprox(res_sde_u0, res_sde_u02,  atol=1e-8)
  @test isapprox(res_sde_p, res_sde_p2,  atol=1e-8)

  res_sde_u02, res_sde_p2 = adjoint_sensitivities(sol,EulerHeun(),dg!,Array(t)
    ,dt=dtscalar,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=ReverseDiffVJP()))

  @test isapprox(res_sde_u0, res_sde_u02,  atol=1e-8)
  @test isapprox(res_sde_p, res_sde_p2,  atol=1e-8)

  @show res_sde_u02, res_sde_p2


  res_sde_u02, res_sde_p2 = adjoint_sensitivities(sol,EulerHeun(),dg!,Array(t)
    ,dt=tend/1e2,adaptive=false,sensealg=InterpolatingAdjoint())


  @test isapprox(res_sde_u0, res_sde_u02,  rtol=1e-4)
  @test isapprox(res_sde_p, res_sde_p2,  rtol=1e-4)

  @show res_sde_u02, res_sde_p2

  res_sde_u02, res_sde_p2 = adjoint_sensitivities(sol,EulerHeun(),dg!,Array(t)
    ,dt=dtscalar,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=false))

  @test isapprox(res_sde_u0, res_sde_u02,  rtol=1e-4)
  @test isapprox(res_sde_p, res_sde_p2,  rtol=1e-4)

  @show res_sde_u02, res_sde_p2

  res_sde_u02, res_sde_p2 = adjoint_sensitivities(sol,EulerHeun(),dg!,Array(t)
    ,dt=dtscalar,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP()))

  @test isapprox(res_sde_u0, res_sde_u02,  rtol=1e-4)
  @test isapprox(res_sde_p, res_sde_p2,  rtol=1e-4)

  @show res_sde_u02, res_sde_p2

  function compute_grads(sol, scale=1.0)
    _sol = deepcopy(sol)
    _sol.W.save_everystep = false
    xdis = _sol(tarray)
    helpu1 = [u[1] for u in xdis.u]
    tmp1 = sum((@. xdis.t*helpu1*helpu1))

    Wtmp = [_sol.W(t)[1][1] for t in tarray]
    tmp2 = sum((@. Wtmp*helpu1*helpu1))

    tmp3 = sum((@. helpu1*helpu1))/helpu1[1]

    return [tmp3, scale*tmp3], [tmp1*(1.0+scale^2), tmp2*(1.0+scale^2)]
  end

  true_grads = compute_grads(sol, u0[2]/u0[1])

  @show  true_grads

  @test isapprox(res_sde_u0, res_sde_u02,  rtol=1e-4)
  @test isapprox(res_sde_p, res_sde_p2,  rtol=1e-4)
  @test isapprox(true_grads[2], res_sde_p', atol=1e-4)
  @test isapprox(true_grads[1], res_sde_u0, rtol=1e-4)
  @test isapprox(true_grads[2], res_sde_p2', atol=1e-4)
  @test isapprox(true_grads[1], res_sde_u02, rtol=1e-4)
end

@testset "SDE oop scalar noise tests" begin
  using DiffEqNoiseProcess

  dtscalar = tend/1e3

  f(u,p,t) = p[1]*u
  σ(u,p,t) = p[2]*u

  Random.seed!(seed)
  W = WienerProcess(0.0,0.0,0.0)
  u0 = rand(2)

  linear_analytic_strat(u0,p,t,W) = @.(u0*exp(p[1]*t+p[2]*W))

  prob = SDEProblem(SDEFunction(f,σ,analytic=linear_analytic_strat),σ,u0,trange,p2,
    noise=W
   )
  sol = solve(prob,EulerHeun(), dt=dtscalar, save_noise=true)

  @test isapprox(sol.u_analytic,sol.u, atol=1e-4)

  res_sde_u0, res_sde_p = adjoint_sensitivities(sol,EulerHeun(),dg!,Array(t)
    ,dt=dtscalar,adaptive=false,sensealg=BacksolveAdjoint())

  @show res_sde_u0, res_sde_p

  res_sde_u02, res_sde_p2 = adjoint_sensitivities(sol,EulerHeun(),dg!,Array(t)
    ,dt=dtscalar,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=false))

  @test isapprox(res_sde_u0, res_sde_u02,  atol=1e-8)
  @test isapprox(res_sde_p, res_sde_p2,  atol=1e-8)

  res_sde_u02, res_sde_p2 = adjoint_sensitivities(sol,EulerHeun(),dg!,Array(t)
    ,dt=dtscalar,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=ReverseDiffVJP()))

  @test isapprox(res_sde_u0, res_sde_u02,  atol=1e-8)
  @test isapprox(res_sde_p, res_sde_p2,  atol=1e-8)

  @show res_sde_u02, res_sde_p2

  res_sde_u02, res_sde_p2 = adjoint_sensitivities(sol,EulerHeun(),dg!,Array(t)
    ,dt=tend/1e2,adaptive=false,sensealg=InterpolatingAdjoint())


  @test isapprox(res_sde_u0, res_sde_u02,  rtol=1e-4)
  @test isapprox(res_sde_p, res_sde_p2,  atol=1e-4)

  @show res_sde_u02, res_sde_p2

  res_sde_u02, res_sde_p2 = adjoint_sensitivities(sol,EulerHeun(),dg!,Array(t)
    ,dt=dtscalar,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=false))

  @test isapprox(res_sde_u0, res_sde_u02,  rtol=1e-4)
  @test isapprox(res_sde_p, res_sde_p2,  atol=1e-4)

  @show res_sde_u02, res_sde_p2

  res_sde_u02, res_sde_p2 = adjoint_sensitivities(sol,EulerHeun(),dg!,Array(t)
    ,dt=dtscalar,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP()))

  @test isapprox(res_sde_u0, res_sde_u02,  rtol=1e-4)
  @test isapprox(res_sde_p, res_sde_p2,  atol=1e-4)

  @show res_sde_u02, res_sde_p2

  function compute_grads(sol, scale=1.0)
    _sol = deepcopy(sol)
    _sol.W.save_everystep = false
    xdis = _sol(tarray)
    helpu1 = [u[1] for u in xdis.u]
    tmp1 = sum((@. xdis.t*helpu1*helpu1))

    Wtmp = [_sol.W(t)[1][1] for t in tarray]
    tmp2 = sum((@. Wtmp*helpu1*helpu1))

    tmp3 = sum((@. helpu1*helpu1))/helpu1[1]

    return [tmp3, scale*tmp3], [tmp1*(1.0+scale^2), tmp2*(1.0+scale^2)]
  end

  true_grads = compute_grads(sol, u0[2]/u0[1])

  @show  true_grads


  @test isapprox(true_grads[2], res_sde_p', atol=1e-4)
  @test isapprox(true_grads[1], res_sde_u0, rtol=1e-4)
  @test isapprox(true_grads[2], res_sde_p2', atol=1e-4)
  @test isapprox(true_grads[1], res_sde_u02, rtol=1e-4)

end
