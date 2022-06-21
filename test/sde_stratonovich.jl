using Test, LinearAlgebra
using OrdinaryDiffEq
using DiffEqSensitivity, StochasticDiffEq, DiffEqBase
using ForwardDiff, ReverseDiff
using Random
import Tracker, Zygote

@info "SDE Adjoints"

seed = 100
Random.seed!(seed)
abstol = 1e-4
reltol = 1e-4

u₀ = [0.5]
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


@testset "SDE oop Tests (no noise)" begin

  f_oop_linear(u,p,t) = p[1]*u
  σ_oop_linear(u,p,t) = p[2]*u

  p = [1.01,0.0]

  # generate ODE adjoint results

  prob_oop_ode = ODEProblem(f_oop_linear,u₀,(tstart,tend),p)
  sol_oop_ode = solve(prob_oop_ode,Tsit5(),saveat=t,abstol=abstol,reltol=reltol)
  res_ode_u0, res_ode_p = adjoint_sensitivities(sol_oop_ode,Tsit5(),dg!,t
    ,abstol=abstol,reltol=reltol,sensealg=BacksolveAdjoint())

  function G(p)
    tmp_prob = remake(prob_oop_ode,u0=eltype(p).(prob_oop_ode.u0),p=p,
                    tspan=eltype(p).(prob_oop_ode.tspan),abstol=abstol, reltol=reltol)
    sol = solve(tmp_prob,Tsit5(),saveat=tarray,abstol=abstol, reltol=reltol)
    res = g(sol,p,nothing)
  end
  res_ode_forward = ForwardDiff.gradient(G,p)

  @test isapprox(res_ode_forward[1], sum(@. u₀^2*exp(2*p[1]*t)*t), rtol = 1e-4)
  #@test isapprox(res_ode_reverse[1], sum(@. u₀^2*exp(2*p[1]*t)*t), rtol = 1e-4)
  @test isapprox(res_ode_p'[1], sum(@. u₀^2*exp(2*p[1]*t)*t), rtol = 1e-4)
  #@test isapprox(res_ode_p', res_ode_trackerp, rtol = 1e-4)

  # SDE adjoint results (with noise == 0, so should agree with above)

  Random.seed!(seed)
  prob_oop_sde = SDEProblem(f_oop_linear,σ_oop_linear,u₀,trange,p)
  sol_oop_sde = solve(prob_oop_sde,EulerHeun(),dt=1e-4,adaptive=false,save_noise=true)
  res_sde_u0, res_sde_p = adjoint_sensitivities(sol_oop_sde,
      EulerHeun(),dg!,t,dt=1e-2,sensealg=BacksolveAdjoint())

  @info res_sde_p

  res_sde_u0a, res_sde_pa = adjoint_sensitivities(sol_oop_sde,
      EulerHeun(),dg!,t,dt=1e-2,sensealg=InterpolatingAdjoint())

  @test isapprox(res_sde_u0, res_sde_u0a, rtol = 1e-6)
  @test isapprox(res_sde_p, res_sde_pa, rtol = 1e-6)

  function GSDE1(p)
    Random.seed!(seed)
    tmp_prob = remake(prob_oop_sde,u0=eltype(p).(prob_oop_sde.u0),p=p,
                      tspan=eltype(p).(prob_oop_sde.tspan))
    sol = solve(tmp_prob,RKMil(interpretation=:Stratonovich),dt=tend/10000,adaptive=false,sensealg=DiffEqBase.SensitivityADPassThrough(),saveat=tarray)
    A = convert(Array,sol)
    res = g(A,p,nothing)
  end
  res_sde_forward = ForwardDiff.gradient(GSDE1,p)

  noise = vec((@. sol_oop_sde.W(tarray)))
  Wfix = [W[1][1] for W in noise]
  @test isapprox(res_sde_forward[1], sum(@. u₀^2*exp(2*p[1]*t)*t), rtol = 1e-4)
  @test isapprox(res_sde_p'[1], sum(@. u₀^2*exp(2*p[1]*t)*t), rtol = 1e-4)
  @test isapprox(res_sde_p'[2], sum(@. (Wfix)*u₀^2*exp(2*(p[1])*tarray+2*p[2]*Wfix)), rtol = 1e-4)
end

@testset "SDE oop Tests (with noise)" begin

  f_oop_linear(u,p,t) = p[1]*u
  σ_oop_linear(u,p,t) = p[2]*u

  # SDE adjoint results (with noise != 0)
  dt1 = tend/1e3

  Random.seed!(seed)
  prob_oop_sde2 = SDEProblem(f_oop_linear,σ_oop_linear,u₀,trange,p2)
  sol_oop_sde2 = solve(prob_oop_sde2,RKMil(interpretation=:Stratonovich),dt=dt1,adaptive=false,save_noise=true)

  res_sde_u02, res_sde_p2 = adjoint_sensitivities(sol_oop_sde2,EulerHeun(),dg!,tarray
      ,dt=dt1,adaptive=false,sensealg=BacksolveAdjoint())

  @info res_sde_p2

  # test consitency for different switches for the noise Jacobian
  res_sde_u02a, res_sde_p2a = adjoint_sensitivities(sol_oop_sde2,EulerHeun(),dg!,tarray
      ,dt=dt1,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=false))

  @test isapprox(res_sde_u02, res_sde_u02a, rtol = 1e-6)
  @test isapprox(res_sde_p2, res_sde_p2a, rtol = 1e-6)

  @info res_sde_p2a

  res_sde_u02a, res_sde_p2a = adjoint_sensitivities(sol_oop_sde2,EulerHeun(),dg!,tarray
      ,dt=dt1,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=ZygoteVJP()))

  @test isapprox(res_sde_u02, res_sde_u02a, rtol = 1e-6)
  @test isapprox(res_sde_p2, res_sde_p2a, rtol = 1e-6)

  @info res_sde_p2a

  res_sde_u02a, res_sde_p2a = adjoint_sensitivities(sol_oop_sde2,EulerHeun(),dg!,tarray
      ,dt=tend/dt1,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=ReverseDiffVJP()))

  @test isapprox(res_sde_u02, res_sde_u02a, rtol = 1e-6)
  @test isapprox(res_sde_p2, res_sde_p2a, rtol = 1e-6)

  @info res_sde_p2a

  function GSDE2(p)
    Random.seed!(seed)
    tmp_prob = remake(prob_oop_sde2,u0=eltype(p).(prob_oop_sde2.u0),p=p,
                      tspan=eltype(p).(prob_oop_sde2.tspan)
                      #,abstol=abstol, reltol=reltol
                      )
    sol = solve(tmp_prob,RKMil(interpretation=:Stratonovich),dt=dt1,adaptive=false,sensealg=DiffEqBase.SensitivityADPassThrough(),saveat=tarray)
    A = convert(Array,sol)
    res = g(A,p,nothing)
  end
  res_sde_forward2 = ForwardDiff.gradient(GSDE2,p2)


  Wfix = [sol_oop_sde2.W(t)[1][1] for t in tarray]
  resp1 = sum(@. tarray*u₀^2*exp(2*(p2[1])*tarray+2*p2[2]*Wfix))
  resp2 = sum(@. (Wfix)*u₀^2*exp(2*(p2[1])*tarray+2*p2[2]*Wfix))
  resp = [resp1, resp2]

  @test isapprox(res_sde_forward2, resp, rtol = 8e-4)

  @test isapprox(res_sde_p2', res_sde_forward2, rtol = 1e-3)
  @test isapprox(res_sde_p2', resp, rtol = 1e-3)

  @info "ForwardDiff" res_sde_forward2
  @info "Exact" resp
  @info "BacksolveAdjoint SDE" res_sde_p2

  # InterpolatingAdjoint
  @info "InterpolatingAdjoint SDE"

  res_sde_u02a, res_sde_p2a = adjoint_sensitivities(sol_oop_sde2,EulerHeun(),dg!,tarray
      ,dt=dt1,adaptive=false,sensealg=InterpolatingAdjoint())

  @test isapprox(res_sde_u02, res_sde_u02a, rtol = 1e-4)
  @test isapprox(res_sde_p2, res_sde_p2a, rtol = 1e-3)

  @info res_sde_p2a

  res_sde_u02, res_sde_p2 = adjoint_sensitivities(sol_oop_sde2,EulerHeun(),dg!,tarray
      ,dt=dt1,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=false))

  @test isapprox(res_sde_u02, res_sde_u02a, rtol = 1e-6)
  @test isapprox(res_sde_p2, res_sde_p2a, rtol = 1e-6)

  res_sde_u02, res_sde_p2 = adjoint_sensitivities(sol_oop_sde2,EulerHeun(),dg!,tarray
      ,dt=dt1,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))

  @test isapprox(res_sde_u02, res_sde_u02a, rtol = 1e-6)
  @test isapprox(res_sde_p2, res_sde_p2a, rtol = 1e-6)


  res_sde_u02, res_sde_p2 = adjoint_sensitivities(sol_oop_sde2,EulerHeun(),dg!,tarray
      ,dt=dt1,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP()))

  @test isapprox(res_sde_u02, res_sde_u02a, rtol = 1e-6)
  @test isapprox(res_sde_p2, res_sde_p2a, rtol = 1e-6)


  # Free memory to help Travis

  Wfix = nothing
  res_sde_forward2 = nothing
  res_sde_reverse2 = nothing
  resp = nothing
  res_sde_trackerp2 = nothing
  res_sde_u02 = nothing
  sol_oop_sde2 = nothing
  res_sde_u02a = nothing
  res_sde_p2a = nothing
  res_sde_p2 = nothing
  sol_oop_sde = nothing
  GC.gc()

  # SDE adjoint results with diagonal noise

  Random.seed!(seed)
  prob_oop_sde2 = SDEProblem(f_oop_linear,σ_oop_linear,[u₀;u₀;u₀],trange,p2)
  sol_oop_sde2 = solve(prob_oop_sde2,EulerHeun(),dt=dt1,adaptive=false,save_noise=true)

  @info "Diagonal Adjoint"

  res_sde_u02, res_sde_p2 = adjoint_sensitivities(sol_oop_sde2,EulerHeun(),dg!,tarray
      ,dt=dt1,adaptive=false,sensealg=BacksolveAdjoint())

  res_sde_u02a, res_sde_p2a = adjoint_sensitivities(sol_oop_sde2,EulerHeun(),dg!,tarray
      ,dt=dt1,adaptive=false,sensealg=InterpolatingAdjoint())

  @test isapprox(res_sde_p2, res_sde_p2a, rtol = 5e-4)
  @test isapprox(res_sde_u02, res_sde_u02a, rtol = 2e-5)

  res_sde_u02a, res_sde_p2a = adjoint_sensitivities(sol_oop_sde2,EulerHeun(),dg!,tarray
      ,dt=dt1,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=false))

  @test isapprox(res_sde_p2, res_sde_p2a, rtol = 5e-4)
  @test isapprox(res_sde_u02, res_sde_u02a, rtol = 2e-5)

  res_sde_u02a, res_sde_p2a = adjoint_sensitivities(sol_oop_sde2,EulerHeun(),dg!,tarray
      ,dt=dt1,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))

  @test isapprox(res_sde_p2, res_sde_p2a, rtol = 5e-4)
  @test isapprox(res_sde_u02, res_sde_u02a, rtol = 2e-5)

  res_sde_u02a, res_sde_p2a = adjoint_sensitivities(sol_oop_sde2,EulerHeun(),dg!,tarray
      ,dt=dt1,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP()))

  @test isapprox(res_sde_p2, res_sde_p2a, rtol = 5e-4)
  @test isapprox(res_sde_u02, res_sde_u02a, rtol = 2e-5)


  res_sde_u02a, res_sde_p2a = adjoint_sensitivities(sol_oop_sde2,EulerHeun(),dg!,tarray
      ,dt=dt1,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=false))

  @test isapprox(res_sde_p2, res_sde_p2a, rtol = 1e-7)
  @test isapprox(res_sde_u02, res_sde_u02a, rtol = 1e-7)

  res_sde_u02a, res_sde_p2a = adjoint_sensitivities(sol_oop_sde2,EulerHeun(),dg!,tarray
      ,dt=dt1,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=ZygoteVJP()))

  @test isapprox(res_sde_p2, res_sde_p2a, rtol = 1e-7)
  @test isapprox(res_sde_u02, res_sde_u02a, rtol = 1e-7)

  res_sde_u02a, res_sde_p2a = adjoint_sensitivities(sol_oop_sde2,EulerHeun(),dg!,tarray
      ,dt=dt1,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=ReverseDiffVJP()))

  @test isapprox(res_sde_p2, res_sde_p2a, rtol = 1e-7)
  @test isapprox(res_sde_u02, res_sde_u02a, rtol = 1e-7)

  @info res_sde_p2

  sol_oop_sde2 = nothing
  GC.gc()


  @info "Diagonal ForwardDiff"
  res_sde_forward2 = ForwardDiff.gradient(GSDE2,p2)

  #@test isapprox(res_sde_forward2, res_sde_reverse2, rtol = 1e-6)
  @test isapprox(res_sde_p2', res_sde_forward2, rtol = 1e-3)
  #@test isapprox(res_sde_p2', res_sde_reverse2, rtol = 1e-3)

  # u0
  function GSDE3(u)
    Random.seed!(seed)
    tmp_prob = remake(prob_oop_sde2,u0=u)
    sol = solve(tmp_prob,RKMil(interpretation=:Stratonovich),dt=dt1,adaptive=false,saveat=tarray)
    A = convert(Array,sol)
    res = g(A,nothing,nothing)
  end

  @info "ForwardDiff u0"
  res_sde_forward2 = ForwardDiff.gradient(GSDE3,[u₀;u₀;u₀])

  @test isapprox(res_sde_u02, res_sde_forward2, rtol = 1e-4)

end



##
## Inplace
##
@testset "SDE inplace Tests" begin

  f!(du,u,p,t) = du.=p[1]*u
  σ!(du,u,p,t) = du.=p[2]*u

  dt1 = tend/1e3

  Random.seed!(seed)
  prob_sde = SDEProblem(f!,σ!,u₀,trange,p2)
  sol_sde = solve(prob_sde,EulerHeun(),dt=dt1,adaptive=false, save_noise=true)

  function GSDE(p)
    Random.seed!(seed)
    tmp_prob = remake(prob_sde,u0=eltype(p).(prob_sde.u0),p=p,
                      tspan=eltype(p).(prob_sde.tspan))
    sol = solve(tmp_prob,EulerHeun(),dt=dt1,adaptive=false,saveat=tarray)
    A = convert(Array,sol)
    res = g(A,p,nothing)
  end

  res_sde_forward = ForwardDiff.gradient(GSDE,p2)


  res_sde_u0, res_sde_p = adjoint_sensitivities(sol_sde,EulerHeun(),dg!,tarray
      ,dt=dt1,adaptive=false,sensealg=BacksolveAdjoint())

  @test isapprox(res_sde_p', res_sde_forward, rtol = 1e-4)

  @info res_sde_p

  res_sde_u02, res_sde_p2 = adjoint_sensitivities(sol_sde,EulerHeun(),dg!,tarray
      ,dt=dt1,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=false))

  @info res_sde_p2

  @test isapprox(res_sde_p, res_sde_p2, rtol = 1e-5)
  @test isapprox(res_sde_u0, res_sde_u02, rtol = 1e-5)

  res_sde_u02, res_sde_p2 = adjoint_sensitivities(sol_sde,EulerHeun(),dg!,tarray
      ,dt=dt1,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=ZygoteVJP()))

  @info res_sde_p2

  @test isapprox(res_sde_p, res_sde_p2, rtol = 1e-5) # not broken here because it just uses the vjps
  @test isapprox(res_sde_u0 ,res_sde_u02, rtol = 1e-5)

  res_sde_u02, res_sde_p2 = adjoint_sensitivities(sol_sde,EulerHeun(),dg!,tarray
      ,dt=dt1,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=ReverseDiffVJP()))

  @info res_sde_p2

  @test isapprox(res_sde_p, res_sde_p2, rtol = 1e-10)
  @test isapprox(res_sde_u0 ,res_sde_u02, rtol = 1e-10)


  res_sde_u02, res_sde_p2 = adjoint_sensitivities(sol_sde,EulerHeun(),dg!,tarray
      ,dt=dt1,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP()))

  @test isapprox(res_sde_p, res_sde_p2, rtol = 2e-4)
  @test isapprox(res_sde_u0 ,res_sde_u02, rtol = 1e-4)

  res_sde_u0, res_sde_p = adjoint_sensitivities(sol_sde,EulerHeun(),dg!,tarray
      ,dt=dt1,adaptive=false,sensealg=InterpolatingAdjoint())

  @test isapprox(res_sde_p, res_sde_p2, rtol = 1e-7)
  @test isapprox(res_sde_u0 ,res_sde_u02, rtol = 1e-7)

  res_sde_u0, res_sde_p = adjoint_sensitivities(sol_sde,EulerHeun(),dg!,tarray
      ,dt=dt1,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=false))

  @test isapprox(res_sde_p, res_sde_p2, rtol = 1e-7)
  @test isapprox(res_sde_u0 ,res_sde_u02, rtol = 1e-7)

  res_sde_u02, res_sde_p2 = adjoint_sensitivities(sol_sde,EulerHeun(),dg!,tarray
      ,dt=dt1,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))

  @test isapprox(res_sde_p, res_sde_p2, rtol = 1e-7)
  @test isapprox(res_sde_u0 ,res_sde_u02, rtol = 1e-7)

  # diagonal noise

  #compare with oop version
  f_oop_linear(u,p,t) = p[1]*u
  σ_oop_linear(u,p,t) = p[2]*u
  Random.seed!(seed)
  prob_oop_sde = SDEProblem(f_oop_linear,σ_oop_linear,[u₀;u₀;u₀],trange,p2)
  sol_oop_sde = solve(prob_oop_sde,EulerHeun(),dt=dt1,adaptive=false,save_noise=true)
  res_oop_u0, res_oop_p = adjoint_sensitivities(sol_oop_sde,EulerHeun(),dg!,tarray
      ,dt=dt1,adaptive=false,sensealg=BacksolveAdjoint())

  @info res_oop_p

  Random.seed!(seed)
  prob_sde = SDEProblem(f!,σ!,[u₀;u₀;u₀],trange,p2)
  sol_sde = solve(prob_sde,EulerHeun(),dt=dt1,adaptive=false,save_noise=true)

  res_sde_u0, res_sde_p = adjoint_sensitivities(sol_sde,EulerHeun(),dg!,tarray
      ,dt=dt1,adaptive=false,sensealg=BacksolveAdjoint())

  isapprox(res_sde_p, res_oop_p, rtol = 1e-6)
  isapprox(res_sde_u0 ,res_oop_u0, rtol = 1e-6)

  @info res_sde_p

  res_sde_u0, res_sde_p = adjoint_sensitivities(sol_sde,EulerHeun(),dg!,tarray
      ,dt=dt1,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=false))

  @test isapprox(res_sde_p, res_oop_p, rtol = 1e-6)
  @test isapprox(res_sde_u0 ,res_oop_u0, rtol = 1e-6)

  @info res_sde_p

  res_sde_u0, res_sde_p = adjoint_sensitivities(sol_sde,EulerHeun(),dg!,tarray
      ,dt=dt1,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=ZygoteVJP()))

  @test isapprox(res_sde_p, res_oop_p, rtol = 1e-6)
  @test isapprox(res_sde_u0 ,res_oop_u0, rtol = 1e-6)

  @info res_sde_p

  res_sde_u0, res_sde_p = adjoint_sensitivities(sol_sde,EulerHeun(),dg!,tarray
      ,dt=dt1,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=ReverseDiffVJP()))

  @test isapprox(res_sde_p, res_oop_p, rtol = 1e-6)
  @test isapprox(res_sde_u0 ,res_oop_u0, rtol = 1e-6)

  @info res_sde_p


  res_sde_u0, res_sde_p = adjoint_sensitivities(sol_sde,EulerHeun(),dg!,tarray
      ,dt=dt1,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP()))

  @test isapprox(res_sde_p, res_oop_p, rtol = 5e-4)
  @test isapprox(res_sde_u0 ,res_oop_u0, rtol = 1e-4)

  res_sde_u0, res_sde_p = adjoint_sensitivities(sol_sde,EulerHeun(),dg!,tarray
      ,dt=dt1,adaptive=false,sensealg=InterpolatingAdjoint())

  @test isapprox(res_sde_p, res_oop_p, rtol = 5e-4)
  @test isapprox(res_sde_u0 ,res_oop_u0, rtol = 1e-4)

  res_sde_u0, res_sde_p = adjoint_sensitivities(sol_sde,EulerHeun(),dg!,tarray
      ,dt=dt1,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=false))

  @test isapprox(res_sde_p, res_oop_p, rtol = 5e-4)
  @test isapprox(res_sde_u0 ,res_oop_u0, rtol = 1e-4)

  res_sde_u0, res_sde_p = adjoint_sensitivities(sol_sde,EulerHeun(),dg!,tarray
      ,dt=dt1,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))

  @test_broken isapprox(res_sde_p, res_oop_p, rtol = 1e-4)
  @test isapprox(res_sde_u0 ,res_oop_u0, rtol = 1e-4)
end


@testset "SDE oop Tests (Tracker)" begin

  f_oop_linear(u,p,t) = p[1]*u
  σ_oop_linear(u,p,t) = p[2]*u

  function f_oop_linear(u::Tracker.TrackedArray,p,t)
    p[1].*u
  end

  function σ_oop_linear(u::Tracker.TrackedArray,p,t)
    p[2].*u
  end

  Random.seed!(seed)
  prob_oop_sde = SDEProblem(f_oop_linear,σ_oop_linear,u₀,trange,p2)

  function GSDE1(p)
    Random.seed!(seed)
    tmp_prob = remake(prob_oop_sde,u0=eltype(p).(prob_oop_sde.u0),p=p,
                      tspan=eltype(p).(prob_oop_sde.tspan))
    sol = solve(tmp_prob,RKMil(interpretation=:Stratonovich),dt=5e-4,adaptive=false,sensealg=DiffEqBase.SensitivityADPassThrough(),saveat=tarray)
    A = convert(Array,sol)
    res = g(A,p,nothing)
  end
  res_sde_forward = ForwardDiff.gradient(GSDE1,p2)

  Random.seed!(seed)
  res_sde_trackeru0, res_sde_trackerp = Zygote.gradient((u0,p)->sum(Array(solve(prob_oop_sde,
    RKMil(interpretation=:Stratonovich),dt=5e-4,adaptive=false,u0=u0,p=p,saveat=tarray,
    sensealg=TrackerAdjoint())).^2.0/2.0),u₀,p2)

  @test isapprox(res_sde_forward, res_sde_trackerp, rtol = 1e-5)
end
