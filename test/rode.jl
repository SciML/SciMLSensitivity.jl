using StochasticDiffEq
using DiffEqSensitivity
using DiffEqNoiseProcess
using LinearAlgebra, Statistics, Random
using Zygote, ReverseDiff, ForwardDiff
using Test
#using DifferentialEquations
seed = 12345
Random.seed!(seed)

function g(u,p,t)
  sum(u.^2.0/2.0)
end

function dg!(out,u,p,t,i)
  (out.=u)
end

@testset "noise iip tests" begin
  function f(du,u,p,t,W)
    du[1] = p[1]*u[1]*sin(W[1] - W[2])
    du[2] = p[2]*u[2]*cos(W[1] + W[2])
    return nothing
  end
  dt = 1e-4
  u0 = [1.00;1.00]
  tspan = (0.0,5.0)
  t = tspan[1]:0.1:tspan[2]
  p = [2.0,-2.0]
  prob = RODEProblem(f,u0,tspan,p)

  sol = solve(prob,RandomEM(),dt=dt, save_noise=true)
  # check reversion with usage of Noise Grid
  _sol = deepcopy(sol)
  noise_reverse = NoiseGrid(reverse(_sol.t),reverse(_sol.W.W))
  prob_reverse = RODEProblem(f,_sol[end],reverse(tspan),p,noise=noise_reverse)
  sol_reverse = solve(prob_reverse,RandomEM(),dt=dt)
  @test sol.u ≈ reverse(sol_reverse.u) rtol=1e-3
  @show minimum(sol.u)

  # Test if Forward and ReverseMode AD agree.
  Random.seed!(seed)
  du0ReverseDiff,dpReverseDiff = Zygote.gradient((u0,p)->sum(
    Array(solve(prob,RandomEM(),dt=dt,u0=u0,p=p,saveat=t,sensealg=ReverseDiffAdjoint())).^2/2)
    ,u0,p)
  Random.seed!(seed)
  dForward = ForwardDiff.gradient((θ)->sum(
    Array(solve(prob,RandomEM(),dt=dt,u0=θ[1:2],p=θ[3:4],saveat=t)).^2/2)
    ,[u0;p])

  @info dForward

  @test du0ReverseDiff ≈ dForward[1:2]
  @test dpReverseDiff ≈ dForward[3:4]

  # test gradients
  Random.seed!(seed)
  sol = solve(prob,RandomEM(),dt=dt, save_noise=true, saveat=t)


  ###
  ## BacksolveAdjoint
  ###

  # ReverseDiff
  du0, dp = adjoint_sensitivities(sol,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=BacksolveAdjoint())

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  @info du0, dp'

  # ReverseDiff with compiled tape
  du0, dp = adjoint_sensitivities(sol,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=ReverseDiffVJP(true)))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2


  du0, dp = adjoint_sensitivities(sol,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=ZygoteVJP()))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  # Tracker
  du0, dp = adjoint_sensitivities(sol,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=TrackerVJP()))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  # isautojacvec = false
  du0, dp = adjoint_sensitivities(sol,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=false))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  # isautojacvec = false and with jac and paramjac
  function jac(J,u,p,t,W)
    J[1,1] = p[1]*sin(W[1] - W[2])
    J[2,1] = zero(u[1])
    J[1,2] = zero(u[1])
    J[2,2] = p[2]*cos(W[1] + W[2])
  end

  function paramjac(J,u,p,t,W)
    J[1,1] = u[1]*sin(W[1] - W[2])
    J[2,1] = zero(u[1])
    J[1,2] = zero(u[1])
    J[2,2] = u[2]*cos(W[1] + W[2])
  end
  Random.seed!(seed)
  faug = RODEFunction(f,jac=jac,paramjac=paramjac)
  prob_aug = RODEProblem{true}(faug,u0,tspan,p)
  sol = solve(prob_aug,RandomEM(),dt=dt, save_noise=true, saveat=t)
  du0, dp = adjoint_sensitivities(sol,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=false))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  ###
  ## InterpolatingAdjoint
  ###

  # test gradients with dense solution and no checkpointing
  Random.seed!(seed)
  sol = solve(prob,RandomEM(),dt=dt, save_noise=true, dense=true)

  # ReverseDiff
  du0, dp = adjoint_sensitivities(sol,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP()))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  # ReverseDiff with compiled tape
  du0, dp = adjoint_sensitivities(sol,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  # Zygote
  du0, dp = adjoint_sensitivities(sol,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  # Tracker
  du0, dp = adjoint_sensitivities(sol,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=TrackerVJP()))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  # isautojacvec = false
  du0, dp = adjoint_sensitivities(sol,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=false))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  # isautojacvec = false and with jac and paramjac
  Random.seed!(seed)
  faug = RODEFunction(f,jac=jac,paramjac=paramjac)
  prob_aug = RODEProblem{true}(faug,u0,tspan,p)
  sol = solve(prob_aug,RandomEM(),dt=dt, save_noise=true, dense=true)
  du0, dp = adjoint_sensitivities(sol,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=false))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  # test gradients with saveat solution and checkpointing
  # need to simulate for dt beyond last tspan to avoid errors in NoiseGrid
  Random.seed!(seed)
  sol = solve(prob,RandomEM(),dt=dt, save_noise=true, dense=true)
  Random.seed!(seed)
  sol_long = solve(remake(prob, tspan=(tspan[1],tspan[2]+10dt)),RandomEM(),dt=dt, save_noise=true, dense=true)

  @test sol_long(t) ≈ sol(t) rtol=1e-12
  @test sol_long.W.W[1:end-10] ≈ sol.W.W[1:end] rtol=1e-12

  # test gradients with saveat solution and checkpointing
  noise = NoiseGrid(sol_long.W.t,sol_long.W.W)
  sol2 = solve(remake(prob,noise=noise,tspan=(tspan[1],tspan[2])),RandomEM(),dt=dt, saveat=t)

  @test sol_long(t) ≈ sol2(t) rtol=1e-12
  @test sol_long.W.W ≈ sol2.W.W rtol=1e-12

  # ReverseDiff
  du0, dp = adjoint_sensitivities(sol2,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=InterpolatingAdjoint(checkpointing=true,autojacvec=ReverseDiffVJP()))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  # ReverseDiff with compiled tape
  du0, dp = adjoint_sensitivities(sol2,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=InterpolatingAdjoint(checkpointing=true,autojacvec=ReverseDiffVJP(true)))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  # Zygote
  du0, dp = adjoint_sensitivities(sol2,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=InterpolatingAdjoint(checkpointing=true,autojacvec=ZygoteVJP()))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  # Tracker
  du0, dp = adjoint_sensitivities(sol2,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=InterpolatingAdjoint(checkpointing=true,autojacvec=TrackerVJP()))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  # isautojacvec = false
  du0, dp = adjoint_sensitivities(sol2,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=InterpolatingAdjoint(checkpointing=true,autojacvec=false))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  # isautojacvec = false and with jac and paramjac
  Random.seed!(seed)
  faug = RODEFunction(f,jac=jac,paramjac=paramjac)
  prob_aug = RODEProblem{true}(faug,u0,tspan,p, noise=noise)
  sol = solve(prob_aug,RandomEM(),dt=dt, save_noise=false, dense=false, saveat=t)
  du0, dp = adjoint_sensitivities(sol,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=InterpolatingAdjoint(checkpointing=true,autojacvec=false))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2
end


@testset "noise oop tests" begin
  function f(u,p,t,W)
    dx = p[1]*u[1]*sin(W[1] - W[2])
    dy = p[2]*u[2]*cos(W[1] + W[2])
    return [dx,dy]
  end
  dt = 1e-4
  u0 = [1.00;1.00]
  tspan = (0.0,5.0)
  t = tspan[1]:0.1:tspan[2]
  p = [2.0,-2.0]
  prob = RODEProblem{false}(f,u0,tspan,p)

  sol = solve(prob,RandomEM(),dt=dt, save_noise=true)
  # check reversion with usage of Noise Grid
  _sol = deepcopy(sol)
  noise_reverse = NoiseGrid(reverse(_sol.t),reverse(_sol.W.W))
  prob_reverse = RODEProblem(f,_sol[end],reverse(tspan),p,noise=noise_reverse)
  sol_reverse = solve(prob_reverse,RandomEM(),dt=dt)
  @test sol.u ≈ reverse(sol_reverse.u) rtol=1e-3
  @show minimum(sol.u)

  # Test if Forward and ReverseMode AD agree.
  Random.seed!(seed)
  du0ReverseDiff,dpReverseDiff = Zygote.gradient((u0,p)->sum(
    Array(solve(prob,RandomEM(),dt=dt,u0=u0,p=p,saveat=t,sensealg=ReverseDiffAdjoint())).^2/2)
    ,u0,p)
  Random.seed!(seed)
  dForward = ForwardDiff.gradient((θ)->sum(
    Array(solve(prob,RandomEM(),dt=dt,u0=θ[1:2],p=θ[3:4],saveat=t)).^2/2)
    ,[u0;p])

  @info dForward

  @test du0ReverseDiff ≈ dForward[1:2]
  @test dpReverseDiff ≈ dForward[3:4]

  # test gradients
  Random.seed!(seed)
  sol = solve(prob,RandomEM(),dt=dt, save_noise=true, saveat=t)

  ###
  ## BacksolveAdjoint
  ###

  # ReverseDiff
  du0, dp = adjoint_sensitivities(sol,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=ReverseDiffVJP()))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  @info du0, dp'

  # ReverseDiff with compiled tape
  du0, dp = adjoint_sensitivities(sol,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=ReverseDiffVJP(true)))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  # Zygote
  du0, dp = adjoint_sensitivities(sol,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=ZygoteVJP()))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  # Tracker
  du0, dp = adjoint_sensitivities(sol,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=TrackerVJP()))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  # isautojacvec = false
  du0, dp = adjoint_sensitivities(sol,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=false))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  # isautojacvec = false and with jac and paramjac
  function jac(J,u,p,t,W)
    J[1,1] = p[1]*sin(W[1] - W[2])
    J[2,1] = zero(u[1])
    J[1,2] = zero(u[1])
    J[2,2] = p[2]*cos(W[1] + W[2])
  end

  function paramjac(J,u,p,t,W)
    J[1,1] = u[1]*sin(W[1] - W[2])
    J[2,1] = zero(u[1])
    J[1,2] = zero(u[1])
    J[2,2] = u[2]*cos(W[1] + W[2])
  end
  Random.seed!(seed)
  faug = RODEFunction(f,jac=jac,paramjac=paramjac)
  prob_aug = RODEProblem{false}(faug,u0,tspan,p)
  sol = solve(prob_aug,RandomEM(),dt=dt, save_noise=true, saveat=t)
  du0, dp = adjoint_sensitivities(sol,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=false))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  ###
  ## InterpolatingAdjoint
  ###

  # test gradients with dense solution and no checkpointing
  Random.seed!(seed)
  sol = solve(prob,RandomEM(),dt=dt, save_noise=true, dense=true)

  # ReverseDiff
  du0, dp = adjoint_sensitivities(sol,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP()))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  # ReverseDiff with compiled tape
  du0, dp = adjoint_sensitivities(sol,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  # Zygote
  du0, dp = adjoint_sensitivities(sol,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  # Tracker
  du0, dp = adjoint_sensitivities(sol,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=TrackerVJP()))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  # isautojacvec = false
  du0, dp = adjoint_sensitivities(sol,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=false))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  # isautojacvec = false and with jac and paramjac
  Random.seed!(seed)
  faug = RODEFunction(f,jac=jac,paramjac=paramjac)
  prob_aug = RODEProblem{false}(faug,u0,tspan,p)
  sol = solve(prob_aug,RandomEM(),dt=dt, save_noise=true, dense=true)
  du0, dp = adjoint_sensitivities(sol,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=false))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  # test gradients with saveat solution and checkpointing
  # need to simulate for dt beyond last tspan to avoid errors in NoiseGrid
  Random.seed!(seed)
  sol = solve(prob,RandomEM(),dt=dt, save_noise=true, dense=true)
  Random.seed!(seed)
  sol_long = solve(remake(prob, tspan=(tspan[1],tspan[2]+10dt)),RandomEM(),dt=dt, save_noise=true, dense=true)

  @test sol_long(t) ≈ sol(t) rtol=1e-12
  @test sol_long.W.W[1:end-10] ≈ sol.W.W[1:end] rtol=1e-12

  # test gradients with saveat solution and checkpointing
  noise = NoiseGrid(sol_long.W.t,sol_long.W.W)
  sol2 = solve(remake(prob,noise=noise,tspan=(tspan[1],tspan[2])),RandomEM(),dt=dt, saveat=t)

  @test sol_long(t) ≈ sol2(t) rtol=1e-12
  @test sol_long.W.W ≈ sol2.W.W rtol=1e-12

  # ReverseDiff
  du0, dp = adjoint_sensitivities(sol2,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=InterpolatingAdjoint(checkpointing=true,autojacvec=ReverseDiffVJP()))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  # ReverseDiff with compiled tape
  du0, dp = adjoint_sensitivities(sol2,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=InterpolatingAdjoint(checkpointing=true,autojacvec=ReverseDiffVJP(true)))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  # Zygote
  du0, dp = adjoint_sensitivities(sol2,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=InterpolatingAdjoint(checkpointing=true,autojacvec=ZygoteVJP()))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  # Tracker
  du0, dp = adjoint_sensitivities(sol2,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=InterpolatingAdjoint(checkpointing=true,autojacvec=TrackerVJP()))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  # isautojacvec = false
  du0, dp = adjoint_sensitivities(sol2,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=InterpolatingAdjoint(checkpointing=true,autojacvec=false))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  # isautojacvec = false and with jac and paramjac
  Random.seed!(seed)
  faug = RODEFunction(f,jac=jac,paramjac=paramjac)
  prob_aug = RODEProblem{false}(faug,u0,tspan,p,noise=noise)
  sol = solve(prob_aug,RandomEM(),dt=dt, save_noise=false, saveat=t, dense=false)
  du0, dp = adjoint_sensitivities(sol,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=InterpolatingAdjoint(checkpointing=true,autojacvec=false))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2
end
