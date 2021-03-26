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
  (out.=-u)
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
    ,dt=dt,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=DiffEqSensitivity.ReverseDiffVJP(true)))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2


  du0, dp = adjoint_sensitivities(sol,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=DiffEqSensitivity.ZygoteVJP()))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  # Tracker
  du0, dp = adjoint_sensitivities(sol,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=DiffEqSensitivity.TrackerVJP()))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  # isautojacvec = true
  du0, dp = adjoint_sensitivities(sol,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=true))

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
    ,dt=dt,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=DiffEqSensitivity.ReverseDiffVJP()))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-1

  # ReverseDiff with compiled tape
  du0, dp = adjoint_sensitivities(sol,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=DiffEqSensitivity.ReverseDiffVJP(true)))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  # Zygote
  du0, dp = adjoint_sensitivities(sol,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=DiffEqSensitivity.ZygoteVJP()))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  # Tracker
  du0, dp = adjoint_sensitivities(sol,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=DiffEqSensitivity.TrackerVJP()))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  # isautojacvec = true
  du0, dp = adjoint_sensitivities(sol,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=true))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  # isautojacvec = false
  du0, dp = adjoint_sensitivities(sol,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=false))

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
  sol = solve(prob_aug,RandomEM(),dt=dt, save_noise=true, dense=true)
  du0, dp = adjoint_sensitivities(sol,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=false))

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
    ,dt=dt,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=DiffEqSensitivity.ReverseDiffVJP()))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  @info du0, dp'

  # ReverseDiff with compiled tape
  du0, dp = adjoint_sensitivities(sol,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=DiffEqSensitivity.ReverseDiffVJP(true)))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  # Zygote
  du0, dp = adjoint_sensitivities(sol,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=DiffEqSensitivity.ZygoteVJP()))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  # Tracker
  du0, dp = adjoint_sensitivities(sol,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=DiffEqSensitivity.TrackerVJP()))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  # isautojacvec = true
  du0, dp = adjoint_sensitivities(sol,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=true))

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
    ,dt=dt,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=DiffEqSensitivity.ReverseDiffVJP()))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-1

  # ReverseDiff with compiled tape
  du0, dp = adjoint_sensitivities(sol,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=DiffEqSensitivity.ReverseDiffVJP(true)))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  # Zygote
  du0, dp = adjoint_sensitivities(sol,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=DiffEqSensitivity.ZygoteVJP()))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  # Tracker
  du0, dp = adjoint_sensitivities(sol,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=DiffEqSensitivity.TrackerVJP()))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  # isautojacvec = true
  du0, dp = adjoint_sensitivities(sol,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=true))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

  # isautojacvec = false
  du0, dp = adjoint_sensitivities(sol,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=false))

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
  sol = solve(prob_aug,RandomEM(),dt=dt, save_noise=true, dense=true)
  du0, dp = adjoint_sensitivities(sol,RandomEM(),dg!,Array(t)
    ,dt=dt,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=false))

  @test du0ReverseDiff ≈ du0 rtol=1e-2
  @test dpReverseDiff ≈ dp' rtol=1e-2

end


function f(u,p,t,W)
  dx = p[1]*u[1]*sin(W[1] - W[2])
  dy = p[2]*u[2]*cos(W[1] + W[2])
  return [dx,dy]
end


dt = 1e-4
u0 = [1.00;1.00]
tspan = (0.0,1.0)
t = tspan[1]:0.1:tspan[2]
p = [2.0,-2.0]

prob = RODEProblem{false}(f,u0,tspan,p)
Random.seed!(seed)
sol = solve(prob, RandomEM(),dt=dt, save_noise=true, dense=true)
# check reversion with usage of Noise Grid
_sol = deepcopy(sol)
noise_reverse = NoiseGrid(reverse(_sol.t),reverse(_sol.W.W))
prob_reverse = RODEProblem(f,_sol[end],reverse(tspan),p,noise=noise_reverse)
sol_reverse = solve(prob_reverse,RandomEM(),dt=dt)
@test sol(t).u ≈ sol_reverse(t).u rtol=1e-3
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

du0, dp = adjoint_sensitivities(_sol,RandomEM(),dg!,Array(t)
  ,dt=dt,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=DiffEqSensitivity.ReverseDiffVJP()))

@test du0ReverseDiff ≈ du0 rtol=1e-2
@test dpReverseDiff ≈ dp' rtol=1e-1

@info du0, dp'

adj_prob_dense = DiffEqSensitivity.RODEAdjointProblem(_sol,InterpolatingAdjoint(),dg!,t,nothing)
adj_sol_dense = solve(adj_prob_dense, RandomEM(), dt=dt)
adj_prob_checkpointing = DiffEqSensitivity.RODEAdjointProblem(sol,InterpolatingAdjoint(autojacvec=DiffEqSensitivity.ZygoteVJP()),dg!,t,nothing)
adj_sol_checkpointing = solve(adj_prob_checkpointing, RandomEM(), dt=dt)

@show adj_sol_dense[end]
@show adj_sol_checkpointing[end]


@test du0ReverseDiff ≈ -adj_sol_checkpointing[end][1:2] rtol=1e-2
@test dpReverseDiff ≈ adj_sol_checkpointing[end][3:4] rtol=1e-2


@test du0ReverseDiff ≈ -adj_sol_dense[end][1:2] rtol=1e-2
@test dpReverseDiff ≈ adj_sol_dense[end][3:4] rtol=1e-2

minimum(abs.(adj_sol_checkpointing.W.t - adj_sol_dense.W.t))

using Plots
pl1 = plot(adj_sol_dense, label="Dense")
plot!(pl1,adj_sol_checkpointing, label="Checkpointing", legend=false)

pl2 = plot(adj_sol_dense.t,hcat(adj_sol_dense-adj_sol_checkpointing ...)')

maximum(adj_sol_dense - adj_sol_checkpointing)

adj_prob_backsolve = DiffEqSensitivity.RODEAdjointProblem(sol,BacksolveAdjoint(),dg!,t,nothing)
adj_sol_backsolve = solve(adj_prob_backsolve, RandomEM(), dt=dt)

plot(getindex.(adj_sol_backsolve(t).u, 3) -getindex.(adj_sol_dense(t).u, 3))
plot!(getindex.(adj_sol_backsolve(t).u, 4) -getindex.(adj_sol_dense(t).u, 4))


plot(getindex.(adj_sol_backsolve(t).u, 3) -getindex.(adj_sol_checkpointing(t).u, 3))
plot!(getindex.(adj_sol_backsolve(t).u, 4) -getindex.(adj_sol_checkpointing(t).u, 4))
