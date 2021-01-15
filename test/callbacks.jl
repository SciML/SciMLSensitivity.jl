using StochasticDiffEq, OrdinaryDiffEq, Zygote
using DiffEqSensitivity, Test, ForwardDiff

# ODEs

function fiip(du,u,p,t)
  du[1] = dx = p[1]*u[1] - p[2]*u[1]*u[2]
  du[2] = dy = -p[3]*u[2] + p[4]*u[1]*u[2]
end
function foop(u,p,t)
  dx = p[1]*u[1] - p[2]*u[1]*u[2]
  dy = -p[3]*u[2] + p[4]*u[1]*u[2]
  [dx,dy]
end

p = [1.5,1.0,3.0,1.0]; u0 = [1.0;1.0]

prob = ODEProblem(fiip,u0,(0.0,10.0),p)
proboop = ODEProblem(foop,u0,(0.0,10.0),p)

### callback with no effect

condition(u,t,integrator) = t == 5
affect!(integrator) = integrator.u[1] += 0.0
cb = DiscreteCallback(condition,affect!,save_positions=(false,false))

sol1 = solve(prob,Tsit5(),u0=u0,p=p,callback=cb,tstops=[5.0],abstol=1e-14,reltol=1e-14,saveat=0.1)
sol2 = solve(prob,Tsit5(),u0=u0,p=p,tstops=[5.0],abstol=1e-14,reltol=1e-14,saveat=0.1)

@test sol1.u == sol2.u

du01,dp1 = Zygote.gradient(
  (u0,p)->sum(solve(prob,Tsit5(),u0=u0,p=p,callback=cb,tstops=[5.0],abstol=1e-14,reltol=1e-14,saveat=0.1,sensealg=BacksolveAdjoint())),
  u0,p)

du01b,dp1b = Zygote.gradient(
  (u0,p)->sum(solve(proboop,Tsit5(),u0=u0,p=p,callback=cb,tstops=[5.0],abstol=1e-14,reltol=1e-14,saveat=0.1,sensealg=BacksolveAdjoint())),
  u0,p)

du02,dp2 = Zygote.gradient(
  (u0,p)->sum(solve(prob,Tsit5(),u0=u0,p=p,callback=cb,tstops=[5.0],abstol=1e-14,reltol=1e-14,saveat=0.1,sensealg=ReverseDiffAdjoint()))
  ,u0,p)

du03,dp3 = Zygote.gradient(
  (u0,p)->sum(solve(prob,Tsit5(),u0=u0,p=p,callback=cb,tstops=[5.0],abstol=1e-14,reltol=1e-14,saveat=0.1,sensealg=InterpolatingAdjoint())),
  u0,p)

du04,dp4 = Zygote.gradient(
  (u0,p)->sum(solve(prob,Tsit5(),u0=u0,p=p,callback=cb,tstops=[5.0],abstol=1e-14,reltol=1e-14,saveat=0.1,sensealg=QuadratureAdjoint())),
  u0,p)

dstuff = ForwardDiff.gradient(
  (θ)->sum(solve(prob,Tsit5(),u0=θ[1:2],p=θ[3:6],callback=cb,tstops=[5.0],abstol=1e-14,reltol=1e-14,saveat=0.1)),
  [u0;p])

@info dstuff

@test du01 ≈ dstuff[1:2]
@test dp1 ≈ dstuff[3:6]
@test du01b ≈ dstuff[1:2]
@test dp1b ≈ dstuff[3:6]
@test du01 ≈ du02
@test du01 ≈ du03
@test du01 ≈ du04
@test dp1 ≈ dp2
@test dp1 ≈ dp3
@test dp1 ≈ dp4

@test du02 ≈ dstuff[1:2]
@test dp2 ≈ dstuff[3:6]

function dg!(out,u,p,t,i)
  (out.=-1)
end


cb2 = DiffEqSensitivity.track_callbacks(CallbackSet(cb),prob.tspan[1])
sol_track = solve(prob,Tsit5(),u0=u0,p=p,callback=cb2,tstops=[5.0],abstol=1e-14,reltol=1e-14,saveat=0.1)
cb_adj = DiffEqSensitivity.setup_reverse_callbacks(cb2,BacksolveAdjoint())

adj_prob = ODEAdjointProblem(sol_track,BacksolveAdjoint(),dg!,sol_track.t,nothing,
						 callback = cb2,
						 abstol=1e-14,reltol=1e-14)
adj_sol = solve(adj_prob, Tsit5(), abstol=1e-14,reltol=1e-14)
@test adj_sol(sol1.t)[end-1:end,:] ≈ sol1[:,:]
@test du01 ≈ -adj_sol[1:2,end]
@test dp1 ≈ adj_sol[3:6,end]

### callback with no effect except saving the state

condition(u,t,integrator) = t == 5
affect!(integrator) = integrator.u[1] += 0.0
cb = DiscreteCallback(condition,affect!)

sol1 = solve(prob,Tsit5(),u0=u0,p=p,callback=cb,tstops=[5.0],abstol=1e-14,reltol=1e-14,saveat=0.1)
sol2 = solve(prob,Tsit5(),u0=u0,p=p,tstops=[5.0],abstol=1e-14,reltol=1e-14,saveat=0.1)

@test length(sol1.t) != length(sol2.t)

du01,dp1 = Zygote.gradient(
  (u0,p)->sum(solve(prob,Tsit5(),u0=u0,p=p,callback=cb,tstops=[5.0],abstol=1e-14,reltol=1e-14,saveat=0.1,sensealg=BacksolveAdjoint(checkpointing=true))),
  u0,p)

du01a,dp1a = Zygote.gradient(
  (u0,p)->sum(solve(prob,Tsit5(),u0=u0,p=p,callback=cb,tstops=[5.0],abstol=1e-14,reltol=1e-14,saveat=0.1,sensealg=BacksolveAdjoint(checkpointing=false))),
  u0,p)

du02,dp2 = Zygote.gradient(
  (u0,p)->sum(solve(prob,Tsit5(),u0=u0,p=p,callback=cb,tstops=[5.0],abstol=1e-14,reltol=1e-14,saveat=0.1,sensealg=ReverseDiffAdjoint()))
  ,u0,p)

du03,dp3 = Zygote.gradient(
  (u0,p)->sum(solve(prob,Tsit5(),u0=u0,p=p,callback=cb,tstops=[5.0],abstol=1e-14,reltol=1e-14,saveat=0.1,sensealg=InterpolatingAdjoint()))
  ,u0,p)

du04,dp4 = Zygote.gradient(
  (u0,p)->sum(solve(prob,Tsit5(),u0=u0,p=p,callback=cb,tstops=[5.0],abstol=1e-14,reltol=1e-14,saveat=0.1,sensealg=QuadratureAdjoint()))
  ,u0,p)

dstuff = ForwardDiff.gradient(
  (θ)->sum(solve(prob,Tsit5(),u0=θ[1:2],p=θ[3:6],callback=cb,tstops=[5.0],abstol=1e-14,reltol=1e-14,saveat=0.1)),
  [u0;p])

@info dstuff

@test du01 ≈ dstuff[1:2]
@test dp1 ≈ dstuff[3:6]
@test du01a ≈ dstuff[1:2]
@test dp1a ≈ dstuff[3:6]
@test du01 ≈ du02
@test du01 ≈ du03
@test du01 ≈ du04
@test dp1 ≈ dp2
@test dp1 ≈ dp3
@test dp1 ≈ dp4

@test du01 ≈ du01a
@test dp1 ≈ dp1a

@test du02 ≈ dstuff[1:2]
@test dp2 ≈ dstuff[3:6]


cb2 = DiffEqSensitivity.track_callbacks(CallbackSet(cb),prob.tspan[1])
sol_track = solve(prob,Tsit5(),u0=u0,p=p,callback=cb2,tstops=[5.0],abstol=1e-14,reltol=1e-14,saveat=0.1)
cb_adj = DiffEqSensitivity.setup_reverse_callbacks(cb2,BacksolveAdjoint())


adj_prob = ODEAdjointProblem(sol_track,BacksolveAdjoint(),dg!,sol_track.t,nothing,
						 callback = cb2,
						 abstol=1e-14,reltol=1e-14)
adj_sol1 = solve(adj_prob, Tsit5(), abstol=1e-14,reltol=1e-14)

@test adj_sol1(sol1.t)[end-1:end,:] ≈ sol1[:,:]
@test adj_sol(sol1.t)[end-1:end,:] ≈ sol1[:,:]
@test du01 ≈ -adj_sol1[1:2,end]
@test dp1 ≈ adj_sol1[3:6,end]

adj_prob = ODEAdjointProblem(sol_track,InterpolatingAdjoint(),dg!,sol_track.t,nothing,
						 callback = cb2,
						 abstol=1e-14,reltol=1e-14)
adj_sol1 = solve(adj_prob, Tsit5(), abstol=1e-14,reltol=1e-14)

@test du01 ≈ -adj_sol1[1:2,end]
@test dp1 ≈ adj_sol1[3:6,end]

### callback at single time point

condition(u,t,integrator) = t == 5
affect!(integrator) = integrator.u[1] += 2.0
cb = DiscreteCallback(condition,affect!)

sol1 = solve(prob,Tsit5(),u0=u0,p=p,callback=cb,tstops=[5.0],abstol=1e-14,reltol=1e-14,saveat=0.1)

du01,dp1 = Zygote.gradient(
  (u0,p)->sum(solve(prob,Tsit5(),u0=u0,p=p,callback=cb,tstops=[5.0],abstol=1e-14,reltol=1e-14,saveat=0.1,sensealg=BacksolveAdjoint(checkpointing=true))),
  u0,p)

du02,dp2 = Zygote.gradient(
  (u0,p)->sum(solve(prob,Tsit5(),u0=u0,p=p,callback=cb,tstops=[5.0],abstol=1e-14,reltol=1e-14,saveat=0.1,sensealg=ReverseDiffAdjoint()))
  ,u0,p)

du03,dp3 = Zygote.gradient(
  (u0,p)->sum(solve(prob,Tsit5(),u0=u0,p=p,callback=cb,tstops=[5.0],abstol=1e-14,reltol=1e-14,saveat=0.1,sensealg=InterpolatingAdjoint()))
  ,u0,p)

du04,dp4 = Zygote.gradient(
  (u0,p)->sum(solve(prob,Tsit5(),u0=u0,p=p,callback=cb,tstops=[5.0],abstol=1e-14,reltol=1e-14,saveat=0.1,sensealg=QuadratureAdjoint()))
  ,u0,p)

dstuff = ForwardDiff.gradient(
  (θ)->sum(solve(prob,Tsit5(),u0=θ[1:2],p=θ[3:6],callback=cb,tstops=[5.0],abstol=1e-14,reltol=1e-14,saveat=0.1)),
  [u0;p])

@test du01 ≈ dstuff[1:2]
@test dp1 ≈ dstuff[3:6]
@test du01 ≈ du02
@test_broken du01 ≈ du03
@test du01 ≈ du04
@test dp1 ≈ dp2
@test_broken dp1 ≈ dp3
@test dp1 ≈ dp4

@test du02 ≈ dstuff[1:2]
@test dp2 ≈ dstuff[3:6]

cb2 = DiffEqSensitivity.track_callbacks(CallbackSet(cb),prob.tspan[1])
sol_track = solve(prob,Tsit5(),u0=u0,p=p,callback=cb2,tstops=[5.0],abstol=1e-14,reltol=1e-14,saveat=0.1)
cb_adj = DiffEqSensitivity.setup_reverse_callbacks(cb2,BacksolveAdjoint())

adj_prob = ODEAdjointProblem(sol_track,BacksolveAdjoint(checkpointing=false),dg!,sol_track.t,nothing,
						 callback = cb2,
						 abstol=1e-14,reltol=1e-14)
adj_sol1 = solve(adj_prob, Tsit5(), tstops=sol_track.t, abstol=1e-14,reltol=1e-14)


@test du01 ≈ -adj_sol1[1:2,end]
@test dp1 ≈ adj_sol1[3:6,end]

adj_prob = ODEAdjointProblem(sol_track,InterpolatingAdjoint(checkpointing=false),dg!,sol_track.t,nothing,
						 callback = cb2,
						 abstol=1e-14,reltol=1e-14)
adj_sol1 = solve(adj_prob, Tsit5(), tstops=sol_track.t, abstol=1e-14,reltol=1e-14)


@test_broken du01 ≈ -adj_sol1[1:2,end]
@test_broken dp1 ≈ adj_sol1[3:6,end]
@test du03 ≈ -adj_sol1[1:2,end]
@test dp3 ≈ adj_sol1[3:6,end]

### other callback at single time point

condition(u,t,integrator) = t == 5
affect!(integrator) = (integrator.u[1] = 2.0; @show "triggered!")
cb = DiscreteCallback(condition,affect!)

du01,dp1 = Zygote.gradient(
  (u0,p)->sum(solve(prob,Tsit5(),u0=u0,p=p,callback=cb,tstops=[5.0],abstol=1e-14,reltol=1e-14,saveat=0.1,sensealg=BacksolveAdjoint(checkpointing=true)))
  ,u0,p)

du02,dp2 = Zygote.gradient(
  (u0,p)->sum(solve(prob,Tsit5(),u0=u0,p=p,callback=cb,tstops=[5.0],abstol=1e-14,reltol=1e-14,saveat=0.1,sensealg=ReverseDiffAdjoint())),
  u0,p)

du03,dp3 = Zygote.gradient(
  (u0,p)->sum(solve(prob,Tsit5(),u0=u0,p=p,callback=cb,tstops=[5.0],abstol=1e-14,reltol=1e-14,saveat=0.1,sensealg=InterpolatingAdjoint())),
  u0,p)

du04,dp4 = Zygote.gradient(
  (u0,p)->sum(solve(prob,Tsit5(),u0=u0,p=p,callback=cb,tstops=[5.0],abstol=1e-14,reltol=1e-14,saveat=0.1,sensealg=QuadratureAdjoint())),
  u0,p)

dstuff = ForwardDiff.gradient(
  (θ)->sum(solve(prob,Tsit5(),u0=θ[1:2],p=θ[3:6],callback=cb,tstops=[5.0],abstol=1e-14,reltol=1e-14,saveat=0.1)),
  [u0;p])

@test_broken du01 ≈ dstuff[1:2]
@test_broken dp1 ≈ dstuff[3:6]
@test_broken du01 ≈ du02
@test_broken du01 ≈ du03
@test_broken du01 ≈ du04
@test_broken dp1 ≈ dp2
@test_broken dp1 ≈ dp3
@test_broken dp1 ≈ dp4

@test du02 ≈ dstuff[1:2]
@test dp2 ≈ dstuff[3:6]


### callbacks at multiple time points

affecttimes = [2.0,4.0,8.0]
condition(u,t,integrator) = t ∈ affecttimes
affect!(integrator) = integrator.u[1] += 2.0
cb = DiscreteCallback(condition,affect!)

du01,dp1 = Zygote.gradient(
  (u0,p)->sum(solve(prob,Tsit5(),u0=u0,p=p,callback=cb,tstops=affecttimes,abstol=1e-14,reltol=1e-14,saveat=0.1,sensealg=BacksolveAdjoint(checkpointing=true))),
  u0,p)

du01a,dp1a = Zygote.gradient(
  (u0,p)->sum(solve(prob,Tsit5(),u0=u0,p=p,callback=cb,tstops=affecttimes,abstol=1e-14,reltol=1e-14,saveat=0.1,sensealg=BacksolveAdjoint(checkpointing=false))),
  u0,p)

du02,dp2 = Zygote.gradient(
  (u0,p)->sum(solve(prob,Tsit5(),u0=u0,p=p,callback=cb,tstops=affecttimes,abstol=1e-14,reltol=1e-14,saveat=0.1,sensealg=ReverseDiffAdjoint())),
  u0,p)

du03,dp3 = Zygote.gradient(
  (u0,p)->sum(solve(prob,Tsit5(),u0=u0,p=p,callback=cb,tstops=affecttimes,abstol=1e-14,reltol=1e-14,saveat=0.1,sensealg=InterpolatingAdjoint())),
  u0,p)

du04,dp4 = Zygote.gradient(
  (u0,p)->sum(solve(prob,Tsit5(),u0=u0,p=p,callback=cb,tstops=affecttimes,abstol=1e-14,reltol=1e-14,saveat=0.1,sensealg=QuadratureAdjoint())),
  u0,p)

dstuff = ForwardDiff.gradient(
  (θ)->sum(solve(prob,Tsit5(),u0=θ[1:2],p=θ[3:6],callback=cb,tstops=affecttimes,abstol=1e-14,reltol=1e-14,saveat=0.1)),
  [u0;p])

@test du01 ≈ dstuff[1:2]
@test dp1 ≈ dstuff[3:6]
@test du01 ≈ du01a
@test du01 ≈ du02
@test_broken du01 ≈ du03
@test du01 ≈ du04
@test dp1 ≈ dp1a
@test dp1 ≈ dp2
@test_broken dp1 ≈ dp3
@test dp1 ≈ dp4

@test du02 ≈ dstuff[1:2]
@test dp2 ≈ dstuff[3:6]

### SDEs

function dt!(du, u, p, t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = α*x - β*x*y
  du[2] = dy = -δ*y + γ*x*y
end

function dW!(du, u, p, t)
  du[1] = 0.1u[1]
  du[2] = 0.1u[2]
end

u0 = [1.0,1.0]
tspan = (0.0, 10.0)
p = [2.2, 1.0, 2.0, 0.4]
prob_sde = SDEProblem(dt!, dW!, u0, tspan,p)

condition(u,t,integrator) = integrator.t >9.0 #some condition
function affect!(integrator)
	 #println("Callback")  #some callback
end
cb = DiscreteCallback(condition,affect!,save_positions=(false,false))

function predict_sde(p)
  return Array(solve(prob_sde, EM(), p=p, saveat = 0.1,sensealg = ForwardDiffSensitivity(), dt=0.001, callback=cb))
end

loss_sde(p)= sum(abs2, x-1 for x in predict_sde(p))

loss_sde(p)
@time dp = gradient(p) do p
	loss_sde(p)
end

@test !iszero(dp[1])
