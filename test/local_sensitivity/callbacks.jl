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

condition(u,t,integrator) = t == 5
affect!(integrator) = integrator.u[1] += 2.0
cb = DiscreteCallback(condition,affect!)
du01,dp1 = Zygote.gradient((u0,p)->sum(solve(prob,Tsit5(),u0=u0,p=p,callback=cb,tstops=[5.0],abstol=1e-14,reltol=1e-14,saveat=0.1,sensealg=BacksolveAdjoint())),u0,p)
du02,dp2 = Zygote.gradient((u0,p)->sum(solve(prob,Tsit5(),u0=u0,p=p,callback=cb,tstops=[5.0],abstol=1e-14,reltol=1e-14,saveat=0.1,sensealg=ReverseDiffAdjoint())),u0,p)
dstuff = ForwardDiff.gradient((θ)->sum(solve(prob,Tsit5(),u0=θ[1:2],p=θ[3:6],callback=cb,tstops=[5.0],abstol=1e-14,reltol=1e-14,saveat=0.1)),[u0;p])
@test du01 ≈ dstuff[1:2]
@test dp1 ≈ dstuff[3:6]
@test du01 ≈ du02
@test dp1 ≈ dp2

condition(u,t,integrator) = t == 5
affect!(integrator) = (integrator.u[1] = 2.0; @show "triggered!")
cb = DiscreteCallback(condition,affect!)
du01,dp1 = Zygote.gradient((u0,p)->sum(solve(prob,Tsit5(),u0=u0,p=p,callback=cb,tstops=[5.0],abstol=1e-14,reltol=1e-14,saveat=0.1,sensealg=BacksolveAdjoint())),u0,p
)
du02,dp2 = Zygote.gradient((u0,p)->sum(solve(prob,Tsit5(),u0=u0,p=p,callback=cb,tstops=[5.0],abstol=1e-14,reltol=1e-14,saveat=0.1,sensealg=ReverseDiffAdjoint())),u0,p
)
dstuff = ForwardDiff.gradient((θ)->sum(solve(prob,Tsit5(),u0=θ[1:2],p=θ[3:6],callback=cb,tstops=[5.0],abstol=1e-14,reltol=1e-14,saveat=0.1)),[u0;p])
@test du01 ≈ dstuff[1:2]
@test dp1 ≈ dstuff[3:6]
@test du01 ≈ du02
@test dp1 ≈ dp2

affecttimes = [2.0,4.0,8.0]
condition(u,t,integrator) = t ∈ affecttimes
affect!(integrator) = integrator.u[1] += 2.0
cb = DiscreteCallback(condition,affect!)
du01,dp1 = Zygote.gradient((u0,p)->sum(solve(prob,Tsit5(),u0=u0,p=p,callback=cb,tstops=affecttimes,abstol=1e-14,reltol=1e-14,saveat=0.1,sensealg=BacksolveAdjoint())),u0,p)
du02,dp2 = Zygote.gradient((u0,p)->sum(solve(prob,Tsit5(),u0=u0,p=p,callback=cb,tstops=affecttimes,abstol=1e-14,reltol=1e-14,saveat=0.1,sensealg=ReverseDiffAdjoint())),u0,p)
dstuff = ForwardDiff.gradient((θ)->sum(solve(prob,Tsit5(),u0=θ[1:2],p=θ[3:6],callback=cb,tstops=affecttimes,abstol=1e-14,reltol=1e-14,saveat=0.1)),[u0;p])
@test du01 ≈ dstuff[1:2]
@test dp1 ≈ dstuff[3:6]
@test du01 ≈ du02
@test dp1 ≈ dp2

# SDEs

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
