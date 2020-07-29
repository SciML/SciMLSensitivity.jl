using Test, LinearAlgebra
using DiffEqSensitivity, StochasticDiffEq
using Random

seed = 100
tspan = (0.0, 0.1)
p = [1.01,0.87]

# scalar
f(u,p,t) = p[1]*u
σ(u,p,t) = p[2]*u

Random.seed!(seed)
u0 = rand(1)
linear_analytic(u0,p,t,W) = @.(u0*exp((p[1]-p[2]^2/2)*t+p[2]*W))

prob = SDEProblem(SDEFunction(f,σ,analytic=linear_analytic),σ,u0,tspan,p)
sol = solve(prob,SOSRI(),adaptive=false, dt=0.001, save_noise=true)

@test isapprox(sol.u_analytic,sol.u, atol=1e-4)

du = zeros(size(u0))
u = sol.u[end]
transformed_function = StochasticTransformedFunction(sol,sol.prob.f,sol.prob.g)
transformed_function(du,u,p,tspan[2])
du2 = transformed_function(u,p,tspan[2])

@test du[1] == (p[1]*u[1]-p[2]^2*u[1])
@test du2[1] == (p[1]*u[1]-p[2]^2*u[1])
@test du2 == du

# inplace

f!(du,u,p,t) = @.(du = p[1]*u)
σ!(du,u,p,t) = @.(du = p[2]*u)

prob = SDEProblem(SDEFunction(f!,σ!,analytic=linear_analytic),σ!,u0,tspan,p)
sol = solve(prob,SOSRI(),adaptive=false, dt=0.001, save_noise=true)

@test isapprox(sol.u_analytic,sol.u, atol=1e-4)

du = zeros(size(u0))
u = sol.u[end]
transformed_function = StochasticTransformedFunction(sol,sol.prob.f,sol.prob.g)
transformed_function(du,u,p,tspan[2])
du2 = transformed_function(u,p,tspan[2])

@test du[1] == (p[1]*u[1]-p[2]^2*u[1])
@test du2[1] == (p[1]*u[1]-p[2]^2*u[1])
@test du2 == du

# diagonal noise

u0 = rand(3)

prob = SDEProblem(SDEFunction(f,σ,analytic=linear_analytic),σ,u0,tspan,p)
sol = solve(prob,SOSRI(),adaptive=false, dt=0.001, save_noise=true)

du = zeros(size(u0))
u = sol.u[end]
transformed_function = StochasticTransformedFunction(sol,sol.prob.f,sol.prob.g)
transformed_function(du,u,p,tspan[2])
du2 = transformed_function(u,p,tspan[2])

@test isapprox(du,(p[1]*u-p[2]^2*u), atol=1e-15)


prob = SDEProblem(SDEFunction(f!,σ!,analytic=linear_analytic),σ!,u0,tspan,p)
sol = solve(prob,SOSRI(),adaptive=false, dt=0.001, save_noise=true)

du = zeros(size(u0))
u = sol.u[end]
transformed_function = StochasticTransformedFunction(sol,sol.prob.f,sol.prob.g)
transformed_function(du,u,p,tspan[2])
du2 = transformed_function(u,p,tspan[2])

@test isapprox(du,(p[1]*u-p[2]^2*u), atol=1e-15)
@test isapprox(du2,(p[1]*u-p[2]^2*u), atol=1e-15)
@test isapprox(du,du2, atol=1e-15)

#  non-diagonal noise torus
u0 = rand(2)
p =  rand(1)

fnd(u,p,t) = 0*u
function σnd(u,p,t)
  du = [cos(p[1])*sin(u[1])  cos(p[1])*cos(u[1])   -sin(p[1])*sin(u[2])   -sin(p[1])*cos(u[2])
   sin(p[1])*sin(u[1])   sin(p[1])*cos(u[1])    cos(p[1])*sin(u[2])   cos(p[1])*cos(u[2]) ]
  return du
end

prob = SDEProblem(fnd,σnd,u0,tspan,p,noise_rate_prototype=zeros(2,4))
sol = solve(prob,EM(),adaptive=false, dt=0.001, save_noise=true)

du = zeros(size(u0))
u = sol.u[end]
transformed_function = StochasticTransformedFunction(sol,sol.prob.f,sol.prob.g)
transformed_function(du,u,p,tspan[2])
du2 = transformed_function(u,p,tspan[2])

@test isapprox(du,zeros(2), atol=1e-15)
@test isapprox(du2,zeros(2), atol=1e-15)
@test isapprox(du,du2, atol=1e-15)


fnd!(du,u,p,t) = du .= false
function σnd!(du,u,p,t)
  du[1,1] = cos(p[1])*sin(u[1])
  du[1,2] = cos(p[1])*cos(u[1])
  du[1,3] = -sin(p[1])*sin(u[2])
  du[1,4] = -sin(p[1])*cos(u[2])
  du[2,1] = sin(p[1])*sin(u[1])
  du[2,2] = sin(p[1])*cos(u[1])
  du[2,3] = cos(p[1])*sin(u[2])
  du[2,4] = cos(p[1])*cos(u[2])
  return nothing
end

prob = SDEProblem(fnd!,σnd!,u0,tspan,p,noise_rate_prototype=zeros(2,4))
sol = solve(prob,EM(),adaptive=false, dt=0.001, save_noise=true)

du = zeros(size(u0))
u = sol.u[end]
transformed_function = StochasticTransformedFunction(sol,sol.prob.f,sol.prob.g)
transformed_function(du,u,p,tspan[2])
du2 = transformed_function(u,p,tspan[2])

@test isapprox(du,zeros(2), atol=1e-15)
@test isapprox(du2,zeros(2), atol=1e-15)
@test isapprox(du,du2, atol=1e-15)
