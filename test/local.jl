using DiffEqSensitivity,OrdinaryDiffEq, ParameterizedFunctions,
      RecursiveArrayTools, DiffEqBase, ForwardDiff, Calculus
using Test
using DiffEqSensitivity: SensitivityAlg


f = @ode_def LotkaVolterra begin
  dx = a*x - b*x*y
  dy = -c*y + x*y
end a b c

function LotkaVolt!(du, u, p, t)
  du[1] = p[1]*u[1] - p[2]*u[1]*u[2]
  du[2] = -p[3]*u[2] + u[1]*u[2]
end

p = [1.5,1.0,3.0]
prob = ODELocalSensitivityProblem(f,[1.0;1.0],(0.0,10.0),p)
probInpl = ODELocalSensitivityProblem(LotkaVolt!,[1.0;1.0],(0.0,10.0),p)
probnoad = ODELocalSensitivityProblem(LotkaVolt!,[1.0;1.0],(0.0,10.0),p,
                                      SensitivityAlg(autodiff=false))
sol = solve(prob,Tsit5(),abstol=1e-14,reltol=1e-14)
@test_broken solInpl = solve(probInpl,KenCarp4(),abstol=1e-14,reltol=1e-14)
solInpl2 = solve(probInpl,Rodas4(autodiff=false),abstol=1e-14,reltol=1e-14)
solInpl = solve(probInpl,KenCarp4(autodiff=false),abstol=1e-14,reltol=1e-14)
solInpl2 = solve(probInpl,Rodas4(autodiff=false),abstol=1e-14,reltol=1e-14)
solnoad = solve(probnoad,KenCarp4(autodiff=false),abstol=1e-14,reltol=1e-14)

x = sol[1:sol.prob.f.numindvar,:]

@test sol(5.0) ≈ solnoad(5.0)
@test sol(5.0) ≈ solInpl(5.0)
@test solInpl(5.0) ≈ solInpl2(5.0)

# Get the sensitivities

da = sol[sol.prob.f.numindvar+1:sol.prob.f.numindvar*2,:]
db = sol[sol.prob.f.numindvar*2+1:sol.prob.f.numindvar*3,:]
dc = sol[sol.prob.f.numindvar*3+1:sol.prob.f.numindvar*4,:]

sense_res1 = [da[:,end] db[:,end] dc[:,end]]

prob = ODELocalSensitivityProblem(f.f,[1.0;1.0],(0.0,10.0),p,SensitivityAlg(autojacvec=true))
sol = solve(prob,Tsit5(),abstol=1e-14,reltol=1e-14)
x = sol[1:sol.prob.f.numindvar,:]

# Get the sensitivities

da = sol[sol.prob.f.numindvar+1:sol.prob.f.numindvar*2,:]
db = sol[sol.prob.f.numindvar*2+1:sol.prob.f.numindvar*3,:]
dc = sol[sol.prob.f.numindvar*3+1:sol.prob.f.numindvar*4,:]

sense_res2 = [da[:,end] db[:,end] dc[:,end]]

function test_f(p)
  prob = ODEProblem(f,eltype(p).([1.0,1.0]),(0.0,10.0),p)
  solve(prob,Tsit5(),abstol=1e-14,reltol=1e-14,save_everystep=false)[end]
end

p = [1.5,1.0,3.0]
fd_res = ForwardDiff.jacobian(test_f,p)
calc_res = Calculus.finite_difference_jacobian(test_f,p)

@test sense_res1 ≈ sense_res2 ≈ fd_res
@test sense_res1 ≈ sense_res2 ≈ calc_res

################################################################################

# Now do from a plain parameterized function

function f2(du,u,p,t)
  du[1] = p[1] * u[1] - p[2] * u[1]*u[2]
  du[2] = -p[3] * u[2] + u[1]*u[2]
end
p = [1.5,1.0,3.0]
prob = ODELocalSensitivityProblem(f2,[1.0;1.0],(0.0,10.0),p)
sol = solve(prob,Tsit5(),abstol=1e-14,reltol=1e-14)
res = sol[1:sol.prob.f.numindvar,:]

# Get the sensitivities

da = sol[sol.prob.f.numindvar+1:sol.prob.f.numindvar*2,:]
db = sol[sol.prob.f.numindvar*2+1:sol.prob.f.numindvar*3,:]
dc = sol[sol.prob.f.numindvar*3+1:sol.prob.f.numindvar*4,:]

sense_res = [da[:,end] db[:,end] dc[:,end]]

p = [1.5,1.0,3.0]
fd_res = ForwardDiff.jacobian(test_f,p)
calc_res = Calculus.finite_difference_jacobian(test_f,p)

@test sense_res ≈ fd_res
@test sense_res ≈ calc_res


## Check extraction

x,dp = extract_local_sensitivities(sol)
x == res
dp[1] == da

x,dp = extract_local_sensitivities(sol,length(sol.t))
sense_res2 = hcat(dp...)
sense_res == sense_res2

extract_local_sensitivities(sol,sol.t[3]) == extract_local_sensitivities(sol,3)

tmp = similar(sol[1])
extract_local_sensitivities(tmp,sol,sol.t[3]) == extract_local_sensitivities(sol,3)
