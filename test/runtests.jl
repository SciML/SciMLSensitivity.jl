using DiffEqSensitivity,OrdinaryDiffEq, ParameterizedFunctions,
      RecursiveArrayTools, DiffEqBase, ForwardDiff, Calculus
using Base.Test

f = @ode_def_nohes LotkaVolterra begin
  dx = a*x - b*x*y
  dy = -c*y + d*x*y
end a=>1.5 b=>1.0 c=>3.0 d=1

prob = ODELocalSensitivityProblem(f,[1.0;1.0],(0.0,10.0))
sol = solve(prob,Vern9(),abstol=1e-14,reltol=1e-14)
x = sol[1:sol.prob.indvars,:]

# Get the sensitivities

da = sol[sol.prob.indvars+1:sol.prob.indvars*2,:]
db = sol[sol.prob.indvars*2+1:sol.prob.indvars*3,:]
dc = sol[sol.prob.indvars*3+1:sol.prob.indvars*4,:]

sense_res = [da[:,end] db[:,end] dc[:,end]]

function test_f(p)
  pf = ParameterizedFunction(f,p)
  prob = ODEProblem(pf,eltype(p).([1.0,1.0]),eltype(p).((0.0,10.0)))
  solve(prob,Vern9(),abstol=1e-14,reltol=1e-14,save_everystep=false)[end]
end

p = [1.5,1.0,3.0]
fd_res = ForwardDiff.jacobian(test_f,p)
calc_res = Calculus.finite_difference_jacobian(test_f,p)

@test sense_res ≈ fd_res
@test sense_res ≈ calc_res

################################################################################

# Now do from a plain parameterized function

function f2(t,u,p,du)
  du[1] = p[1] * u[1] - p[2] * u[1]*u[2]
  du[2] = -p[3] * u[2] + u[1]*u[2]
end
pf = ParameterizedFunction(f2,[1.5,1.0,3.0])
prob = ODELocalSensitivityProblem(pf,[1.0;1.0],(0.0,10.0))
sol = solve(prob,Vern9(),abstol=1e-14,reltol=1e-14)
x = sol[1:sol.prob.indvars,:]

# Get the sensitivities

da = sol[sol.prob.indvars+1:sol.prob.indvars*2,:]
db = sol[sol.prob.indvars*2+1:sol.prob.indvars*3,:]
dc = sol[sol.prob.indvars*3+1:sol.prob.indvars*4,:]

sense_res = [da[:,end] db[:,end] dc[:,end]]

p = [1.5,1.0,3.0]
fd_res = ForwardDiff.jacobian(test_f,p)
calc_res = Calculus.finite_difference_jacobian(test_f,p)

@test sense_res ≈ fd_res
@test sense_res ≈ calc_res
