using DiffEqSensitivity,OrdinaryDiffEq, ParameterizedFunctions, RecursiveArrayTools, DiffEqBase
using Base.Test

f = @ode_def_nohes LotkaVolterra begin
  dx = a*x - b*x*y
  dy = -c*y + d*x*y
end a=>1.5 b=>1 c=>3 d=1

prob = ODELocalSensitivityProblem(f,[1.0;1.0],(0.0,30.0))
sol = solve(prob,DP8())
x = sol[1:sol.prob.indvars,:]

prob2 = ODEProblem(f,[1.0;1.0],(0.0,30.0))
sol2 = solve(prob2,DP8(),adaptive=true,abstol=1,tstops=sol.t)
x2 = convert(Array,sol2)

@test maximum(x2-x) < 1e-12 # Solve the same problem

# Get the sensitivities

da = sol[sol.prob.indvars+1:sol.prob.indvars*2,:]
db = sol[sol.prob.indvars*2+1:sol.prob.indvars*3,:]
dc = sol[sol.prob.indvars*3+1:sol.prob.indvars*4,:]

@test (abs.(da[:,end]) .> abs.(da[:,length(sol)÷2])) == [false;true]
@test (abs.(db[:,end]) .> abs.(db[:,length(sol)÷2])) == [true;true]
@test (abs.(dc[:,end]) .> abs.(dc[:,length(sol)÷2])) == [false;true]
#using Plots
#gr()
#plot(sol.t,x')
#plot(sol.t,da')
#plot(sol.t,db')
#plot(sol.t,dc')
