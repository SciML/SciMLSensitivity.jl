using DiffEqSensitivity,OrdinaryDiffEq, ParameterizedFunctions, RecursiveArrayTools, DiffEqBase
using Base.Test

f = @ode_def_nohes LotkaVolterra begin
  dx = a*x - b*x*y
  dy = -c*y + d*x*y
end a=>1.5 b=>1 c=>3 d=1

prob = ODELocalSensitivityProblem(f,[1.0;1.0],(0.0,30.0))
sol = solve(prob,DP8())
x = vecarr_to_arr([sol[i][1:sol.prob.indvars] for i in 1:length(sol)])

prob2 = ODEProblem(f,[1.0;1.0],(0.0,30.0))
sol2 = solve(prob2,DP8(),adaptive=true,abstol=1,tstops=sol.t)
x2 = vecarr_to_arr(sol2[:])

@test maximum(x2-x) < 1e-12 # Solve the same problem

# Get the sensitivities

da=[sol[i][sol.prob.indvars+1:sol.prob.indvars*2] for i in 1:length(sol)]
db=[sol[i][sol.prob.indvars*2+1:sol.prob.indvars*3] for i in 1:length(sol)]
dc=[sol[i][sol.prob.indvars*3+1:sol.prob.indvars*4] for i in 1:length(sol)]

@test (abs.(da[end]) .> abs.(da[length(sol)÷2])) == [false;true]
@test (abs.(db[end]) .> abs.(db[length(sol)÷2])) == [true;true]
@test (abs.(dc[end]) .> abs.(dc[length(sol)÷2])) == [false;true]
#using Plots
#gr()
#plot(sol.t,vecarr_to_arr(x))
#plot(sol.t,vecarr_to_arr(da))
#plot(sol.t,vecarr_to_arr(db))
#plot(sol.t,vecarr_to_arr(dc))
