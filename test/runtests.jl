using DiffEqSensitivity,OrdinaryDiffEq, ParameterizedFunctions, RecursiveArrayTools
using Base.Test

f = @ode_def_nohes LotkaVolterra begin
  dx = a*x - b*x*y
  dy = -c*y + d*x*y
end a=>1.5 b=>1 c=>3 d=1

prob = ODELocalSensitivityProblem(f,[1.0;1.0])
sol = solve(prob,[0;30],alg=:DP8)
x = vecvec_to_mat([sol[i][1:sol.prob.numvars] for i in 1:length(sol)])

prob2 = ODEProblem(f,[1.0;1.0])
sol2 = solve(prob2,sol.t,alg=:DP8,adaptive=true,abstol=1)
x2 = vecvec_to_mat(sol2[:])

@test maximum(x2-x) < 1e-12 # Solve the same problem

# Get the sensitivities

da=[sol[i][sol.prob.numvars+1:sol.prob.numvars*2] for i in 1:length(sol)]
db=[sol[i][sol.prob.numvars*2+1:sol.prob.numvars*3] for i in 1:length(sol)]
dc=[sol[i][sol.prob.numvars*3+1:sol.prob.numvars*4] for i in 1:length(sol)]

@test (abs(da[end]) .> abs(da[length(sol)÷2])) == [false;true]
@test (abs(db[end]) .> abs(db[length(sol)÷2])) == [true;true]
@test (abs(dc[end]) .> abs(dc[length(sol)÷2])) == [false;true]
#using Plots
#gr()
#plot(sol.t,vecvec_to_mat(x))
#plot(sol.t,vecvec_to_mat(da))
#plot(sol.t,vecvec_to_mat(db))
#plot(sol.t,vecvec_to_mat(dc))
