using OrdinaryDiffEq,ParameterizedFunctions

f = @ode_def LotkaVolterra begin
    dx = a*x - b*x*y
    dy = -c*y + x*y
end a b c
  
p = [1.5,1.0,3.0]
prob = ODEProblem(f,[1.0;1.0],(0.0,10.0),p)
m = sample_matrices(prob,[[0,10],[0,10],[0,10]],[10,10,10],simulations=100)
println(m)