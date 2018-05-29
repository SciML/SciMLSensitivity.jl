using OrdinaryDiffEq, ParameterizedFunctions, DiffEqSensitivity

f = @ode_def LotkaVolterra begin
    dx = a*x - b*x*y
    dy = -c*y + x*y
end a b c

p = [1.5,1.0,3.0]
t = collect(linspace(0,10,200))
prob = ODEProblem(f,[1.0;1.0],(0.0,10.0),p)
m = DiffEqSensitivity.morris_sensitivity(prob,Tsit5(),t,[[1,2],[1,2],[2,4]],[10,10,10],simulations=100)
println(m)
