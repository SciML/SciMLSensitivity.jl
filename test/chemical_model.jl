using DiffEqSensitivity, OrdinaryDiffEq, ParameterizedFunctions

f = @ode_def_nohes chem_model begin
    dy1 = -k1*y1
    dy2 = k1*y1 - k2*(y2^2)
    dy3 = k2*(y2^2)
end k1 k2

p = [1.0,1.0]
u0 = [1.0,0.0,0.0]
prob1 = ODEProblem(f,u0,(0.0,10.0),p)
prob1

A = [1,0,2,3]
function f1(p)
    x = A[1]*p[1] + A[2]*p[2]
    y = A[3]*p[1] + A[4]*p[2]
    [x,y] 
end
f1