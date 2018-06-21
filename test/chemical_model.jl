using DiffEqSensitivity, OrdinaryDiffEq, ParameterizedFunctions

# f = @ode_def_nohes chem_model begin
#     dy1 = -k1*y1
#     dy2 = k1*y1 - k2*(y2^2)
#     dy3 = k2*(y2^2)
# end k1 k2

# p = [1.0,1.0]
# u0 = [1.0,0.0,0.0]
# prob1 = ODEProblem(f,u0,(0.0,10.0),p)
# prob1

A = [1,0,2,3]
function f1(p)
    x = A[1]*p[1] + A[2]*p[2]
    y = A[3]*p[1] + A[4]*p[2]
    [x,y] 
end
p_range = [[0.5,1.5],[0.5,1.5]]
N = 100000
total = sobol_sensitivity(f1,p_range,N)
first_order = sobol_sensitivity(f1,p_range,N,1)
second_order = sobol_sensitivity(f1,p_range,N,2)

@test [total[1][1],total[2][1]] ≈ [first_order[1][1],first_order[2][1]] + second_order[1][1] atol=1e-1
@test [total[1][2],total[2][2]] ≈ [first_order[1][2],first_order[2][2]] + second_order[2][2] atol=1e-1

# fitz = @ode_def_nohes FitzhughNagumo begin
#     dV = c*(V - V^3/3 + R)
#     dR = -(1/c)*(V -  a - b*R)
# end a b c
# u0 = [-1.0;1.0]
# tspan = (0.0,20.0)
# p = [0.2,0.2,3.0]
# prob2 = ODEProblem(fitz,u0,tspan,p)
# prob2