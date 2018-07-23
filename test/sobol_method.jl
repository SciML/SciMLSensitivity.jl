using DiffEqSensitivity, OrdinaryDiffEq, ParameterizedFunctions, Test

A = [1,0,2,3]
function f_sobol(p)
    x = A[1]*p[1] + A[2]*p[2]
    y = A[3]*p[1] + A[4]*p[2]
    [x,y]
end
p_range = [[0.5,1.5],[0.5,1.5]]
N = 1000000
total = sobol_sensitivity(f_sobol,p_range,N,0)
first_order = sobol_sensitivity(f_sobol,p_range,N,1)
second_order = sobol_sensitivity(f_sobol,p_range,N,2)

@test [total[1][1],total[2][1]] ≈ [first_order[1][1],first_order[2][1]] .+ second_order[1][1] atol=1e-1
@test [total[1][2],total[2][2]] ≈ [first_order[1][2],first_order[2][2]] .+ second_order[1][2] atol=1e-1
