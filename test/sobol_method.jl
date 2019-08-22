using DiffEqSensitivity, OrdinaryDiffEq, Test

A = [1,0,2,3]
function f_sobol(p)
    x = A[1]*p[1] + A[2]*p[2]
    y = A[3]*p[1] + A[4]*p[2]
    [x,y]
end
p_range = [[0.5,1.5],[0.5,1.5]]
N = 1000000
sobol = sobol_sensitivity(f_sobol,p_range,N,[0,1,2])

@test_broken [sobol.ST[1][1],sobol.ST[2][1]] ≈ [sobol.S1[1][1],sobol.S1[2][1]] .+ sobol.S2[1][1] atol=1e-1
@test_broken [sobol.ST[1][2],sobol.ST[2][2]] ≈ [sobol.S1[1][2],sobol.S1[2][2]] .+ sobol.S2[1][2] atol=1e-1