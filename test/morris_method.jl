using OrdinaryDiffEq, ParameterizedFunctions, DiffEqSensitivity

A = reshape([1,0,2,3],2,2)
function f_morris(p)
    A*p
end
m = DiffEqSensitivity.morris_sensitivity(f_morris,[[1,5],[1,5]],[10,10],len_trajectory=1500,total_num_trajectory=1000,num_trajectory=150)
@test m.means[1] ≈ A[:,1] atol=1e-12
@test m.means[2] ≈ A[:,2] atol=1e-12
@test m.variances ≈ [[0,0],[0,0]] atol=1e-12

m = DiffEqSensitivity.morris_sensitivity(f_morris,[[1,5],[1,5]],[10,10],relative_scale=true,len_trajectory=2000,total_num_trajectory=1000,num_trajectory=150)
@test m.means[1][2] ≈ 0 atol=1e-12
@test m.variances[1][2] ≈ 0 atol=1e-12
@test m.means[2][1] < m.means[2][2]
