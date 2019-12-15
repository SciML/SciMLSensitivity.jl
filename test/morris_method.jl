using DiffEqSensitivity, Test

A = reshape([1,0,2,3],2,2)
function f_morris(p)
    A*p
end

function linear_batch(X)
    A= 7
    B= 0.1
    @. A*X[1,:]+B*X[2,:]
end

m = gsa(f_morris,Morris(p_steps=[10,10],total_num_trajectory=1000,num_trajectory=150),[[1,5],[1,5]])
@test m.means[:,1] ≈ A[:,1] atol=1e-12
@test m.means[:,2] ≈ A[:,2] atol=1e-12
@test m.variances ≈ reshape([0,0,0,0],2,2) atol=1e-12

m = gsa(f_morris,Morris(p_steps=[10,10],relative_scale=true,total_num_trajectory=1000,num_trajectory=150),[[1,5],[1,5]])
@test m.means[2,1] ≈ 0 atol=1e-12
@test m.variances[2,1] ≈ 0 atol=1e-12
@test m.means[1,2] < m.means[2,2]

m = gsa(linear_batch,Morris(p_steps=[10,10],relative_scale=true,num_trajectory=10000),[[1,5],[1,5]],batch=true)
@test m.means ≈ [0.439769  0.00579919] atol = 1e-2