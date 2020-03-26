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

function neg_linear_batch(X)
    A= -7
    B= 0.1
    @. A*X[1,:]+B*X[2,:]
end

function ishi(X)
    A= 7
    B= 0.1
    sin(X[1]) + A*sin(X[2])^2+ B*X[3]^4 *sin(X[1])
end

function ishi_linear(X)
    A= 7
    B= 0.1
    [sin(X[1]) + A*sin(X[2])^2+ B*X[3]^4 *sin(X[1]),A*X[1]+B*X[2]]
end

function ishi_linear_batch(X)
    A= 7
    B= 0.1
    X1 = @. sin(X[1,:]) + A*sin(X[2,:])^2+ B*X[3,:]^4 *sin(X[1,:])
    X2 = @. A*X[1,:]+B*X[2,:]
    vcat(X1',X2')
end

lb = -ones(4)*π
ub = ones(4)*π

m = gsa(f_morris,Morris(p_steps=[10,10],total_num_trajectory=1000,num_trajectory=150),[[1,5],[1,5]])
@test m.means[:,1] ≈ A[:,1] atol=1e-12
@test m.means[:,2] ≈ A[:,2] atol=1e-12
@test m.means_star[:,1] ≈ A[:,1] atol=1e-12
@test m.means_star[:,2] ≈ A[:,2] atol=1e-12
@test m.variances ≈ reshape([0,0,0,0],2,2) atol=1e-12

m = gsa(f_morris,Morris(p_steps=[10,10],relative_scale=true,total_num_trajectory=1000,num_trajectory=150),[[1,5],[1,5]])
@test m.means[2,1] ≈ 0 atol=1e-12
@test m.means_star[2,1] ≈ 0 atol=1e-12
@test m.variances[2,1] ≈ 0 atol=1e-12
@test m.means[1,2] < m.means[2,2]
@test m.means_star[1,2] < m.means_star[2,2]

m = gsa(linear_batch,Morris(p_steps=[10,10],num_trajectory=50000),[[1,5],[1,5]],batch=true)
@test m.means ≈ [7.0  0.1] atol = 1e-2
@test m.means_star ≈ [7.0  0.1] atol = 1e-2
@test m.variances ≈ reshape([0,0], 1, 2) atol=1e-12

m = gsa(neg_linear_batch,Morris(p_steps=[10,10],num_trajectory=50000),[[1,5],[1,5]],batch=true)
@test m.means ≈ [-7.0  0.1] atol = 1e-2
@test m.means_star ≈ [7.0  0.1] atol = 1e-2
@test m.variances ≈ reshape([0,0], 1, 2) atol=1e-12

m = gsa(ishi, Morris(num_trajectory=500000), [[lb[i],ub[i]] for i in 1:4])
@test m.means_star[1,:] ≈ [2.25341,4.40246,2.5049,0.0] atol = 5e-2
@test m.means[1, :] ≈ [-0.416876, -0.0077712, -0.015714,  0.0] atol = 5e-2
@test m.means_star[1,:] ≈ [2.25341,4.40246,2.5049,0.0] atol = 5e-2

m = gsa(ishi_linear,Morris(num_trajectory=500000),[[lb[i],ub[i]] for i in 1:4])
@test m.means_star[1,:] ≈ [2.25341,4.40246,2.5049,0.0] atol = 5e-2
@test m.means_star[2,:] ≈ [7.0,0.1,0.0,0.0] rtol = 1e-12

m = gsa(ishi_linear_batch,Morris(num_trajectory=500000),[[lb[i],ub[i]] for i in 1:4],batch=true)
@test m.means_star[1,:] ≈ [2.25341,4.40246,2.5049,0.0] atol = 5e-2
@test m.means_star[2,:] ≈ [7.0,0.1,0.0,0.0] rtol = 1e-12

function f(du,u,p,t)
  du[1] = p[1]*u[1] - p[2]*u[1]*u[2] #prey
  du[2] = -p[3]*u[2] + p[4]*u[1]*u[2] #predator
end

u0 = [1.0;1.0]
tspan = (0.0,10.0)
p = [1.5,1.0,3.0,1.0]
prob = ODEProblem(f,u0,tspan,p)
t = collect(range(0, stop=10, length=200))

f1 = function (p)
  prob1 = remake(prob;p=p)
  sol = solve(prob1,Tsit5();saveat=t)
  return sol
end

m = gsa(f1,Morris(total_num_trajectory=1000,num_trajectory=150),[[1,5],[1,5],[1,5],[1,5]])