using OrdinaryDiffEq, ParameterizedFunctions, DiffEqSensitivity

# f = @ode_def LotkaVolterra begin
#     dx = a*x - b*x*y
#     dy = -c*y + x*y
# end a b c

# p = [1.5,1.0,3.0]
# t = collect(linspace(0,10,200))
# prob = ODEProblem(f,[1.0;1.0],(0.0,10.0),p)
# m = DiffEqSensitivity.morris_sensitivity(prob,Tsit5(),t,[[1,2],[1,2],[2,4]],[10,10,10],simulations=100)
# println(m)

A = reshape([1,0,2,3],2,2)
function f(p)
    A*p
end
m = DiffEqSensitivity.morris_sensitivity(f,[[1,5],[1,5]],[10,10],k =1500,simulations=1000,r=150)
@test m.means[1] ≈ A[:,1] atol=1e-12
@test m.means[2] ≈ A[:,2] atol=1e-12
@test m.variances ≈ [[0,0],[0,0]] atol=1e-12

m = DiffEqSensitivity.morris_sensitivity(f,[[1,5],[1,5]],[10,10],relative_scale=true,k =2000,simulations=1000,r=150)
@test m.means[1][2] ≈ 0 atol=1e-12
@test m.variances[1][2] ≈ 0 atol=1e-12
@test m.means[2][1] < m.means[2][2]
