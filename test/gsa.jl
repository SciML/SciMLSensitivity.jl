using DiffEqSensitivity, Test, QuasiMonteCarlo


function ishi_batch(X)
    A= 7
    B= 0.1
    @. sin(X[1,:]) + A*sin(X[2,:])^2+ B*X[3,:]^4 *sin(X[1,:])
end
function ishi(X)
    A= 7
    B= 0.1
    sin(X[1]) + A*sin(X[2])^2+ B*X[3]^4 *sin(X[1])
end

function linear_batch(X)
    A= 7
    B= 0.1
    @. A*X[1,:]+B*X[2,:]
end
function linear(X)
    A= 7
    B= 0.1
    A*X[1]+B*X[2]
end

n = 600000
lb = -ones(4)*π
ub = ones(4)*π
sampler = SobolSample()
A,B = QuasiMonteCarlo.generate_design_matrices(n,lb,ub,sampler)

res1 = gsa(ishi,Sobol(order=[0,1,2]),A,B)
res2 = gsa(ishi_batch,Sobol(),A,B,batch=true)

@test res1.S1 ≈ [0.3139335358797363, 0.44235918402206326, 0.0, 0.0] atol=1e-4
@test res2.S1 ≈ [0.3139335358797363, 0.44235918402206326, 0.0, 0.0] atol=1e-4

@test res1.ST ≈ [0.5576009081644232, 0.44237102330046346, 0.24366241588532553, 0.0] atol=1e-4
@test res2.ST ≈ [0.5576009081644232, 0.44237102330046346, 0.24366241588532553, 0.0] atol=1e-4


res1 = gsa(ishi,eFAST(),[[lb[i],ub[i]] for i in 1:4],n=15000)
res2 = gsa(ishi_batch,eFAST(),[[lb[i],ub[i]] for i in 1:4],n=15000,batch=true)

@test res1.S1 ≈ [0.307599  0.442412  3.0941e-25  3.42372e-28] atol=1e-4
@test res2.S1 ≈ [0.307599  0.442412  3.0941e-25  3.42372e-28] atol=1e-4

@test res1.ST ≈ [0.556244  0.446861  0.239259  0.027099] atol=1e-4
@test res2.ST ≈ [0.556244  0.446861  0.239259  0.027099] atol=1e-4

m = gsa(ishi, Morris(num_trajectory=500000), [[lb[i],ub[i]] for i in 1:4])
@test m.means_star[1,:] ≈ [2.25341,4.40246,2.5049,0.0] atol = 5e-2
@test m.means[1, :] ≈ [-0.416876, -0.0077712, -0.015714,  0.0] atol = 5e-2
@test m.means_star[1,:] ≈ [2.25341,4.40246,2.5049,0.0] atol = 5e-2
