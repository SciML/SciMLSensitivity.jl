using DiffEqSensitivity, QuasiMonteCarlo, Test

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

lb = -ones(4)*π
ub = ones(4)*π

res1 = gsa(ishi,eFAST(),[[lb[i],ub[i]] for i in 1:4],N=15000)
res2 = gsa(ishi_batch,eFAST(),[[lb[i],ub[i]] for i in 1:4],N=15000,batch=true)

@test res1.first_order ≈ [0.307599  0.442412  3.0941e-25  3.42372e-28] atol=1e-4
@test res2.first_order ≈ [0.307599  0.442412  3.0941e-25  3.42372e-28] atol=1e-4

@test res1.total_order ≈ [0.556244  0.446861  0.239259  0.027099] atol=1e-4
@test res2.total_order ≈ [0.556244  0.446861  0.239259  0.027099] atol=1e-4

res1 = gsa(linear,eFAST(),[[lb[i],ub[i]] for i in 1:4],N=1000)
res2 = gsa(linear_batch,eFAST(),[[lb[i],ub[i]] for i in 1:4],N=100,batch=true)

@test res1.first_order ≈ [0.997504  0.000203575  2.1599e-10  2.18296e-10] atol=1e-4
@test res2.first_order ≈ [0.997504  0.000203575  2.1599e-10  2.18296e-10] atol=1e-4

@test res1.total_order ≈ [0.999796  0.000204698  7.26874e-7  7.59996e-7] atol=1e-4
@test res2.total_order ≈ [0.999796  0.000204698  7.26874e-7  7.59996e-7] atol=1e-4

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

res1 = gsa(ishi_linear,eFAST(),[[lb[i],ub[i]] for i in 1:4],N=15000)
res2 = gsa(ishi_linear_batch,eFAST(),[[lb[i],ub[i]] for i in 1:4],N=15000,batch=true)

# Now both tests together

@test res1.first_order ≈ [0.307595  0.442411     7.75353e-26  2.3468e-28 
                        0.997498  0.000203571  3.18996e-35  4.19822e-35] atol=1e-4
@test res2.first_order ≈ [0.307598  0.442411     5.27085e-26  3.50751e-29
                        0.997498  0.000203571  1.08441e-34  9.90366e-35] atol=1e-4

@test res1.total_order ≈ [ 0.556246  0.446861    0.239258    0.027104  
                            0.999796  0.00020404  6.36917e-8  6.34754e-8] atol=1e-4
@test res2.total_order ≈ [0.556243  0.446861    0.239258    0.0271024 
                            0.999796  0.00020404  6.35579e-8  6.36016e-8] atol=1e-4
