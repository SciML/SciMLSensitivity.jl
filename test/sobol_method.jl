using DiffEqSensitivity, QuasiMonteCarlo, Test

function ishi_batch(X,p=nothing)
    A= 7
    B= 0.1
    @. sin(X[1,:]) + A*sin(X[2,:])^2+ B*X[3,:]^4 *sin(X[1,:])
end
function ishi(X,p=nothing)
    A= 7
    B= 0.1
    sin(X[1]) + A*sin(X[2])^2+ B*X[3]^4 *sin(X[1])
end

n = 600000
lb = -ones(4)*π
ub = ones(4)*π
sampler = SobolSample()
A,B = QuasiMonteCarlo.generate_design_matrices(n,lb,ub,sampler)

res1 = gsa(ishi,Sobol(),A,B)
res2 = gsa(ishi_batch,Sobol(),A,B,batch=true)

@test res1.S1 ≈ [0.3139335358797363, 0.44235918402206326, 0.0, 0.0] atol=1e-4
@test res2.S1 ≈ [0.3139335358797363, 0.44235918402206326, 0.0, 0.0] atol=1e-4

@test res1.ST ≈ [0.5576009081644232, 0.44237102330046346, 0.24366241588532553, 0.0] atol=1e-4
@test res2.ST ≈ [0.5576009081644232, 0.44237102330046346, 0.24366241588532553, 0.0] atol=1e-4

#=
library(sensitivity)
ishigami.fun <- function(X) {
  A <- 7
  B <- 0.1
  sin(X[, 1]) + A * sin(X[, 2])^2 + B * X[, 3]^4 * sin(X[, 1])
}
n <- 600000
X1 <- data.frame(matrix(runif(4 * n,-pi,pi), nrow = n))
X2 <- data.frame(matrix(runif(4 * n,-pi,pi), nrow = n))
sobol2007(ishigami.fun, X1, X2)
sobolSalt(ishigami.fun, X1, X2, scheme="A")
sobolSalt(ishigami.fun, X1, X2, scheme="B")
=#
