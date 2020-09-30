using DiffEqSensitivity, QuasiMonteCarlo, Test, OrdinaryDiffEq

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

res1 = gsa(linear,Sobol(),A,B)
res2 = gsa(linear_batch,Sobol(),A,B,batch=true)

@test res1.S1 ≈ [0.9997953478183109, 0.0002040399766938839, 0.0, 0.0] atol=1e-4
@test res2.S1 ≈ [0.9997953478183109, 0.0002040399766938839, 0.0, 0.0] atol=1e-4

@test res1.ST ≈ [0.9997953478183109, 0.0002040399766938839, 0.0, 0.0] atol=1e-4
@test res2.ST ≈ [0.9997953478183109, 0.0002040399766938839, 0.0, 0.0] atol=1e-4

@test_nowarn gsa(linear,Sobol(order=[0,1,2], nboot = 100),A,B)

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

library(sensitivity)
ishigami.fun <- function(X) {
  A <- 7
  B <- 0.1
  A * X[, 1] + B * X[, 2]
}
n <- 6000000
X1 <- data.frame(matrix(runif(4 * n,-pi,pi), nrow = n))
X2 <- data.frame(matrix(runif(4 * n,-pi,pi), nrow = n))
sobol2007(ishigami.fun, X1, X2)
sobolSalt(ishigami.fun, X1, X2, scheme="A")
sobolSalt(ishigami.fun, X1, X2, scheme="B")
=#

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

res1 = gsa(ishi_linear,Sobol(),A,B)
res2 = gsa(ishi_linear_batch,Sobol(),A,B,batch=true)

# Now both tests together

@test res1.S1[1,:] ≈ [0.3139335358797363, 0.44235918402206326, 0.0, 0.0] atol=1e-4
@test res2.S1[1,:] ≈ [0.3139335358797363, 0.44235918402206326, 0.0, 0.0] atol=1e-4

@test res1.ST[1,:] ≈ [0.5576009081644232, 0.44237102330046346, 0.24366241588532553, 0.0] atol=1e-4
@test res2.ST[1,:] ≈ [0.5576009081644232, 0.44237102330046346, 0.24366241588532553, 0.0] atol=1e-4

@test res1.S1[2,:] ≈ [0.9997953478183109, 0.0002040399766938839, 0.0, 0.0] atol=1e-4
@test res2.S1[2,:] ≈ [0.9997953478183109, 0.0002040399766938839, 0.0, 0.0] atol=1e-4

@test res1.ST[2,:] ≈ [0.9997953478183109, 0.0002040399766938839, 0.0, 0.0] atol=1e-4
@test res2.ST[2,:] ≈ [0.9997953478183109, 0.0002040399766938839, 0.0, 0.0] atol=1e-4

function f(du,u,p,t)
  du[1] = p[1]*u[1] - p[2]*u[1]*u[2] #prey
  du[2] = -p[3]*u[2] + p[4]*u[1]*u[2] #predator
end

u0 = [1.0;1.0]
tspan = (0.0,10.0)
p = [1.5,1.0,3.0,1.0]
prob = ODEProblem(f,u0,tspan,p)
t = collect(range(0, stop=10, length=200))


f1 = let prob = prob, t = t
    function (p)
        prob1 = remake(prob;p=p)
        sol = solve(prob1,Tsit5();saveat=t)
        return sol
    end
end

m = gsa(f1,Sobol(),[[1,5],[1,5],[1,5],[1,5]],N=100)
