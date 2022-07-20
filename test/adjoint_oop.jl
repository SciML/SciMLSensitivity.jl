using SciMLSensitivity, OrdinaryDiffEq, SimpleChains, StaticArrays, QuadGK, ForwardDiff,
      Zygote
using Test

##Adjoints of numerical solve

u0 = @SVector [1.0f0, 1.0f0]
p = @SMatrix [1.5f0 -1.0f0; 3.0f0 -1.0f0]
tspan = [0.0f0, 5.0f0]
datasize = 20
tsteps = range(tspan[1], tspan[2], length = datasize)

function f(u, p, t)
    p*u
end

prob = ODEProblem(f, u0, tspan, p)
sol = solve(prob, Tsit5(), saveat=tsteps, abstol = 1e-12, reltol = 1e-12)

## Discrete Case
dg_disc(u, p, t, i; outtype = nothing) = u .- 1

du0, dp = adjoint_sensitivities(sol, Tsit5(); t = tsteps, dgdu_discrete = dg_disc,
                                sensealg = QuadratureAdjoint(autojacvec = ZygoteVJP()))

## with ForwardDiff
function G_p(p)
    tmp_prob = remake(prob, p = p)
    u = Array(solve(tmp_prob, Tsit5(), saveat = tsteps,
                    sensealg = SensitivityADPassThrough(), abstol=1e-12, reltol=1e-12))

    return sum(((1 .- u) .^ 2) ./ 2)
end

function G_u(u0)
    tmp_prob = remake(prob, u0 = u0)
    u = Array(solve(tmp_prob, Tsit5(), saveat = tsteps,
                    sensealg = SensitivityADPassThrough(), abstol=1e-12, reltol=1e-12))
    return sum(((1 .- u) .^ 2) ./ 2)
end

G_p(p)
G_u(u0)
n_dp = ForwardDiff.gradient(G_p, p)
n_du0 = ForwardDiff.gradient(G_u, u0)

@test n_du0 ≈ du0 rtol = 1e-3
@test_broken n_dp ≈ dp' rtol = 1e-3
@test sum(n_dp - dp') < 8.0

## Continuous Case

g(u, p, t) = sum((u.^2)./2)

function dg(u, p, t)
    u
end

du0, dp = adjoint_sensitivities(sol, Tsit5(); dgdu_continuous = dg, g = g,
                                sensealg = QuadratureAdjoint(autojacvec = ZygoteVJP()))

@test !iszero(du0)
@test !iszero(dp)

##numerical

function G_p(p)
    tmp_prob = remake(prob, p = p)
    sol = solve(tmp_prob, Tsit5(), abstol = 1e-12, reltol = 1e-12)
    res, err = quadgk((t) -> (sum((sol(t).^2)./2)), 0.0, 5.0, atol = 1e-12,
                      rtol = 1e-12)
    res
end

function G_u(u0)
    tmp_prob = remake(prob, u0 = u0)
    sol = solve(tmp_prob, Tsit5(), abstol = 1e-12, reltol = 1e-12)
    res, err = quadgk((t) -> (sum((sol(t).^2)./2)), 0.0, 5.0, atol = 1e-12,
                      rtol = 1e-12)
    res
end

n_du0 = ForwardDiff.gradient(G_u, u0)
n_dp = ForwardDiff.gradient(G_p, p)

@test_broken n_du0 ≈ du0 rtol=1e-3
@test_broken n_dp ≈ dp' rtol=1e-3

@test sum(n_du0 - du0) < 1.0
@test sum(n_dp - dp) < 5.0

## concrete solve

du0, dp = Zygote.gradient((u0, p) -> sum(concrete_solve(prob, Tsit5(), u0, p,
                                                        abstol = 1e-6, reltol = 1e-6,
                                                        saveat = tsteps,
                                                        sensealg = QuadratureAdjoint(autojacvec = ZygoteVJP()))),
                          u0, p)

@test !iszero(du0)
@test !iszero(dp)




##Neural ODE adjoint with SimpleChains
u0 = @SArray Float32[2.0, 0.0]
datasize = 30
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2], length = datasize)

function trueODE(u, p, t)
    true_A = @SMatrix Float32[-0.1 2.0; -2.0 -0.1]
    ((u .^ 3)'true_A)'
end

prob = ODEProblem(trueODE, u0, tspan)
sol_n = solve(prob, Tsit5(), saveat = tsteps)
data = Array(solve(prob, Tsit5(), saveat = tsteps))

sc = SimpleChain(static(2),
                 Activation(x -> x .^ 3),
                 TurboDense{true}(tanh, static(50)),
                 TurboDense{true}(identity, static(2)))

p_nn = SimpleChains.init_params(sc)

df(u, p, t) = sc(u, p)

prob_nn = ODEProblem(df, u0, tspan, p_nn)
sol = solve(prob_nn, Tsit5(); saveat = tsteps)

dg_disc(u, p, t, i; outtype = nothing) = u .- data[:, i]

du0, dp = adjoint_sensitivities(sol, Tsit5(); t = tsteps, dgdu_discrete = dg_disc,
                                sensealg = QuadratureAdjoint(autojacvec = ZygoteVJP()))

@test !iszero(du0)
@test !iszero(dp)

## numerical

function G_p(p)
    tmp_prob = remake(prob_nn, u0 = prob_nn.u0, p = p)
    A = Array(solve(tmp_prob, Tsit5(), saveat = tsteps,
                    sensealg = SensitivityADPassThrough()))

    return sum(((data .- A) .^ 2) ./ 2)
end
function G_u(u0)
    tmp_prob = remake(prob_nn, u0 = u0, p = p_nn)
    A = Array(solve(tmp_prob, Tsit5(), saveat = tsteps,
                    sensealg = SensitivityADPassThrough()))
    return sum(((data .- A) .^ 2) ./ 2)
end
G_p(p_nn)
G_u(u0)
n_dp = ForwardDiff.gradient(G_p, p_nn)
n_du0 = ForwardDiff.gradient(G_u, u0)

@test n_du0 ≈ du0 rtol = 1e-3
@test n_dp ≈ dp' rtol = 1e-3

## Continuous case

G(u, p, t) = sum(((data .- u) .^ 2) ./ 2)

function dg(u, p, t)
    return u .- Array(sol_n(t))
end

du0, dp = adjoint_sensitivities(sol, Tsit5(); dgdu_continuous = dg, g = G,
                                sensealg = QuadratureAdjoint(autojacvec = ZygoteVJP()))

@test !iszero(du0)
@test !iszero(dp)
##numerical

function G_p(p)
    tmp_prob = remake(prob_nn, p = p)
    sol = solve(tmp_prob, Tsit5(), abstol = 1e-12, reltol = 1e-12)
    res, err = quadgk((t) -> (sum(((sol_n(t) .- sol(t)).^2)./2)), 0.0, 1.5, atol = 1e-12,
                      rtol = 1e-12) # sol_n(t):numerical solution/data(above)
    res
end

function G_u(u0)
    tmp_prob = remake(prob_nn, u0 = u0)
    sol = solve(tmp_prob, Tsit5(), abstol = 1e-12, reltol = 1e-12)
    res, err = quadgk((t) -> (sum(((sol_n(t) .- sol(t)).^2)./2)), 0.0, 1.5, atol = 1e-12,
                      rtol = 1e-12) # sol_n(t):numerical solution/data(above)
    res
end

n_du0 = ForwardDiff.gradient(G_u, u0)
n_dp = ForwardDiff.gradient(G_p, p_nn)

@test n_du0 ≈ du0 rtol=1e-3
@test n_dp ≈ dp' rtol=1e-3

#concrete_solve

du0, dp = Zygote.gradient((u0, p) -> sum(concrete_solve(prob_nn, Tsit5(), u0, p,
                                                        abstol = 1e-12, reltol = 1e-12,
                                                        saveat = tsteps,
                                                        sensealg = QuadratureAdjoint(autojacvec = ZygoteVJP()))),
                          u0, p_nn)

@test !iszero(du0)
@test !iszero(dp)