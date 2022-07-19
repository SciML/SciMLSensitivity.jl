using SciMLSensitivity, OrdinaryDiffEq, SimpleChains, StaticArrays, QuadGK, ForwardDiff,
      Zygote
using Test



#####
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

#####Delete################################################################
using Flux

u0 = Float32[2.0; 0.0]
datasize = 30
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2], length = datasize)

function trueODEfunc(u, p, t)
    true_A = [-0.1f0 2.0f0; -2.0f0 -0.1f0]
     return ((u.^3)'true_A)'
end

prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))

dudt2 = Chain((x) -> x.^3,
                  Dense(2, 50, tanh),
                  Dense(50, 2))
p, re = Flux.destructure(dudt2)
f(u, p, t) = re(p)(u)

prob_nn = ODEProblem(f, u0, tspan)

du0, dp = Zygote.gradient((u0, p) -> sum(concrete_solve(prob_nn, Tsit5(), u0, p,
                                                        abstol = 1e-12, reltol = 1e-12,
                                                        saveat = tsteps,
                                                        sensealg = QuadratureAdjoint(autojacvec = ZygoteVJP()))),
                          u0, p)