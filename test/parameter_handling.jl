using SciMLSensitivity, Lux, Random, Zygote, NonlinearSolve, OrdinaryDiffEq, Test

@info "Testing Nonlinear Solve Adjoint with Nested Parameters"

const model_nls = Chain(Dense(2 => 2, tanh), Dense(2 => 2))
ps, st = Lux.setup(Random.default_rng(), model_nls)
const st_nls = st

x = ones(Float32, 2, 3)

nlprob(u, p) = first(model_nls(u, p, st_nls)) .- u

prob = NonlinearProblem(nlprob, zeros(2, 3), ps)

@test_nowarn solve(prob, NewtonRaphson())

gs = only(Zygote.gradient(ps) do ps
    prob = NonlinearProblem(nlprob, zero.(x), ps)
    sol = solve(prob, NewtonRaphson())
    return sum(sol.u)
end)

@test gs.layer_1.weight !== nothing
@test gs.layer_1.bias !== nothing
@test gs.layer_2.weight !== nothing
@test gs.layer_2.bias !== nothing

@info "Testing Gauss Adjoint with Nested Parameters"

const model_ga = Chain(Dense(2 => 2, tanh), Dense(2 => 2))
ps, st = Lux.setup(Random.default_rng(), model_ga)
const st_ga = st

x = ones(Float32, 2, 3)

odeprob(u, p, t) = first(model_ga(u, p, st_ga))

prob = ODEProblem(odeprob, ones(2, 3), (0.0f0, 1.0f0), ps)

@test_nowarn solve(prob, Tsit5())

gs = only(Zygote.gradient(ps) do ps
    prob = ODEProblem(odeprob, ones(2, 3), (0.0f0, 1.0f0), ps)
    sol = solve(prob, Tsit5(); sensealg = GaussAdjoint(; autojacvec = ZygoteVJP()))
    return sum(last(sol.u))
end)

@test gs.layer_1.weight !== nothing
@test gs.layer_1.bias !== nothing
@test gs.layer_2.weight !== nothing
@test gs.layer_2.bias !== nothing
