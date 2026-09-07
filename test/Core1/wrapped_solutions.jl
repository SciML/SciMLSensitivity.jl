using OrdinaryDiffEq, SciMLSensitivity, SciMLBase, Zygote, ForwardDiff, Test

# A problem can carry discretization metadata as its `problem_type`; `solve` then passes the
# solver's solution through `wrap_sol`, which returns a higher-level object (PDE discretizers do
# this). Sensitivity code must see the solver's solution, so the forward solves inside the
# sensitivity rules pass `wrap = Val(false)`. The wrapper here is opaque on purpose: it is not a
# time series, and any code that receives it instead of the solution fails.
struct WrapMeta <: SciMLBase.AbstractDiscretizationMetadata{Val{true}} end
struct Wrapped{S}
    sol::S
end
SciMLBase.wrap_sol(sol, ::WrapMeta) = Wrapped(sol)

f(u, p, t) = [p[1] * u[1] - p[2] * u[1] * u[2], p[2] * u[1] * u[2] - p[3] * u[2]]
f!(du, u, p, t) = (du .= f(u, p, t); nothing)
u0 = [1.0, 1.0]
p = [1.5, 1.0, 3.0]
tspan = (0.0, 1.0)
ts = 0.0:0.1:1.0
plain = ODEProblem{false}(ODEFunction{false}(f), u0, tspan, p)
wrapped = ODEProblem{false}(ODEFunction{false}(f), u0, tspan, p, WrapMeta())
plain! = ODEProblem{true}(ODEFunction{true}(f!), u0, tspan, p)
wrapped! = ODEProblem{true}(ODEFunction{true}(f!), u0, tspan, p, WrapMeta())

@test solve(wrapped, Tsit5()) isa Wrapped
@test solve(wrapped, Tsit5(); wrap = Val(false)) isa ODESolution

function loss(prob, p, sensealg)
    sol = solve(
        remake(prob; p), Tsit5(); saveat = ts, wrap = Val(false), sensealg,
        abstol = 1.0e-10, reltol = 1.0e-10
    )
    return sum(abs2, Array(sol))
end
reference = ForwardDiff.gradient(p -> loss(plain, p, nothing), p)
label(sensealg) = string(
    nameof(typeof(sensealg)),
    hasproperty(sensealg, :checkpointing) && sensealg.checkpointing ? " checkpointing" : ""
)

# `ForwardSensitivity` needs an in-place function; the rest run on the out-of-place problem.
@testset "$(label(sensealg))" for (sensealg, prob, wprob) in (
        (ForwardDiffSensitivity(), plain, wrapped),
        (ForwardSensitivity(), plain!, wrapped!),
        (InterpolatingAdjoint(), plain, wrapped),
        (InterpolatingAdjoint(checkpointing = true), plain, wrapped),
        (QuadratureAdjoint(), plain, wrapped),
        (GaussAdjoint(), plain, wrapped),
        (GaussAdjoint(checkpointing = true), plain, wrapped),
        (BacksolveAdjoint(), plain, wrapped),
        (ReverseDiffAdjoint(), plain, wrapped),
        (TrackerAdjoint(), plain, wrapped),
    )
    g_plain = Zygote.gradient(p -> loss(prob, p, sensealg), p)[1]
    g_wrapped = Zygote.gradient(p -> loss(wprob, p, sensealg), p)[1]
    @test g_plain ≈ reference rtol = 1.0e-5
    @test g_wrapped == g_plain
end

# The same for a fully implicit DAE, whose adjoint interpolates the forward solution.
using OrdinaryDiffEqBDF: DFBDF

function rober!(res, du, u, p, t)
    y₁, y₂, y₃ = u
    k₁, k₂, k₃ = p
    res[1] = -k₁ * y₁ + k₃ * y₂ * y₃ - du[1]
    res[2] = k₁ * y₁ - k₂ * y₂^2 - k₃ * y₂ * y₃ - du[2]
    res[3] = y₁ + y₂ + y₃ - 1
    return nothing
end
dae_u0 = [1.0, 0.0, 0.0]
dae_p = [0.04, 3.0e7, 1.0e4]
dae_du0 = [-0.04, 0.04, 0.0]
dae_kw = (; differential_vars = [true, true, false])
dae_plain = DAEProblem{true}(DAEFunction{true}(rober!), dae_du0, dae_u0, (0.0, 10.0), dae_p; dae_kw...)
dae_wrapped = DAEProblem{true}(
    DAEFunction{true}(rober!), dae_du0, dae_u0, (0.0, 10.0), dae_p, WrapMeta(); dae_kw...
)
@test solve(dae_wrapped, DFBDF()) isa Wrapped

function dae_loss(prob, p, sensealg)
    sol = solve(
        remake(prob; p), DFBDF(); saveat = 0.0:1.0:10.0, wrap = Val(false), sensealg,
        abstol = 1.0e-10, reltol = 1.0e-10
    )
    return sum(abs2, Array(sol))
end
@testset "DAEProblem $(label(sensealg))" for sensealg in (
        InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)),
        GaussAdjoint(autojacvec = ReverseDiffVJP(true)),
    )
    g_plain = Zygote.gradient(p -> dae_loss(dae_plain, p, sensealg), dae_p)[1]
    g_wrapped = Zygote.gradient(p -> dae_loss(dae_wrapped, p, sensealg), dae_p)[1]
    @test g_wrapped == g_plain
end
