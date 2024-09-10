using Test, LinearAlgebra
using SciMLSensitivity, StochasticDiffEq
using Random

@info "Test SDE Transformation"

seed = 100
tspan = (0.0, 0.1)
p = [1.01, 0.87]

# scalar
f(u, p, t) = p[1] * u
σ(u, p, t) = p[2] * u

Random.seed!(seed)
u0 = rand(1)
linear_analytic(u0, p, t, W) = @.(u0*exp((p[1] - p[2]^2 / 2) * t + p[2] * W))

prob = SDEProblem(SDEFunction(f, σ, analytic = linear_analytic), σ, u0, tspan, p)
sol = solve(prob, SOSRI(), adaptive = false, dt = 0.001, save_noise = true)

@test isapprox(sol.u_analytic, sol.u, atol = 1e-4)

du = zeros(size(u0))
u = sol.u[end]
transformed_function = StochasticTransformedFunction(sol, sol.prob.f, sol.prob.g)
#transformed_function(du,u,p,tspan[2])
du2 = transformed_function(u, p, tspan[2])

#@test du[1] == (p[1]*u[1]-p[2]^2*u[1])
@test isapprox(du2[1], (p[1] * u[1] - p[2]^2 * u[1]), atol = 1e-15)
#@test du2 == du
transformed_function = StochasticTransformedFunction(sol, sol.prob.f, sol.prob.g,
    (u, p, t) -> p[2]^2 * u)
du2 = transformed_function(u, p, tspan[2])
@test isapprox(du2[1], (p[1] * u[1] - p[2]^2 * u[1]), atol = 1e-15)

linear_analytic_strat(u0, p, t, W) = @.(u0*exp((p[1]) * t + p[2] * W))

prob_strat = SDEProblem{false}(
    SDEFunction((u, p, t) -> p[1] * u - 1 // 2 * p[2]^2 * u, σ,
        analytic = linear_analytic_strat),
    σ,
    u0,
    tspan,
    p)
Random.seed!(seed)
sol_strat = solve(
    prob_strat, RKMil(interpretation = SciMLBase.AlgorithmInterpretation.Stratonovich),
    adaptive = false,
    dt = 0.0001, save_noise = true)
prob_strat1 = SDEProblem{false}(
    SDEFunction((u, p, t) -> transformed_function(u, p, t) .+
                             1 // 2 * p[2]^2 * u[1], σ,
        analytic = linear_analytic),
    σ,
    u0,
    tspan,
    p)
Random.seed!(seed)
sol_strat1 = solve(
    prob_strat1, RKMil(interpretation = SciMLBase.AlgorithmInterpretation.Stratonovich),
    adaptive = false,
    dt = 0.0001, save_noise = true)

# Test if we recover Ito solution in Stratonovich sense
@test isapprox(sol_strat.u, sol_strat1.u, atol = 1e-4) # own transformation and custom function agree
@test !isapprox(sol_strat.u_analytic, sol_strat.u, atol = 1e-4) # we don't get the stratonovich solution for the linear SDE
@test isapprox(sol_strat1.u_analytic, sol_strat.u, atol = 1e-3) # we do recover the analytic solution from the Ito sense

# inplace

f!(du, u, p, t) = @.(du=p[1] * u)
σ!(du, u, p, t) = @.(du=p[2] * u)

prob = SDEProblem(SDEFunction(f!, σ!, analytic = linear_analytic), σ!, u0, tspan, p)
sol = solve(prob, SOSRI(), adaptive = false, dt = 0.001, save_noise = true)

@test isapprox(sol.u_analytic, sol.u, atol = 1e-4)

du = zeros(size(u0))
u = sol.u[end]
transformed_function = StochasticTransformedFunction(sol, sol.prob.f, sol.prob.g)
transformed_function(du, u, p, tspan[2])

@test isapprox(du[1], (p[1] * u[1] - p[2]^2 * u[1]), atol = 1e-15)
# @test isapprox(du2[1], (p[1]*u[1]-p[2]^2*u[1]), atol=1e-15)
# @test isapprox(du2, du,  atol=1e-15)

transformed_function = StochasticTransformedFunction(sol, sol.prob.f, sol.prob.g,
    (du, u, p, t) -> (du .= p[2]^2 * u))
transformed_function(du, u, p, tspan[2])
@test du[1] == (p[1] * u[1] - p[2]^2 * u[1])

# diagonal noise

u0 = rand(3)

prob = SDEProblem(SDEFunction(f, σ, analytic = linear_analytic), σ, u0, tspan, p)
sol = solve(prob, SOSRI(), adaptive = false, dt = 0.001, save_noise = true)
u = sol.u[end]

transformed_function = StochasticTransformedFunction(sol, sol.prob.f, sol.prob.g)
du2 = transformed_function(u, p, tspan[2])
@test isapprox(du2, (p[1] * u - p[2]^2 * u), atol = 1e-15)

transformed_function = StochasticTransformedFunction(sol, sol.prob.f, sol.prob.g,
    (u, p, t) -> p[2]^2 * u)
du2 = transformed_function(u, p, tspan[2])
@test du2[1] == (p[1] * u[1] - p[2]^2 * u[1])

prob = SDEProblem(SDEFunction(f!, σ!, analytic = linear_analytic), σ!, u0, tspan, p)
sol = solve(prob, SOSRI(), adaptive = false, dt = 0.001, save_noise = true)

du = zeros(size(u0))
u = sol.u[end]
transformed_function = StochasticTransformedFunction(sol, sol.prob.f, sol.prob.g)
transformed_function(du, u, p, tspan[2])
@test isapprox(du, (p[1] * u - p[2]^2 * u), atol = 1e-15)

transformed_function = StochasticTransformedFunction(sol, sol.prob.f, sol.prob.g,
    (du, u, p, t) -> (du .= p[2]^2 * u))
transformed_function(du, u, p, tspan[2])
@test isapprox(du, (p[1] * u - p[2]^2 * u), atol = 1e-15)

#  non-diagonal noise torus
u0 = rand(2)
p = rand(1)

fnd(u, p, t) = 0 * u
function σnd(u, p, t)
    du = [cos(p[1])*sin(u[1]) cos(p[1])*cos(u[1]) -sin(p[1])*sin(u[2]) -sin(p[1])*cos(u[2])
          sin(p[1])*sin(u[1]) sin(p[1])*cos(u[1]) cos(p[1])*sin(u[2]) cos(p[1])*cos(u[2])]
    return du
end

prob = SDEProblem(fnd, σnd, u0, tspan, p, noise_rate_prototype = zeros(2, 4))
sol = solve(prob, EM(), adaptive = false, dt = 0.001, save_noise = true)

transformed_function = StochasticTransformedFunction(sol, sol.prob.f, sol.prob.g)
du2 = transformed_function(u0, p, tspan[2])
@test isapprox(du2, zeros(2), atol = 1e-15)

transformed_function = StochasticTransformedFunction(sol, sol.prob.f, sol.prob.g,
    (u, p, t) -> false * u)
du2 = transformed_function(u0, p, tspan[2])
@test isapprox(du2, zeros(2), atol = 1e-15)

fnd!(du, u, p, t) = du .= false
function σnd!(du, u, p, t)
    du[1, 1] = cos(p[1]) * sin(u[1])
    du[1, 2] = cos(p[1]) * cos(u[1])
    du[1, 3] = -sin(p[1]) * sin(u[2])
    du[1, 4] = -sin(p[1]) * cos(u[2])
    du[2, 1] = sin(p[1]) * sin(u[1])
    du[2, 2] = sin(p[1]) * cos(u[1])
    du[2, 3] = cos(p[1]) * sin(u[2])
    du[2, 4] = cos(p[1]) * cos(u[2])
    return nothing
end

prob = SDEProblem(fnd!, σnd!, u0, tspan, p, noise_rate_prototype = zeros(2, 4))
sol = solve(prob, EM(), adaptive = false, dt = 0.001, save_noise = true)

du = zeros(size(u0))
u = sol.u[end]
transformed_function = StochasticTransformedFunction(sol, sol.prob.f, sol.prob.g)
transformed_function(du, u, p, tspan[2])
@test isapprox(du, zeros(2), atol = 1e-15)

transformed_function = StochasticTransformedFunction(sol, sol.prob.f, sol.prob.g,
    (du, u, p, t) -> (du .= false * u))
transformed_function(du, u, p, tspan[2])
@test isapprox(du, zeros(2), atol = 1e-15)

t = sol.t[end]

"""
Check compatibility of StochasticTransformedFunction with vjp for adjoints
"""

###
# Check general compatibility of StochasticTransformedFunction() with Zygote
###

using Zygote

# scalar case
Random.seed!(seed)
u0 = rand(1)
p = rand(2)
λ = rand(1)

_dy, back = Zygote.pullback(u0, p) do u, p
    vec(f(u, p, t) - p[2]^2 * u)
end
∇1, ∇2 = back(λ)

@test isapprox(∇1, (p[1] - p[2]^2) * λ, atol = 1e-15)
@test isapprox(∇2, (@. [1, -2 * p[2]] * u0 * λ[1]), atol = 1e-15)

prob = SDEProblem(f, σ, u0, tspan, p)
sol = solve(prob, SOSRI(), adaptive = false, dt = 0.001, save_noise = true)
transformed_function = StochasticTransformedFunction(sol, sol.prob.f, sol.prob.g)

# Zygote doesn't allow nesting
_dy, back = Zygote.pullback(u0, p) do u, p
    vec(transformed_function(u, p, t))
end
@test isapprox(∇1, (p[1] - p[2]^2) * λ, atol = 1e-15)
@test isapprox(∇2, (@. [1, -2 * p[2]] * u0 * λ[1]), atol = 1e-15)

transformed_function = StochasticTransformedFunction(sol, sol.prob.f, sol.prob.g,
    (u, p, t) -> p[2]^2 * u)
_dy, back = Zygote.pullback(u0, p) do u, p
    vec(transformed_function(u, p, t))
end
∇1, ∇2 = back(λ)
@test isapprox(∇1, (p[1] - p[2]^2) * λ, atol = 1e-15)
@test isapprox(∇2, (@. [1, -2 * p[2]] * u0 * λ[1]), atol = 1e-15)

###
# Check general compatibility of StochasticTransformedFunction() with ReverseDiff
###

using ReverseDiff

# scalar

tape = ReverseDiff.GradientTape((u0, p, [t])) do u, p, t
    vec(f(u, p, t) - p[2]^2 * u)
end

tu, tp, tt = ReverseDiff.input_hook(tape)

output = ReverseDiff.output_hook(tape)
ReverseDiff.unseed!(tu) # clear any "leftover" derivatives from previous calls
ReverseDiff.unseed!(tp)
ReverseDiff.unseed!(tt)

ReverseDiff.value!(tu, u0)
ReverseDiff.value!(tp, prob.p)
ReverseDiff.value!(tt, [t])
ReverseDiff.forward_pass!(tape)
ReverseDiff.increment_deriv!(output, λ)
ReverseDiff.reverse_pass!(tape)

@test isapprox(ReverseDiff.deriv(tu), (p[1] - p[2]^2) * λ, atol = 1e-15)
@test isapprox(ReverseDiff.deriv(tp), (@. [1, -2 * p[2]] * u0 * λ[1]), atol = 1e-15)

tape = ReverseDiff.GradientTape((u0, p, [t])) do u, p, t
    _dy, back = Zygote.pullback(u, p) do u, p
        vec(σ(u, p, t))
    end
    tmp1, tmp2 = back(_dy)
    return f(u, p, t) - vec(tmp1)
end

tu, tp, tt = ReverseDiff.input_hook(tape)

output = ReverseDiff.output_hook(tape)
ReverseDiff.unseed!(tu) # clear any "leftover" derivatives from previous calls
ReverseDiff.unseed!(tp)
ReverseDiff.unseed!(tt)

ReverseDiff.value!(tu, u0)
ReverseDiff.value!(tp, prob.p)
ReverseDiff.value!(tt, [t])
ReverseDiff.forward_pass!(tape)
ReverseDiff.increment_deriv!(output, λ)
ReverseDiff.reverse_pass!(tape)

@test isapprox(ReverseDiff.deriv(tu), (p[1] - p[2]^2) * λ, atol = 1e-15)
@test isapprox(ReverseDiff.deriv(tp), (@. [1, -2 * p[2]] * u0 * λ[1]), atol = 1e-15)

transformed_function = StochasticTransformedFunction(sol, sol.prob.f, sol.prob.g)

tape = ReverseDiff.GradientTape((u0, p, [t])) do u, p, t
    vec(transformed_function(u, p, first(t)))
end

tu, tp, tt = ReverseDiff.input_hook(tape)

output = ReverseDiff.output_hook(tape)
ReverseDiff.unseed!(tu) # clear any "leftover" derivatives from previous calls
ReverseDiff.unseed!(tp)
ReverseDiff.unseed!(tt)

ReverseDiff.value!(tu, u0)
ReverseDiff.value!(tp, prob.p)
ReverseDiff.value!(tt, [t])
ReverseDiff.forward_pass!(tape)
ReverseDiff.increment_deriv!(output, λ)
ReverseDiff.reverse_pass!(tape)

@test isapprox(ReverseDiff.deriv(tu), (p[1] - p[2]^2) * λ, atol = 1e-15)
@test isapprox(ReverseDiff.deriv(tp), (@. [1, -2 * p[2]] * u0 * λ[1]), atol = 1e-15)

# diagonal
Random.seed!(seed)
u0 = rand(3)
λ = rand(3)

_dy, back = Zygote.pullback(u0, p) do u, p
    vec(f(u, p, t) - p[2]^2 * u)
end
∇1, ∇2 = back(λ)

@test isapprox(∇1, (p[1] - p[2]^2) * λ, atol = 1e-15)
@test isapprox(∇2[1], dot(u0, λ), atol = 1e-15)
@test isapprox(∇2[2], -2 * p[2] * dot(u0, λ), atol = 1e-15)

tape = ReverseDiff.GradientTape((u0, p, [t])) do u, p, t
    vec(transformed_function(u, p, first(t)))
end

tu, tp, tt = ReverseDiff.input_hook(tape)

output = ReverseDiff.output_hook(tape)
ReverseDiff.unseed!(tu) # clear any "leftover" derivatives from previous calls
ReverseDiff.unseed!(tp)
ReverseDiff.unseed!(tt)

ReverseDiff.value!(tu, u0)
ReverseDiff.value!(tp, p)
ReverseDiff.value!(tt, [t])
ReverseDiff.forward_pass!(tape)
ReverseDiff.increment_deriv!(output, λ)
ReverseDiff.reverse_pass!(tape)

tmptp = ReverseDiff.deriv(tp)

@test isapprox(ReverseDiff.deriv(tu), (p[1] - p[2]^2) * λ, atol = 1e-15)
@test isapprox(tmptp[1], dot(u0, λ), atol = 1e-15)
@test isapprox(tmptp[2], -2 * p[2] * dot(u0, λ), atol = 1e-15)

# non-diagonal
Random.seed!(seed)
u0 = rand(2)
p = rand(1)
λ = rand(2)

_dy, back = Zygote.pullback(u0, p) do u, p
    vec(fnd(u, p, t))
end
∇1, ∇2 = back(λ)

@test isapprox(∇1, zero(∇1), atol = 1e-15)

prob = SDEProblem(fnd, σnd, u0, tspan, p, noise_rate_prototype = zeros(2, 4))
sol = solve(prob, EM(), adaptive = false, dt = 0.001, save_noise = true)
transformed_function = StochasticTransformedFunction(sol, sol.prob.f, sol.prob.g)

tape = ReverseDiff.GradientTape((u0, p, [t])) do u, p, t
    vec(transformed_function(u, p, first(t)))
end

tu, tp, tt = ReverseDiff.input_hook(tape)

output = ReverseDiff.output_hook(tape)
ReverseDiff.unseed!(tu) # clear any "leftover" derivatives from previous calls
ReverseDiff.unseed!(tp)
ReverseDiff.unseed!(tt)

ReverseDiff.value!(tu, u0)
ReverseDiff.value!(tp, p)
ReverseDiff.value!(tt, [t])
ReverseDiff.forward_pass!(tape)
ReverseDiff.increment_deriv!(output, λ)
ReverseDiff.reverse_pass!(tape)

@test isapprox(ReverseDiff.deriv(tu), zero(u0), atol = 1e-15)
@test isapprox(ReverseDiff.deriv(tp), zero(p), atol = 1e-15)

###
# Check Mutating functions
###
# scalar
Random.seed!(seed)
u0 = rand(1)
p = rand(2)
λ = rand(1)

prob = SDEProblem(SDEFunction(f!, σ!, analytic = linear_analytic), σ!, u0, tspan, p)
sol = solve(prob, SOSRI(), adaptive = false, dt = 0.001, save_noise = true)

du = zeros(size(u0))
u = sol.u[end]
transformed_function = StochasticTransformedFunction(sol, sol.prob.f, sol.prob.g)

function inplacefunc!(du, u, p, t)
    du .= p[2]^2 * u
    return nothing
end

tape = ReverseDiff.GradientTape((u0, p, [t])) do u, p, t
    du1 = similar(u, size(u))
    du2 = similar(u, size(u))
    f!(du1, u, p, first(t))
    inplacefunc!(du2, u, p, first(t))
    return vec(du1 - du2)
end

tu, tp, tt = ReverseDiff.input_hook(tape) # u0

output = ReverseDiff.output_hook(tape) # p[1]*u0 -p[2]^2*u0
ReverseDiff.unseed!(tu) # clear any "leftover" derivatives from previous calls
ReverseDiff.unseed!(tp)
ReverseDiff.unseed!(tt)

ReverseDiff.value!(tu, u0)
ReverseDiff.value!(tp, p)
ReverseDiff.value!(tt, [t])
ReverseDiff.forward_pass!(tape)
ReverseDiff.increment_deriv!(output, λ)
ReverseDiff.reverse_pass!(tape)

@test isapprox(ReverseDiff.deriv(tu), (p[1] - p[2]^2) * λ, atol = 1e-15) # -0.016562475307537294
@test isapprox(ReverseDiff.deriv(tp), (@. [1, -2 * p[2]] * u0 * λ[1]), atol = 1e-15)  #[0.017478629739736098, -0.023103635221731166]

tape = ReverseDiff.GradientTape((u0, p, [t])) do u, p, t
    _dy, back = Zygote.pullback(u, p) do u, p
        out_ = Zygote.Buffer(similar(u))
        σ!(out_, u, p, t)
        vec(copy(out_))
    end
    tmp1, tmp2 = back(λ)
    du1 = similar(u, size(u))
    f!(du1, u, p, first(t))
    return vec(du1 - tmp1)
end

tu, tp, tt = ReverseDiff.input_hook(tape)

output = ReverseDiff.output_hook(tape)
ReverseDiff.unseed!(tu) # clear any "leftover" derivatives from previous calls
ReverseDiff.unseed!(tp)
ReverseDiff.unseed!(tt)

ReverseDiff.value!(tu, u0)
ReverseDiff.value!(tp, p)
ReverseDiff.value!(tt, [t])
ReverseDiff.forward_pass!(tape)
ReverseDiff.increment_deriv!(output, λ)
ReverseDiff.reverse_pass!(tape)

@test_broken isapprox(ReverseDiff.deriv(tu), (p[1] - p[2]^2) * λ, atol = 1e-15)
@test_broken isapprox(ReverseDiff.deriv(tp), (@. [1, -2 * p[2]] * u0 * λ[1]), atol = 1e-15)

tape = ReverseDiff.GradientTape((u0, p, [t])) do u1, p1, t1
    du1 = similar(u1, size(u1))
    transformed_function(du1, u1, p1, first(t1))
    return vec(du1)
end

tu, tp, tt = ReverseDiff.input_hook(tape) # p[1]*u0

output = ReverseDiff.output_hook(tape) # p[1]*u0 -p[2]^2*u0
ReverseDiff.unseed!(tu) # clear any "leftover" derivatives from previous calls
ReverseDiff.unseed!(tp)
ReverseDiff.unseed!(tt)

ReverseDiff.value!(tu, u0)
ReverseDiff.value!(tp, p)
ReverseDiff.value!(tt, [t])
ReverseDiff.forward_pass!(tape)
ReverseDiff.increment_deriv!(output, λ)
ReverseDiff.reverse_pass!(tape)

@test isapprox(ReverseDiff.deriv(tu), (p[1] - p[2]^2) * λ, atol = 1e-15)
@test isapprox(ReverseDiff.deriv(tp), (@. [1, -2 * p[2]] * u0 * λ[1]), atol = 1e-15)

transformed_function = StochasticTransformedFunction(sol, sol.prob.f, sol.prob.g,
    (du, u, p, t) -> (du .= p[2]^2 * u))

tape = ReverseDiff.GradientTape((u0, p, [t])) do u1, p1, t1
    du1 = similar(u1, size(u1))
    transformed_function(du1, u1, p1, first(t1))
    return vec(du1)
end

tu, tp, tt = ReverseDiff.input_hook(tape) # p[1]*u0

output = ReverseDiff.output_hook(tape) # p[1]*u0 -p[2]^2*u0
ReverseDiff.unseed!(tu) # clear any "leftover" derivatives from previous calls
ReverseDiff.unseed!(tp)
ReverseDiff.unseed!(tt)

ReverseDiff.value!(tu, u0)
ReverseDiff.value!(tp, p)
ReverseDiff.value!(tt, [t])
ReverseDiff.forward_pass!(tape)
ReverseDiff.increment_deriv!(output, λ)
ReverseDiff.reverse_pass!(tape)

@test isapprox(ReverseDiff.deriv(tu), (p[1] - p[2]^2) * λ, atol = 1e-15)
@test isapprox(ReverseDiff.deriv(tp), (@. [1, -2 * p[2]] * u0 * λ[1]), atol = 1e-15)
