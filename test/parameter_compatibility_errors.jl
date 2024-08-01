using OrdinaryDiffEq, SciMLSensitivity, Zygote, Test

function f!(du, u, p, t)
    du[1] = -p[1]' * u
    du[2] = (p[2].a + p[2].b)u[2]
    du[3] = p[3](u, t)
    return nothing
end

struct mystruct
    a::Any
    b::Any
end

function control(u, t)
    return -exp(-t) * u[3]
end

u0 = [10, 15, 20]
p = [[1; 2; 3], mystruct(-1, -2), control]
tspan = (0.0, 10.0)

prob = ODEProblem(f!, u0, tspan, p)

sol = solve(prob, Tsit5()) # Solves without errors

function loss(p1)
    sol = solve(prob, Tsit5(), p = [p1, mystruct(-1, -2), control])
    return sum(abs2, sol)
end

grad(p) = Zygote.gradient(loss, p)

p2 = [4; 5; 6]
@test_throws SciMLSensitivity.SciMLStructuresCompatibilityError grad(p2)

function loss(p1)
    sol = solve(prob, Tsit5(), p = [p1, mystruct(-1, -2), control],
        sensealg = InterpolatingAdjoint())
    return sum(abs2, sol)
end

@test_throws SciMLSensitivity.AdjointSensitivityParameterCompatibilityError grad(p2)

function loss(p1)
    sol = solve(prob, Tsit5(), p = [p1, mystruct(-1, -2), control],
        sensealg = ForwardSensitivity())
    return sum(abs2, sol)
end

@test_throws SciMLSensitivity.SciMLStructuresCompatibilityError grad(p2)
@test_throws SciMLSensitivity.ForwardSensitivityParameterCompatibilityError ODEForwardSensitivityProblem(
    f!,
    u0,
    tspan,
    p)
