using DiffEqSensitivity, Test

@test DiffEqSensitivity.hasbranching(1, 2) do x, y
   (x < 0 ? -x : x) + exp(y)
end

@test !DiffEqSensitivity.hasbranching(1, 2) do x, y
   ifelse(x < 0, -x, x) + exp(y)
end

# SciML/DiffEqFlux#586

using DifferentialEquations, DiffEqFlux
using Flux, Flux.Data

pi_32 = convert(Float32, π)
P = 2 * pi_32
T = 1.0f0

h(x::Float32) = sin(2 * pi_32 / P * x)
dh(x::Float32) = 2 * pi_32 / P * cos(2 * pi_32 / P * x)

# %% Sample points
using Random
Random.seed!(42)

N1 = 1000
t1 = convert.(Float32, rand(N1) .* T)
x1 = convert.(Float32, rand(N1) .* P)

# %% Model definition
function create_base_model()
    return Chain(Dense(1, 5, σ), Dense(5, 1))
end
function nn_basis_functions(model::Chain)
    f(z) = model([z])[1]
    f_z(z) = Flux.gradient(f, z)[1]
    return (f, f_z)
end

# %% Compute z function given x value (solve ODE)
function compute_z(x::Float32, f, f_z; saveat=[])
    function node_system!(dz, z, _, _)
        z1, z2 = z
        dz[1] = f(z1)
        dz[2] = f_z(z1) * z2
    end
    z0 = [x, 1.0f0]
    tspan = (0.0f0, T)
    prob = ODEProblem(node_system!, z0, tspan; saveat=saveat)

    return solve(prob)
end

# %% Loss functions
function loss_1(S, f, f_z)
    rows = size(S, 1)
    loss = 0.0f0
    for i = 1:rows
        t = S[i, 1]
        x = S[i, 2]
        z = compute_z(x, f, f_z; saveat=[t])
        # z1, z2 = z(t)
        z1, z2 = z[1][1], z[1][2]
        u_t = dh(z1) * f(z1)
        u_x = dh(z1) * z2
        loss += (u_t + u_x)^2
    end
    return loss / rows
end
function loss_flux(model, t1, x1)
    f, f_z = nn_basis_functions(model)
    b1 = hcat(t1, x1)
    return loss_1(b1, f, f_z)
end

# %% Optimization loop
function run_optimization_flux()
    model = create_base_model()
    data_loader = DataLoader((t1, x1), batchsize=10, shuffle=true)
    loss(t1, x1) = loss_flux(model, t1, x1)
    callback() = Flux.throttle(() => @show(loss(t1, x1)), 10)
    Flux.train!(loss, Flux.params(model), data_loader, ADAM(0.05); cb=callback)
end

@testset "SciML/DiffEqFlux#586" begin
    run_optimization_flux()
    @test true
end
