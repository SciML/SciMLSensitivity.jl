# Learning Non-Linear Reaction Dynamics for the Gray–Scott Reaction–Diffusion Model using Universal Differential Equations

## Introduction
The Gray–Scott model is a prototypical reaction–diffusion system known for generating a wide variety of spatial patterns — from spots and stripes to labyrinthine structures — driven purely by simple chemical kinetics and diffusion. 

In this tutorial, we’ll employ a Universal Differential Equation (UDE) framework: embedding a small neural network within the PDE’s reaction term to learn unknown dynamics from data, while retaining the known diffusion physics.


## Equations of the Gray-Scott Model
The system is governed by the coupled PDEs:
```math
\frac{\partial u}{\partial t} = D_1\,\nabla^2 u + \frac{a\,u^2}{v} + \bar{u} - \alpha
```

```math
\frac{\partial v}{\partial t} = D_2\,\nabla^2 v + a\,u^2 + \beta\,v 
```

where $u$ and $v$ are the two chemical concentrations, $D_1$ and $D_2$ are diffusion coefficients, and $a$, $\bar{u}$, $\alpha$, $\beta$ are reaction parameters.

In its spatially discretized form (using Neumann boundary conditions and the tridiagonal stencil $[1, -2, 1]$), the Gray–Scott PDE reduces to:

```math
du = D1 * (A_y u + u A_x) + \frac{a u^2}{v} + \bar{u} - \alpha u
```

```math
dv = D2 (A_y v + v A_x) + a u^2 + \beta v
````
Here $A_x$ and $A_y$ are the 1D Laplacian matrices for x- and y-directions, respectively.

Now we will dive into the implementation of the UDE. 

## Ground-truth data generation.
```@example gray_scott
using LinearAlgebra, DifferentialEquations
using Plots

const N = 16
a = 1.0
α = 1.0
ubar = 1.0
β = 10.0 
D1 = 0.001
D2 = 100.0

Ax = Array(Tridiagonal([1.0 for i in 1:(N - 1)], [-2.0 for i in 1:N],
    [1.0 for i in 1:(N - 1)]))
Ay = copy(Ax)
Ax[2, 1] = 2.0
Ax[end - 1, end] = 2.0
Ay[1, 2] = 2.0
Ay[end, end - 1] = 2.0

uss = (ubar + β) / α
vss = (a / β) * uss^2
r0 = zeros(N, N, 2)
r0[:, :, 1] .= uss .+ 0.1 .* rand.()
r0[:, :, 2] .= vss
```

Having set up the grid, parameters, and initial condition, we now generate “ground truth” data by integrating the pure physics Gray–Scott model. This dataset will serve as the target that our UDE aims to learn.

```@example gray_scott
function fast_gm!(du, u, p, t)
    p = nothing
    @inbounds for j in 2:(N - 1), i in 2:(N - 1)
        du[i, j, 1] = D1 *
                      (u[i - 1, j, 1] + u[i + 1, j, 1] + u[i, j + 1, 1] + u[i, j - 1, 1] -
                       4u[i, j, 1]) +
                      a * u[i, j, 1]^2 / u[i, j, 2] + ubar - α * u[i, j, 1]
    end

    @inbounds for j in 2:(N - 1), i in 2:(N - 1)
        du[i, j, 2] = D2 *
                      (u[i - 1, j, 2] + u[i + 1, j, 2] + u[i, j + 1, 2] + u[i, j - 1, 2] -
                       4u[i, j, 2]) +
                      a * u[i, j, 1]^2 - β * u[i, j, 2]
    end

    @inbounds for j in 2:(N - 1)
        i = 1
        du[1, j, 1] = D1 *
                      (2u[i + 1, j, 1] + u[i, j + 1, 1] + u[i, j - 1, 1] - 4u[i, j, 1]) +
                      a * u[i, j, 1]^2 / u[i, j, 2] + ubar - α * u[i, j, 1]
    end
    @inbounds for j in 2:(N - 1)
        i = 1
        du[1, j, 2] = D2 *
                      (2u[i + 1, j, 2] + u[i, j + 1, 2] + u[i, j - 1, 2] - 4u[i, j, 2]) +
                      a * u[i, j, 1]^2 - β * u[i, j, 2]
    end
    @inbounds for j in 2:(N - 1)
        i = N
        du[end, j, 1] = D1 *
                        (2u[i - 1, j, 1] + u[i, j + 1, 1] + u[i, j - 1, 1] - 4u[i, j, 1]) +
                        a * u[i, j, 1]^2 / u[i, j, 2] + ubar - α * u[i, j, 1]
    end
    @inbounds for j in 2:(N - 1)
        i = N
        du[end, j, 2] = D2 *
                        (2u[i - 1, j, 2] + u[i, j + 1, 2] + u[i, j - 1, 2] - 4u[i, j, 2]) +
                        a * u[i, j, 1]^2 - β * u[i, j, 2]
    end

    @inbounds for i in 2:(N - 1)
        j = 1
        du[i, 1, 1] = D1 *
                      (u[i - 1, j, 1] + u[i + 1, j, 1] + 2u[i, j + 1, 1] - 4u[i, j, 1]) +
                      a * u[i, j, 1]^2 / u[i, j, 2] + ubar - α * u[i, j, 1]
    end
    @inbounds for i in 2:(N - 1)
        j = 1
        du[i, 1, 2] = D2 *
                      (u[i - 1, j, 2] + u[i + 1, j, 2] + 2u[i, j + 1, 2] - 4u[i, j, 2]) +
                      a * u[i, j, 1]^2 - β * u[i, j, 2]
    end
    @inbounds for i in 2:(N - 1)
        j = N
        du[i, end, 1] = D1 *
                        (u[i - 1, j, 1] + u[i + 1, j, 1] + 2u[i, j - 1, 1] - 4u[i, j, 1]) +
                        a * u[i, j, 1]^2 / u[i, j, 2] + ubar - α * u[i, j, 1]
    end
    @inbounds for i in 2:(N - 1)
        j = N
        du[i, end, 2] = D2 *
                        (u[i - 1, j, 2] + u[i + 1, j, 2] + 2u[i, j - 1, 2] - 4u[i, j, 2]) +
                        a * u[i, j, 1]^2 - β * u[i, j, 2]
    end

    @inbounds begin
        i = 1
        j = 1
        du[1, 1, 1] = D1 * (2u[i + 1, j, 1] + 2u[i, j + 1, 1] - 4u[i, j, 1]) +
                      a * u[i, j, 1]^2 / u[i, j, 2] + ubar - α * u[i, j, 1]
        du[1, 1, 2] = D2 * (2u[i + 1, j, 2] + 2u[i, j + 1, 2] - 4u[i, j, 2]) +
                      a * u[i, j, 1]^2 - β * u[i, j, 2]

        i = 1
        j = N
        du[1, N, 1] = D1 * (2u[i + 1, j, 1] + 2u[i, j - 1, 1] - 4u[i, j, 1]) +
                      a * u[i, j, 1]^2 / u[i, j, 2] + ubar - α * u[i, j, 1]
        du[1, N, 2] = D2 * (2u[i + 1, j, 2] + 2u[i, j - 1, 2] - 4u[i, j, 2]) +
                      a * u[i, j, 1]^2 - β * u[i, j, 2]

        i = N
        j = 1
        du[N, 1, 1] = D1 * (2u[i - 1, j, 1] + 2u[i, j + 1, 1] - 4u[i, j, 1]) +
                      a * u[i, j, 1]^2 / u[i, j, 2] + ubar - α * u[i, j, 1]
        du[N, 1, 2] = D2 * (2u[i - 1, j, 2] + 2u[i, j + 1, 2] - 4u[i, j, 2]) +
                      a * u[i, j, 1]^2 - β * u[i, j, 2]

        i = N
        j = N
        du[end, end, 1] = D1 * (2u[i - 1, j, 1] + 2u[i, j - 1, 1] - 4u[i, j, 1]) +
                          a * u[i, j, 1]^2 / u[i, j, 2] + ubar - α * u[i, j, 1]
        du[end, end, 2] = D2 * (2u[i - 1, j, 2] + 2u[i, j - 1, 2] - 4u[i, j, 2]) +
                          a * u[i, j, 1]^2 - β * u[i, j, 2]
    end
end

prob = ODEProblem(fast_gm!, r0, (0.0, 10), p = nothing)
solution_true = solve(prob, FBDF(), saveat=0.5, abstol=1e-6, reltol=1e-6);
```

With the ground-truth solutions computed, we can now proceed to visualize the spatiotemporal evolution of $u$ and $v$ on the grid.

```@example gray_scott
tsteps = 0:0.5:10
println("Plotting ground truth concentrations at the grid center...")
c = N ÷ 2
p1 = plot(tsteps, solution_true[c,c,1,:], lw=2, label="U True", color=:blue, title="Ground Truth: Center U")
xlabel!("Time"); ylabel!("Concentration")
p2 = plot(tsteps, solution_true[c,c,2,:], lw=2, label="V True", color=:red, title="Ground Truth: Center V")
xlabel!("Time")
p_center_combined = plot(p1, p2, layout=(1,2), size=(900,350))
display(p_center_combined)
```

## Defining the UDE
Now that we have an understanding of the data and its visualization, we can define the neural network and the UDE structure. We replace the $\frac{a u^2}{v} + \bar{u} - \alpha u$ term with a neural network, giving the resultant ODEs.

```math
du = D1 (A_y u + u A_x) + \mathcal{N}_\theta(u,v)
```

```math
dv = D2 (A_y v + v A_x) + a u^2 + \beta v
````

The first step is to define the neural network structure.

```@example gray_scott
using Lux
using Random
using ComponentArrays
neural_network = Lux.Chain(Dense(2, 16, tanh), Dense(16, 16, tanh), Dense(16, 1, tanh))
ps, st = Lux.setup(Random.default_rng(), neural_network)
ps = ComponentArray(ps)
global global_st = st
```

The following code describes the UDE formulation with the neural network predictions embedded, as well as a function wrapper to make sure the arrays work with `Optimization.jl`

```@example gray_scott
function ude!(du, u, p, t)
    @inbounds for j in 2:(N - 1), i in 2:(N - 1)
        u_val = u[i, j, 1]
        v_val = u[i, j, 2]
        input = [u_val, v_val]
        nn_out, _ = Lux.apply(neural_network, input, p, global_st)
        du[i, j, 1] = D1 *
                      (u[i - 1, j, 1] + u[i + 1, j, 1] + u[i, j + 1, 1] + u[i, j - 1, 1] -
                       4u[i, j, 1]) +
                      nn_out[1]
    end

    @inbounds for j in 2:(N - 1), i in 2:(N - 1)
        du[i, j, 2] = D2 *
                      (u[i - 1, j, 2] + u[i + 1, j, 2] + u[i, j + 1, 2] + u[i, j - 1, 2] -
                       4u[i, j, 2]) +
                      a * u[i, j, 1]^2 - β * u[i, j, 2]
    end

    @inbounds for j in 2:(N - 1)
        i = 1
        u_val = u[i, j, 1]
        v_val = u[i, j, 2]
        input = [u_val, v_val]
        nn_out, _ = Lux.apply(neural_network, input, p, global_st)
        du[1, j, 1] = D1 *
                      (2u[i + 1, j, 1] + u[i, j + 1, 1] + u[i, j - 1, 1] - 4u[i, j, 1]) +
                      nn_out[1]
    end
    @inbounds for j in 2:(N - 1)
        i = 1
        du[1, j, 2] = D2 *
                      (2u[i + 1, j, 2] + u[i, j + 1, 2] + u[i, j - 1, 2] - 4u[i, j, 2]) +
                      a * u[i, j, 1]^2 - β * u[i, j, 2]
    end
    @inbounds for j in 2:(N - 1)
        i = N
        u_val = u[i, j, 1]
        v_val = u[i, j, 2]
        input = [u_val, v_val]
        nn_out, _ = Lux.apply(neural_network, input, p, global_st)
        du[end, j, 1] = D1 *
                        (2u[i - 1, j, 1] + u[i, j + 1, 1] + u[i, j - 1, 1] - 4u[i, j, 1]) +
                                nn_out[1]
    end
    @inbounds for j in 2:(N - 1)
        i = N
        du[end, j, 2] = D2 *
                        (2u[i - 1, j, 2] + u[i, j + 1, 2] + u[i, j - 1, 2] - 4u[i, j, 2]) +
                        a * u[i, j, 1]^2 - β * u[i, j, 2]
    end

    @inbounds for i in 2:(N - 1)
        j = 1
        u_val = u[i, j, 1]
        v_val = u[i, j, 2]
        input = [u_val, v_val]
        nn_out, _ = Lux.apply(neural_network, input, p, global_st)
        du[i, 1, 1] = D1 *
                      (u[i - 1, j, 1] + u[i + 1, j, 1] + 2u[i, j + 1, 1] - 4u[i, j, 1]) +
                      nn_out[1]
    end
    @inbounds for i in 2:(N - 1)
        j = 1
        du[i, 1, 2] = D2 *
                      (u[i - 1, j, 2] + u[i + 1, j, 2] + 2u[i, j + 1, 2] - 4u[i, j, 2]) +
                      a * u[i, j, 1]^2 - β * u[i, j, 2]
    end
    @inbounds for i in 2:(N - 1)
        j = N
        u_val = u[i, j, 1]
        v_val = u[i, j, 2]
        input = [u_val, v_val]
        nn_out, _ = Lux.apply(neural_network, input, p, global_st)
        du[i, end, 1] = D1 *
                        (u[i - 1, j, 1] + u[i + 1, j, 1] + 2u[i, j - 1, 1] - 4u[i, j, 1]) +
                        nn_out[1]
    end
    @inbounds for i in 2:(N - 1)
        j = N
        du[i, end, 2] = D2 *
                        (u[i - 1, j, 2] + u[i + 1, j, 2] + 2u[i, j - 1, 2] - 4u[i, j, 2]) +
                        a * u[i, j, 1]^2 - β * u[i, j, 2]
    end

    @inbounds begin
        i = 1
        j = 1
        u_val = u[i, j, 1]
        v_val = u[i, j, 2]
        input = [u_val, v_val]
        nn_out, _ = Lux.apply(neural_network, input, p, global_st)
        du[1, 1, 1] = D1 * (2u[i + 1, j, 1] + 2u[i, j + 1, 1] - 4u[i, j, 1]) +
                      nn_out[1]
        du[1, 1, 2] = D2 * (2u[i + 1, j, 2] + 2u[i, j + 1, 2] - 4u[i, j, 2]) +
                      a * u[i, j, 1]^2 - β * u[i, j, 2]

        i = 1
        j = N
        u_val = u[i, j, 1]
        v_val = u[i, j, 2]
        input = [u_val, v_val]
        nn_out, _ = Lux.apply(neural_network, input, p, global_st)
        du[1, N, 1] = D1 * (2u[i + 1, j, 1] + 2u[i, j - 1, 1] - 4u[i, j, 1]) +
                      nn_out[1]
        du[1, N, 2] = D2 * (2u[i + 1, j, 2] + 2u[i, j - 1, 2] - 4u[i, j, 2]) +
                      a * u[i, j, 1]^2 - β * u[i, j, 2]

        i = N
        j = 1
        u_val = u[i, j, 1]
        v_val = u[i, j, 2]
        input = [u_val, v_val]
        nn_out, _ = Lux.apply(neural_network, input, p, global_st)
        du[N, 1, 1] = D1 * (2u[i - 1, j, 1] + 2u[i, j + 1, 1] - 4u[i, j, 1]) +
                      nn_out[1]
        du[N, 1, 2] = D2 * (2u[i - 1, j, 2] + 2u[i, j + 1, 2] - 4u[i, j, 2]) +
                      a * u[i, j, 1]^2 - β * u[i, j, 2]

        i = N
        j = N
        u_val = u[i, j, 1]
        v_val = u[i, j, 2]
        input = [u_val, v_val]
        nn_out, _ = Lux.apply(neural_network, input, p, global_st)
        du[end, end, 1] = D1 * (2u[i - 1, j, 1] + 2u[i, j - 1, 1] - 4u[i, j, 1]) +
                          nn_out[1]
        du[end, end, 2] = D2 * (2u[i - 1, j, 2] + 2u[i, j - 1, 2] - 4u[i, j, 2]) +
                          a * u[i, j, 1]^2 - β * u[i, j, 2]
    end
end

# Create wrapper for ODEProblem to work with Optimization.jl
const nstate = N*N*2
function ude_wrapper!(du, u, p, t)

    # take a view of just the real state‐portion
    du_state = @view du[1:nstate]
    u_state  = @view u[1:nstate]

    # reshape those into your 3-D arrays
    @views du3 = reshape(du_state, N, N, 2)
    @views u3  = reshape(u_state,  N, N, 2)

    # call your in-place UDE
    ude!(du3, u3, p, t)

    return nothing
end
```

## Loss Function and Optimization
The loss function is defined as:
```@example gray_scott
function loss_fn(θ)
    sol = solve(prob_ude, FBDF(), saveat=tsteps, p = θ, abstol=1e-6, reltol=1e-6)
    if sol.retcode != :Success
        println("Solver failed at some iteration.")
        return Inf
    end
    pred_u = [mean(reshape(s, N, N, 2)[:, :, 1]) for s in sol.u]
    loss = sum(abs2, pred_u .- data_u)
    return loss
end
```

The callback function is defined as:
```@example gray_scott
function training_callback(state, loss)
    push!(loss_hist, loss)
    epoch = length(loss_hist)
    println("Callback @ Epoch = $(epoch), Loss = $(round(loss, digits=5))")
    return false  # don't stop training
end
```
We reshape the initial condition tensor `r0` into a 1D vector to define the initial state of the ODE system. Then, we construct the `ODEProblem` using the UDE dynamics and extract the time series of average `U` and `V` concentrations from the ground-truth solution to serve as training targets.

```@example gray_scott
u0_vec = reshape(r0, :)
ode_fn = ODEFunction(ude_wrapper!, jac = nothing)
prob_ude = ODEProblem(ode_fn, u0_vec, (0.0, 10.0), ps)

using Statistics
using SciMLSensitivity
data_u = [mean(s[:, :, 1]) for s in solution_true.u]
data_v = [mean(s[:, :, 2]) for s in solution_true.u]
```

Finally, we create the optimization problem, and solve the UDE.

```@example gray_scott
using Optimization, OptimizationOptimisers, BSON, Zygote

# Define the OptimizationFunction using EnzymeVJP for AD
optf = OptimizationFunction((θ, p) -> loss_fn(θ), Optimization.AutoZygote())

# Create the optimization problem with current ps
optprob = OptimizationProblem(optf, ps)

# Solve using Adam
res = solve(optprob, Optimisers.Adam(0.003); maxiters=10000, callback=training_callback)
res.objective
```

## Results and conclusion
We can visualize the final results of the UDE and the ground-truth data with the following code.

```@example gray_scott
# Final model prediction and plot
sol_pred = solve(prob_ude, FBDF(), saveat=tsteps, p = res.u, abstol=1e-6, reltol=1e-6)
pred_u = [mean(s[:, :, 1]) for s in sol_pred.u]

using Plots
plot(tsteps, data_u; lw=2, label="True ⟨U⟩", color=:blue)
plot!(tsteps, pred_u; lw=2, label="UDE ⟨U⟩", color=:red, ls=:dash)
xlabel!("Time"); ylabel!("Mean U"); title!("Training Fit")
```

Now, as we can see, the UDE predictions match the ground-truth data very well, indicating the model has successfully learned the non-linear term.
