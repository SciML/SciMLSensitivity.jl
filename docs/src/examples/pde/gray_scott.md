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
using DifferentialEquations
using Lux, Random, Optim, ComponentArrays, Statistics 
using LinearAlgebra
using Plots, SciMLBase
using Optimization, OptimizationOptimisers 
using SciMLSensitivity

const N = 32 # Smaller grid for faster training

# Constants for the "true" model
p_true = (a=1.0, α=1.0, ubar=1.0, β=10.0, D1=0.001, D2=0.1)

# Laplacian operator with Neumann (zero-flux) boundary conditions
Ax = Array(Tridiagonal([1.0 for i in 1:(N - 1)], [-2.0 for i in 1:N], [1.0 for i in 1:(N - 1)]))
Ay = copy(Ax)
Ax[1, 2] = 2.0
Ax[end, end - 1] = 2.0
Ay[1, 2] = 2.0
Ay[end, end - 1] = 2.0

# Initial condition
const uss = (p_true.ubar + p_true.β) / p_true.α
const vss = (p_true.a / p_true.β) * uss^2
const r0 = zeros(N, N, 2)
r0[:, :, 1] .= uss
r0[:, :, 2] .= vss
r0[div(N,2)-5:div(N,2)+5, div(N,2)-5:div(N,2)+5, 1] .+= 0.1 .* rand.()
r0[div(N,2)-5:div(N,2)+5, div(N,2)-5:div(N,2)+5, 2] .+= 0.1 .* rand.()
```

Having set up the grid, parameters, and initial condition, we now generate “ground truth” data by integrating the pure physics Gray–Scott model. This dataset will serve as the target that our UDE aims to learn.

```@example gray_scott
function true_model!(dr, r, p, t)
    a, α, ubar, β, D1, D2 = p
    u = @view r[:, :, 1]
    v = @view r[:, :, 2]
    Du = D1 .* (Ay * u + u * Ax)
    Dv = D2 .* (Ay * v + v * Ax)
    react_u = a .* u .* u ./ v .+ ubar .- α * u
    react_v = a .* u .* u .- β * v
    @. dr[:, :, 1] = Du + react_u
    @. dr[:, :, 2] = Dv + react_v
end

tspan = (0.0, 0.1)
tsteps = 0.0:0.01:0.1
prob_true = ODEProblem(true_model!, r0, tspan, p_true)
solution_true = solve(prob_true, Tsit5(), saveat=tsteps)
data_to_train = Array(solution_true)
```

With the ground-truth solutions computed, we can now proceed to visualize the spatiotemporal evolution of $u$ and $v$ on the grid.

```@example gray_scott
println("Plotting ground truth concentrations at the grid center...")
center = N ÷ 2
p1 = plot(tsteps, solution_true[center,center,1,:], lw=2, label="U True", color=:blue)
title!(p1, "Ground Truth: Center U")
xlabel!("Time")
ylabel!("Concentration")

p2 = plot(tsteps, solution_true[center,center,2,:], lw=2, label="V True", color=:red)
title!(p2, "Ground Truth: Center V")
xlabel!("Time")

p_center_combined = plot(p1, p2, layout=(1,2), size=(900,350))
display(p_center_combined)
```

## Defining the UDE
Now that we have an understanding of the data and its visualization, we can define the neural network and the UDE structure. We replace the $\frac{a u^2}{v} + \bar{u} - \alpha u$ term with a neural network, giving the resultant ODEs.

```math
du = D1 * (A_y u + u A_x) + \mathcal{N}_\theta(u,v)
```

```math
dv = D2 (A_y v + v A_x) + a u^2 + \beta v
````

The first step is to do data normalization, or compute the statistical properties of the dataset. We normalize the inputs to the neural network to make the training process more stable and efficient. Neural networks learn best when their input data is scaled to a consistent range, typically centered around zero, which helps prevent the gradients from becoming too large or too small during training. This leads to faster convergence and allows the optimizer to find a good solution more reliably.

```@example gray_scott
# Calculate mean and std dev for u and v across all space and time
u_data = @view data_to_train[:, :, 1, :]
v_data = @view data_to_train[:, :, 2, :]

u_mean = mean(u_data)
u_std = std(u_data)
v_mean = mean(v_data)
v_std = std(v_data)

norm_stats = (u_mean=u_mean, u_std=u_std, v_mean=v_mean, v_std=v_std)
```

Next, we define the neural network structure.
```@example gray_scott
rng = Random.default_rng()
nn = Lux.Chain(Lux.Dense(2 => 16, tanh), Lux.Dense(16 => 1))
p_nn, st_nn = Lux.setup(rng, nn)

# Add the normalization stats to the non-trainable parameters
p_ude = ComponentArray(
    p_physics=(β=p_true.β, D1=p_true.D1, D2=p_true.D2, a=p_true.a),
    p_nn=p_nn,
    p_norm=norm_stats # Add normalization stats here
)
```

The following code describes the UDE formulation with the neural network predictions embedded.

```@example gray_scott
function create_ude_model(nn_model, nn_state)
    function ude_model!(dr, r, p, t)
        β, D1, D2, a = p.p_physics
        u_mean, u_std, v_mean, v_std = p.p_norm
        
        u = @view r[:, :, 1]
        v = @view r[:, :, 2]
        Du = D1 .* (Ay * u + u * Ax)
        Dv = D2 .* (Ay * v + v * Ax)
        react_v = a .* u .* u .- β * v
        @. dr[:, :, 2] = Dv + react_v
        
        nn_reaction_u = similar(u)
        for i in 1:N, j in 1:N
            # Normalize the input to the NN
            u_norm = (u[i, j] - u_mean) / (u_std + 1f-8) # Add epsilon for stability
            v_norm = (v[i, j] - v_mean) / (v_std + 1f-8)
            input = [u_norm, v_norm]
            
            # The NN receives normalized data
            nn_reaction_u[i, j] = nn_model(input, p.p_nn, nn_state)[1][1]
        end
        @. dr[:, :, 1] = Du + nn_reaction_u
    end
    return ude_model!
end

ude_model! = create_ude_model(nn, st_nn)
prob_ude = ODEProblem(ude_model!, r0, tspan, p_ude)
```

## Loss Function and Optimization
The loss function is defined as:
```@example gray_scott
function loss(params_to_train)
    prediction = predict(params_to_train)
    if prediction.retcode != SciMLBase.ReturnCode.Success
        return Inf
    end
    return sum(abs2, Array(prediction) .- data_to_train)
end
```
The function used to predict is defined as below. We have explicitly defined the sensealg here to ensure the correct AD is used, which is the `QuadratureAdjoint` here. It is a method that provides a correct gradient without errors is infinitely better than a slightly faster one that fails. It represents a pragmatic engineering choice to ensure the optimization can proceed reliably.

```@example gray_scott
function predict(params_to_train)
    return solve(prob_ude, Tsit5(), p=params_to_train, saveat=solution_true.t,
                 sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true)))
end
```

## Training
Next, we start the training loop.
```@example gray_scott
loss_history = []
callback = function (p, l)
    push!(loss_history, l)
    println("Current loss: ", l)
    return false
end
```
Here, two training stages were used. The first stage uses a high learning rate to quickly move the model's parameters across the loss landscape towards the general area of a good solution. The second stage then uses a much lower learning rate to carefully fine-tune the parameters, allowing the model to settle precisely into a deep minimum without the risk of overshooting it. This two-phase approach combines the benefits of rapid initial progress with a stable and accurate final convergence.

```@example gray_scott
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((p_train, p) -> loss(p_train), adtype)
optprob = Optimization.OptimizationProblem(optf, p_ude)

println("Phase 1: Training with ADAM (learning rate 0.01)...")
result = Optimization.solve(optprob, ADAM(0.01), callback=callback, maxiters=100)

println("\nPhase 2: Refining with ADAM (learning rate 0.001)...")
optprob2 = Optimization.OptimizationProblem(optf, result.u)
result2 = Optimization.solve(optprob2, ADAM(0.001), callback=callback, maxiters=100)
println("Training complete.")
```

## Results and conclusion
We can visualize the final results of the UDE and the ground-truth data with the following code.

```@example gray_scott
p_trained = result2.u
final_prediction = predict(p_trained)

avg_u_true = [mean(data_to_train[:, :, 1, i]) for i in 1:length(tsteps)]
avg_v_true = [mean(data_to_train[:, :, 2, i]) for i in 1:length(tsteps)]
avg_u_pred = [mean(final_prediction[i][:, :, 1]) for i in 1:length(tsteps)]
avg_v_pred = [mean(final_prediction[i][:, :, 2]) for i in 1:length(tsteps)]

p_comp_u = plot(tsteps, avg_u_true, label="True", color=:blue, lw=2, title="Comparison: Avg. U")
plot!(tsteps, avg_u_pred, label="UDE", color=:black, linestyle=:dash, lw=2)
xlabel!("Time")
ylabel!("Avg. Concentration")

p_comp_v = plot(tsteps, avg_v_true, label="True", color=:red, lw=2, title="Comparison: Avg. V")
plot!(tsteps, avg_v_pred, label="UDE", color=:black, linestyle=:dash, lw=2)
 xlabel!("Time")

p_comp_combined = plot(p_comp_u, p_comp_v, layout=(1, 2), size=(900, 350))
display(p_comp_combined)
```

Now, as we can see, the UDE predictions match the ground-truth data very well, indicating the model has successfully learned the non-linear term.
