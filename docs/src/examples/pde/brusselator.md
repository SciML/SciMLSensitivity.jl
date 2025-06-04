# Learning Nonlinear Reaction Dynamics in the 2D Brusselator PDE Using Universal Differential Equations

## Introduction

The Brusselator is a mathematical model used to describe oscillating chemical reactions and spatial pattern formation, capturing how concentrations of chemical species evolve over time and space. In this documentation, we simulate the two-dimensional Brusselator partial differential equation (PDE) on a periodic square domain, generate time-resolved data using a finite difference discretization, and use this data to train a **Universal Differential Equation (UDE)**. Specifically, we replace the known nonlinear reaction term with a neural network, enabling us to learn complex dynamics directly from the generated data while preserving the known physical structure of the system.

## The Brusselator PDE

The Brusselator PDE is defined on a unit square periodic domain as follows:

$$
\frac{\partial U}{\partial t} = B + U^2V - (A+1)U + \alpha \nabla^2 U + f(x, y, t)
$$

$$
\frac{\partial V}{\partial t} = AU - U^2V + \alpha \nabla^2 V
$$

where $A=3.4, B=1$ and the forcing term is:

$$
f(x, y, t) =
\begin{cases}
5 & \text{if } (x - 0.3)^2 + (y - 0.6)^2 \leq 0.1^2 \text{ and } t \geq 1.1 \\
0 & \text{otherwise}
\end{cases}
$$

and the Laplacian operator is:

$$
\nabla^2 = \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}
$$

These equations are solved over the time interval:

$$
t \in [0, 11.5]
$$

with the initial conditions:

$$
U(x, y, 0) = 22 \cdot \left( y(1 - y) \right)^{3/2}
$$

$$
V(x, y, 0) = 27 \cdot \left( x(1 - x) \right)^{3/2}
$$

and the periodic boundary conditions:

$$
U(x + 1, y, t) = U(x, y, t)
$$

$$
V(x, y + 1, t) = V(x, y, t)
$$

## Numerical Discretization

To numerically solve this PDE, we discretize the unit square domain using $N$ grid points along each spatial dimension. The variables $U[i,j]$ and $V[i,j]$ then denote the concentrations at the grid point $(i, j)$ at a given time $t$.

We represent the spatially discretized fields as:

$$
U[i,j] = U(i \cdot \Delta x, j \cdot \Delta y), \quad V[i,j] = V(i \cdot \Delta x, j \cdot \Delta y),
$$

where  $\Delta x = \Delta y = \frac{1}{N}$ for a grid of size  $N \times N$. To organize the simulation state efficiently, we store both $ U $ and $ V $ in a single 3D array:

$$
u[i,j,1] = U[i,j], \quad u[i,j,2] = V[i,j],
$$

giving us a field tensor of shape $(N, N, 2)$. This structure is flexible and extends naturally to systems with additional field variables.

## Finite Difference Laplacian and Forcing


For spatial derivatives, we apply a second-order central difference scheme using a three-point stencil. The Laplacian is discretized as:

$$
[\ 1,\ -2,\ 1\ ]
$$

in both the $ x $ and $ y $ directions, forming a tridiagonal structure in both the x and y directions; applying this 1D stencil (scaled appropriately by $\frac{1}{Δx^2}$ or $\frac{1}{Δy^2}$) along each axis and summing the contributions yields the standard 5-point stencil computation for the 2D Laplacian. Periodic boundary conditions are incorporated by wrapping the stencil at the domain edges, effectively connecting the boundaries. The nonlinear interaction terms are computed directly at each grid point, making the implementation straightforward and local in nature.

## Generating Training Data

This provides us with an `ODEProblem` that can be solved to obtain training data. 

```@example bruss
using ComponentArrays, Random, Plots, OrdinaryDiffEq

N_GRID = 16
XYD = range(0f0, stop = 1f0, length = N_GRID)
dx = step(XYD)
T_FINAL = 11.5f0
SAVE_AT = 0.5f0
tspan = (0.0f0, T_FINAL)
t_points = range(tspan[1], stop=tspan[2], step=SAVE_AT)
A, B, alpha = 3.4f0, 1.0f0, 10.0f0

brusselator_f(x, y, t) = (((x - 0.3f0)^2 + (y - 0.6f0)^2) <= 0.01f0) * (t >= 1.1f0) * 5.0f0
limit(a, N) = a == 0 ? N : a == N+1 ? 1 : a

function init_brusselator(xyd)
    println("[Init] Creating initial condition array...")
    u0 = zeros(Float32, N_GRID, N_GRID, 2)
    for I in CartesianIndices((N_GRID, N_GRID))
        x, y = xyd[I[1]], xyd[I[2]]
        u0[I,1] = 22f0 * (y * (1f0 - y))^(3f0/2f0)
        u0[I,2] = 27f0 * (x * (1f0 - x))^(3f0/2f0)
    end
    println("[Init] Done.")
    return u0
end
u0 = init_brusselator(XYD)

function pde_truth!(du, u, p, t)
    A, B, alpha, dx = p
    αdx = alpha / dx^2
    for I in CartesianIndices((N_GRID, N_GRID))
        i, j = Tuple(I)
        x, y = XYD[i], XYD[j]
        ip1, im1 = limit(i+1, N_GRID), limit(i-1, N_GRID)
        jp1, jm1 = limit(j+1, N_GRID), limit(j-1, N_GRID)
        U, V = u[i,j,1], u[i,j,2]
        ΔU = u[im1,j,1] + u[ip1,j,1] + u[i,jp1,1] + u[i,jm1,1] - 4f0 * U
        ΔV = u[im1,j,2] + u[ip1,j,2] + u[i,jp1,2] + u[i,jm1,2] - 4f0 * V
        du[i,j,1] = αdx*ΔU + B + U^2 * V - (A+1f0)*U + brusselator_f(x, y, t)
        du[i,j,2] = αdx*ΔV + A*U - U^2 * V
    end
end

p_tuple = (A, B, alpha, dx)
@time sol_truth = solve(ODEProblem(pde_truth!, u0, tspan, p_tuple), FBDF(), saveat=t_points)
u_true = Array(sol_truth)
```

## Visualizing Mean Concentration Over Time

We can now use this code for training our UDE, and generating time-series plots of the concentrations of species of U and V using the code:
```@example bruss
using Plots, Statistics

# Compute average concentration at each timestep
avg_U = [mean(snapshot[:, :, 1]) for snapshot in sol_truth.u]
avg_V = [mean(snapshot[:, :, 2]) for snapshot in sol_truth.u]

# Plot average concentrations over time
plot(sol_truth.t, avg_U, label="Mean U", lw=2, xlabel="Time", ylabel="Concentration",
     title="Mean Concentration of U and V Over Time")
plot!(sol_truth.t, avg_V, label="Mean V", lw=2, linestyle=:dash)
```

With the ground truth data generated and visualized, we are now ready to construct a Universal Differential Equation (UDE) by replacing the nonlinear term  $U^2V$ with a neural network. The next section outlines how we define this hybrid model and train it to recover the reaction dynamics from data.

## Universal Differential Equation (UDE) Formulation

In the original Brusselator model, the nonlinear reaction term \( U^2V \) governs key dynamic behavior. In our UDE approach, we replace this known term with a trainable neural network \( \mathcal{N}_\theta(U, V) \), where \( \theta \) are the learnable parameters.

The resulting system becomes:

$$
\frac{\partial U}{\partial t} = 1 + \mathcal{N}_\theta(U, V) - 4.4U + \alpha \nabla^2 U + f(x, y, t)
$$

$$
\frac{\partial V}{\partial t} = 3.4U - \mathcal{N}_\theta(U, V) + \alpha \nabla^2 V
$$

Here, $\mathcal{N}_\theta(U, V)$ is trained to approximate the true interaction term $U^2V$ using simulation data. This hybrid formulation allows us to recover unknown or partially known physical processes while preserving the known structural components of the PDE.

First, we have to define and configure the neural network that has to be used for the training. The implementation for that is as follows:

```@example bruss
using Lux, Random, Optimization, OptimizationOptimJL, SciMLSensitivity, Zygote

model = Lux.Chain(Dense(2 => 16, tanh), Dense(16 => 1))
rng = Random.default_rng()
ps_init, st = Lux.setup(rng, model)
ps_init = ComponentArray(ps_init)
```

We use a simple fully connected neural network with one hidden layer of 16 tanh-activated units to approximate the nonlinear interaction term.

To ensure consistency between the ground truth simulation and the learned Universal Differential Equation (UDE) model, we preserve the same spatial discretization scheme used in the original ODEProblem. This includes:

* the finite difference Laplacian,
* periodic boundary conditions, and
* the external forcing function.

The only change lies in the replacement of the known nonlinear term $U^2V$ with a neural network approximation 
$\mathcal{N}_\theta(U, V)$. This design enables the UDE to learn complex or unknown dynamics from data while maintaining the underlying physical structure of the system.

The function below implements this hybrid formulation:
```@example bruss
function pde_ude!(du, u, ps_nn, t)
    αdx = alpha / dx^2
    for I in CartesianIndices((N_GRID, N_GRID))
        i, j = Tuple(I)
        x, y = XYD[i], XYD[j]
        ip1, im1 = limit(i+1, N_GRID), limit(i-1, N_GRID)
        jp1, jm1 = limit(j+1, N_GRID), limit(j-1, N_GRID)
        U, V = u[i,j,1], u[i,j,2]
        ΔU = u[im1,j,1] + u[ip1,j,1] + u[i,jp1,1] + u[i,jm1,1] - 4f0 * U
        ΔV = u[im1,j,2] + u[ip1,j,2] + u[i,jp1,2] + u[i,jm1,2] - 4f0 * V
        nn_val, _ = model([U, V], ps_nn, st)
        val = nn_val[1]
        du[i,j,1] = αdx*ΔU + B + val - (A+1f0)*U + brusselator_f(x, y, t)
        du[i,j,2] = αdx*ΔV + A*U - val
    end
end
prob_ude_template = ODEProblem(pde_ude!, u0, tspan, ps_init)
```
## Loss Function and Optimization
To train the neural network 
$\mathcal{N}_\theta(U, V)$ embedded in the UDE, we define a loss function that measures how closely the solution of the UDE matches the ground truth data generated earlier.

The loss is computed as the sum of squared errors between the predicted solution from the UDE and the true solution at each saved time point. If the solver fails (e.g., due to numerical instability or incorrect parameters), we return an infinite loss to discard that configuration during optimization. We use ```FBDF()``` as the solver due to the stiff nature of the brusselators euqation. Other solvers like ```KenCarp47()``` could also be used. 

To efficiently compute gradients of the loss with respect to the neural network parameters, we use an adjoint sensitivity method (`GaussAdjoint`), which performs high-accuracy quadrature-based integration of the adjoint equations. This approach enables scalable and memory-efficient training for stiff PDEs by avoiding full trajectory storage while maintaining accurate gradient estimates.

The loss function and initial evaluation are implemented as follows:

```@example bruss
println("[Loss] Defining loss function...")
function loss_fn(ps, _)
    prob = remake(prob_ude_template, p=ps)
    sol = solve(prob, FBDF(), saveat=t_points, sensealg=GaussAdjoint())
    # Failed solve 
    if !SciMLBase.successful_retcode(sol)
        return Inf32
    end
    pred = Array(sol)
    lval = sum(abs2, pred .- u_true) / length(u_true)
    return lval
end
```

Once the loss function is defined, we use the ADAM optimizer to train the neural network. The optimization problem is defined using SciML's ```Optimization.jl``` tools, and gradients are computed via automatic differentiation using ```AutoZygote()``` from ```SciMLSensitivity```:

```@example bruss
println("[Training] Starting optimization...")
using OptimizationOptimisers
optf = OptimizationFunction(loss_fn, AutoZygote())
optprob = OptimizationProblem(optf, ps_init)
loss_history = Float32[]


callback = (ps, l) -> begin
    push!(loss_history, l)
    println("Epoch $(length(loss_history)): Loss = $l")
    false
end
```

Finally to run everything:

```@example bruss
res = solve(optprob, Optimisers.Adam(0.01), callback=callback, maxiters=100)
```

```@example bruss
res.objective
```

```@example bruss
println("[Plot] Final U/V comparison plots...")
center = N_GRID ÷ 2
sol_final = solve(remake(prob_ude_template, p=res.u), FBDF(), saveat=t_points)
pred = Array(sol_final)

p1 = plot(t_points, u_true[center,center,1,:], lw=2, label="U True")
plot!(p1, t_points, pred[center,center,1,:], lw=2, ls=:dash, label="U Pred")
title!(p1, "Center U Concentration Over Time")

p2 = plot(t_points, u_true[center,center,2,:], lw=2, label="V True")
plot!(p2, t_points, pred[center,center,2,:], lw=2, ls=:dash, label="V Pred")
title!(p2, "Center V Concentration Over Time")

plot(p1, p2, layout=(1,2), size=(900,400))
```

## Results and Conclusion

After training the Universal Differential Equation (UDE), we compared the predicted dynamics to the ground truth for both chemical species.

The low training loss shows us that the neural network in the UDE was able to understand the underlying dynamics, and it was able to learn the $U^2V$ term in the partial differential equation. 
