## Discovering Unknown Closure Term for Approximated Boussinesq Equation using Universal Partial Differential Equations

## Introduction
The Boussinesq equations, which are derived from simplifying incompressible Navier-Stokes equations, are often used in climate modelling. In this documentation, we solve the **Universal Partial Differential Equation (UPDE)** by training a neural network with generated data to discover the unknown function in the UPDE, instead of a conventional approach, which is to manually approximate the function by physical laws.


## The Approximated Boussinesq Equation without Closure
By an approximation of Boussinesq equations, we obtain a local advection-diffusion equation describing the evolution of the horizontally-averaged temperature $\overline{T}$:

$$\frac{\partial \overline{T}}{\partial t} + \frac{\partial \overline{wT}}{\partial z} = \kappa \frac{\partial^2 \overline{T}}{\partial z^2}$$

where $\overline{T}(z, t)$ is the horizontally-averaged temperature, $\kappa$ is the thermal diffusivity, and $\overline{wT}$ is the horizontal average temperature flux in the vertical direction.

Since $\overline{wT}$ is unknown, this one-dimensional approximating system is not closed. Instead of closing the system manually by determining an approximating $\overline{wT}$ from ad-hoc models, physical reasoning and scaling laws, we can use an UDE-automated approach to approximate $\overline{wT}$ from data. We let 

$$\overline{wT} = {U}_\theta \left( \mathbf{P}, \overline{T}, \frac{\partial \overline{T}}{\partial z} \right)$$

where $P$ are the physical parameters, $\overline{T}$ is the averaged temperature, and $\frac{\partial \overline{T}}{\partial z}$ is its gradient.


## Generating Data for Training

To train the neural network, we can generate data using the function: 

$$\overline{wT} = {cos(sin(T^3)) + sin(cos(T^2))}$$

with $N$ spatial points discretized by a finite difference method, with the time domain $t \in [0,1.5]$ and Neumann zero-flux boundary conditions, meaning $\frac{\partial \overline{T}}{\partial z} = 0$ at the edges.

```
using OrdinaryDiffEq, SciMLSensitivity
using Lux, Optimization, OptimizationOptimisers, ComponentArrays
using Random, Statistics, LinearAlgebra, Plots

const N = 32
const L = 1.0f0
const dx = L / N
const κ = 0.01f0
const tspan = (0.0f0, 1.5f0)

function true_flux_closure(T)
    cos(sin(T^3)) + sin(cos(T^2))
end

function boussinesq_pde!(du, u, p, t, flux_model)
    wT = flux_model(u, p)
    
    for i in 2:N-1
        diffusion = κ * (u[i+1] - 2u[i] + u[i-1]) / (dx^2)
        advection = (wT[i+1] - wT[i-1]) / (2dx)
        du[i] = diffusion - advection
    end
    
    du[1] = κ * (2(u[2] - u[1])) / (dx^2) 
    du[N] = κ * (2(u[N-1] - u[N])) / (dx^2)
end

u0 = Float32[exp(-(x-0.5)^2 / 0.1) for x in range(0, L, length=N)]

function pde_true!(du, u, p, t)
    wrapper(u_in, p_in) = true_flux_closure.(u_in)
    boussinesq_pde!(du, u, p, t, wrapper)
end

prob_true = ODEProblem(pde_true!, u0, tspan)
sol_true = solve(prob_true, Tsit5(), saveat=0.1)
training_data = Array(sol_true)
```


## Neural Network Setup and Training

We train a neural network with two hidden layers, each of size 16, and with tanh activation functions against time snapshots sampled every 0.1 seconds from the true PDE. 
```
rng = Random.default_rng()
nn = Lux.Chain(
    Lux.Dense(1, 16, tanh),
    Lux.Dense(16, 16, tanh),
    Lux.Dense(16, 1)
)
p_nn, st = Lux.setup(rng, nn)

function nn_closure(u, p)
    u_in = reshape(u, 1, length(u))
    pred = nn(u_in, p, st)[1]
    
    zero_in = reshape(u[1:1] .* 0, 1, 1)
    offset = nn(zero_in, p, st)[1]
    
    return vec(pred .- offset)
end

function pde_ude!(du, u, p, t)
    boussinesq_pde!(du, u, p, t, nn_closure)
end

prob_ude = ODEProblem(pde_ude!, u0, tspan, p_nn)
```

The ADAM optimizer is used to fit the UPDE. Learning rate $10^{−2}$ for 200 iterations and then ADAM with a learning rate of $10^{−3}$ for 1000 iterations.

```
function loss(p)
    sol_pred = solve(prob_ude, Tsit5(), p = p, saveat = 0.1, 
                     sensealg = InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)))
    
    if sol_pred.retcode != :Success
        return Inf
    end
    return mean(abs2, Array(sol_pred) .- training_data)
end

callback = function (p, l)
    println("Current Loss: $l")
    return false
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentVector(p_nn))

res1 = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.01), callback = callback, maxiters = 200)

optprob2 = Optimization.OptimizationProblem(optf, res1.u)
res2 = Optimization.solve(optprob2, OptimizationOptimisers.Adam(0.001), callback = callback, maxiters = 1000)

println("Final Loss: ", loss(res2.u))
```


## Visualization

We can now compare the results by plotting the trained data and the ground truth data.
```
final_prob = remake(prob_ude, p = res2.u)
sol_final = solve(final_prob, Tsit5(), saveat = 0.1)

p1 = plot(sol_true, color = :blue, alpha = 0.3, label = "", title = "Dynamics Reconstruction")
plot!(p1, sol_final, color = :red, linestyle = :dash, alpha = 0.5, label = "")
plot!(p1, [], color = :blue, label = "True Data")
plot!(p1, [], color = :red, linestyle = :dash, label = "Neural UDE")

T_range = range(0.0, 1.0, length = 100)

true_flux = true_flux_closure.(T_range)
true_flux = true_flux .- true_flux_closure(0.0)

learned_flux = [first(nn_closure([t], res2.u)) for t in T_range]

p2 = plot(T_range, true_flux, label = "True Physics (Centered)", 
          lw = 3, color = :blue, alpha = 0.6,
          title = "Discovered Physical Law (wT)",
          xlabel = "Temperature (T)", ylabel = "Flux (wT)")

plot!(p2, T_range, learned_flux, label = "Neural Network", 
      linestyle = :dash, lw = 3, color = :red)

display(plot(p1, p2, layout = (1,2), size = (900, 400)))
```


## Results and Conclusion
After training the neural network, and comparing the predicted data to the actual ground truth, the low training loss shows that the neural network was able to accurately learn the unknown flux closure term $\overline{wT}$ in the approximated Boussinesq equation.