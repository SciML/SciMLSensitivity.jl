# Partial Differential Equation (PDE) Constrained Optimization

This example uses a prediction model to optimize the one-dimensional Heat Equation.
(Step-by-step description below)

```@example pde
using DelimitedFiles,Plots
using DifferentialEquations, Optimization, OptimizationPolyalgorithms, Zygote, OptimizationOptimJL

# Problem setup parameters:
Lx = 10.0
x  = 0.0:0.01:Lx
dx = x[2] - x[1]
Nx = size(x)

u0 = exp.(-(x.-3.0).^2) # I.C

## Problem Parameters
p        = [1.0,1.0]    # True solution parameters
xtrs     = [dx,Nx]      # Extra parameters
dt       = 0.40*dx^2    # CFL condition
t0, tMax = 0.0 ,1000*dt
tspan    = (t0,tMax)
t        = t0:dt:tMax;

## Definition of Auxiliary functions
function ddx(u,dx)
    """
    2nd order Central difference for 1st degree derivative
    """
    return [[zero(eltype(u))] ; (u[3:end] - u[1:end-2]) ./ (2.0*dx) ; [zero(eltype(u))]]
end


function d2dx(u,dx)
    """
    2nd order Central difference for 2nd degree derivative
    """
    return [[zero(eltype(u))]; (u[3:end] - 2.0.*u[2:end-1] + u[1:end-2]) ./ (dx^2); [zero(eltype(u))]]
end

## ODE description of the Physics:
function heat(u,p,t)
    # Model parameters
    a0, a1 = p
    dx,Nx = xtrs #[1.0,3.0,0.125,100]
    return 2.0*a0 .* u +  a1 .* d2dx(u, dx)
end

# Testing Solver on linear PDE
prob = ODEProblem(heat,u0,tspan,p)
sol = solve(prob,Tsit5(), dt=dt,saveat=t);

plot(x, sol.u[1], lw=3, label="t0", size=(800,500))
plot!(x, sol.u[end],lw=3, ls=:dash, label="tMax")

ps  = [0.1, 0.2];   # Initial guess for model parameters
function predict(θ)
    Array(solve(prob,Tsit5(),p=θ,dt=dt,saveat=t))
end

## Defining Loss function
function loss(θ)
    pred = predict(θ)
    l = predict(θ)  - sol
    return sum(abs2, l), pred # Mean squared error
end

l,pred   = loss(ps)
size(pred), size(sol), size(t) # Checking sizes

LOSS  = []                              # Loss accumulator
PRED  = []                              # prediction accumulator
PARS  = []                              # parameters accumulator

callback = function (θ,l,pred) #callback function to observe training
  display(l)
  append!(PRED, [pred])
  append!(LOSS, l)
  append!(PARS, [θ])
  false
end

callback(ps,loss(ps)...) # Testing callback function

# Let see prediction vs. Truth
scatter(sol[:,end], label="Truth", size=(800,500))
plot!(PRED[end][:,end], lw=2, label="Prediction")

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p)->loss(x), adtype)

optprob = Optimization.OptimizationProblem(optf, ps)
res = Optimization.solve(optprob, PolyOpt(), callback = callback)
@show res.u # returns [0.999999999613485, 0.9999999991343996]
```

## Step-by-step Description

### Load Packages

```@example pde2
using DelimitedFiles,Plots
using DifferentialEquations, Optimization, OptimizationPolyalgorithms, 
      Zygote, OptimizationOptimJL
```

### Parameters

First, we setup the 1-dimensional space over which our equations will be evaluated.
`x` spans **from 0.0 to 10.0** in steps of **0.01**; `t` spans **from 0.00 to 0.04** in
steps of **4.0e-5**.

```@example pde2
# Problem setup parameters:
Lx = 10.0
x  = 0.0:0.01:Lx
dx = x[2] - x[1]
Nx = size(x)

u0 = exp.(-(x.-3.0).^2) # I.C

## Problem Parameters
p        = [1.0,1.0]    # True solution parameters
xtrs     = [dx,Nx]      # Extra parameters
dt       = 0.40*dx^2    # CFL condition
t0, tMax = 0.0 ,1000*dt
tspan    = (t0,tMax)
t        = t0:dt:tMax;
```

In plain terms, the quantities that were defined are:

- `x` (to `Lx`) spans the specified 1D space
- `dx` = distance between two points
- `Nx` = total size of space
- `u0` = initial condition
- `p` = true solution
- `xtrs` = convenient grouping of `dx` and `Nx` into Array
- `dt` = time distance between two points
- `t` (`t0` to `tMax`) spans the specified time frame
- `tspan` = span of `t`

### Auxiliary Functions

We then define two functions to compute the derivatives numerically. The **Central
Difference** is used in both the 1st and 2nd degree derivatives.

```@example pde2
## Definition of Auxiliary functions
function ddx(u,dx)
    """
    2nd order Central difference for 1st degree derivative
    """
    return [[zero(eltype(u))] ; (u[3:end] - u[1:end-2]) ./ (2.0*dx) ; [zero(eltype(u))]]
end


function d2dx(u,dx)
    """
    2nd order Central difference for 2nd degree derivative
    """
    return [[zero(eltype(u))]; (u[3:end] - 2.0.*u[2:end-1] + u[1:end-2]) ./ (dx^2); [zero(eltype(u))]]
end
```

### Heat Differential Equation

Next, we setup our desired set of equations in order to define our problem.

```@example pde2
## ODE description of the Physics:
function heat(u,p,t)
    # Model parameters
    a0, a1 = p
    dx,Nx = xtrs #[1.0,3.0,0.125,100]
    return 2.0*a0 .* u +  a1 .* d2dx(u, dx)
end
```

### Solve and Plot Ground Truth

We then solve and plot our partial differential equation. This is the true solution which we
will compare to further on.

```@example pde2
# Testing Solver on linear PDE
prob = ODEProblem(heat,u0,tspan,p)
sol = solve(prob,Tsit5(), dt=dt,saveat=t);

plot(x, sol.u[1], lw=3, label="t0", size=(800,500))
plot!(x, sol.u[end],lw=3, ls=:dash, label="tMax")
```

### Building the Prediction Model

Now we start building our prediction model to try to obtain the values `p`. We make an
initial guess for the parameters and name it `ps` here. The `predict` function is a
non-linear transformation in one layer using `solve`. If unfamiliar with the concept,
refer to [here](https://julialang.org/blog/2019/01/fluxdiffeq/).

```@example pde2
ps  = [0.1, 0.2];   # Initial guess for model parameters
function predict(θ)
    Array(solve(prob,Tsit5(),p=θ,dt=dt,saveat=t))
end
```

### Train Parameters

Training our model requires a **loss function**, an **optimizer** and a **callback
function** to display the progress.

#### Loss

We first make our predictions based on the current values of our parameters `ps`, then
take the difference between the predicted solution and the truth above. For the loss, we
use the **Mean squared error**.

```@example pde2
## Defining Loss function
function loss(θ)
    pred = predict(θ)
    l = predict(θ)  - sol
    return sum(abs2, l), pred # Mean squared error
end

l,pred   = loss(ps)
size(pred), size(sol), size(t) # Checking sizes
```

#### Optimizer

The optimizers `ADAM` with a learning rate of 0.01 and `BFGS` are directly passed in
training (see below)

#### Callback

The callback function displays the loss during training. We also keep a history of the
loss, the previous predictions and the previous parameters with `LOSS`, `PRED` and `PARS`
accumulators.

```@example pde2
LOSS  = []                              # Loss accumulator
PRED  = []                              # prediction accumulator
PARS  = []                              # parameters accumulator

callback = function (θ,l,pred) #callback function to observe training
  display(l)
  append!(PRED, [pred])
  append!(LOSS, l)
  append!(PARS, [θ])
  false
end

callback(ps,loss(ps)...) # Testing callback function
```

### Plotting Prediction vs Ground Truth

The scatter points plotted here are the ground truth obtained from the actual solution we
solved for above. The solid line represents our prediction. The goal is for both to overlap
almost perfectly when the PDE finishes its training and the loss is close to 0.

```@example pde2
# Let see prediction vs. Truth
scatter(sol[:,end], label="Truth", size=(800,500))
plot!(PRED[end][:,end], lw=2, label="Prediction")
```

### Train

The parameters are trained using `Optimization.solve` and adjoint sensitivities.
The resulting best parameters are stored in `res` and `res.u` returns the
parameters that minimizes the cost function.

```@example pde2
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p)->loss(x), adtype)

optprob = Optimization.OptimizationProblem(optf, ps)
res = Optimization.solve(optprob, PolyOpt(), callback = callback)
@show res.u # returns [0.999999999613485, 0.9999999991343996]
```

We successfully predict the final `ps` to be equal to **[0.999999999999975,
1.0000000000000213]** vs the true solution of `p` = **[1.0, 1.0]**