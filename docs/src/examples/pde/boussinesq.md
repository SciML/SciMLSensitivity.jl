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

To train the neural network, we can generate data using the function $\overline{wT} = cos(sin(T^3)) + sin(cos(T^2))$ with $N$ spatial points discretized by a finite difference method, with the time domain $t \in [0,1.5]$ and Neumann zero-flux boundary conditions, meaning $\frac{\partial \overline{T}}{\partial z} = 0$ at the edges.


## Training Neural Network
We train a neural network with two hidden layers, each of size 8, and with tanh activation functions against 30 data points sampled from the true PDE. 

The ADAM optimizer is used to fit the UPDE. Learning rate $10^{−2}$ for 200 iterations and then ADAM with a learning rate of $10^{−3}$ for 1000 iterations.


## Results and Conclusion
