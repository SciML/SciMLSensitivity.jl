using DiffEqSensitivity, OrdinaryDiffEq, Flux, DiffEqFlux, CUDA, Zygote
CUDA.allowscalar(false) # Makes sure no slow operations are occuring

# Generate Data
u0 = Float32[2.0; 0.0]
datasize = 30
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2], length = datasize)
function trueODEfunc(du, u, p, t)
    true_A = Float32[-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end
prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
# Make the data into a GPU-based array if the user has a GPU
ode_data = gpu(solve(prob_trueode, Tsit5(), saveat = tsteps))


dudt2 = Chain(x -> x.^3,
              Dense(2, 50, tanh),
              Dense(50, 2)) |> gpu
u0 = Float32[2.0; 0.0] |> gpu

_p,re = Flux.destructure(dudt2)
p = gpu(_p)

prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps)

function predict_neuralode(p)
  gpu(prob_neuralode(u0,p))
end
function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss
end
# Callback function to observe training
list_plots = []
iter = 0

Zygote.gradient(loss_neuralode, p)