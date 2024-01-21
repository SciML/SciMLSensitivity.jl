using SciMLSensitivity, OrdinaryDiffEq, Lux, DiffEqFlux, LuxCUDA, Zygote, Random
using ComponentArrays
CUDA.allowscalar(false) # Makes sure no slow operations are occurring

const gdev = gpu_device()
const cdev = cpu_device()

# Generate Data
u0 = Float32[2.0; 0.0]
datasize = 30
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2], length = datasize)
function trueODEfunc(du, u, p, t)
    true_A = Float32[-0.1 2.0; -2.0 -0.1]
    du .= ((u .^ 3)'true_A)'
end
prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
# Make the data into a GPU-based array if the user has a GPU
ode_data = gdev(solve(prob_trueode, Tsit5(), saveat = tsteps))

dudt2 = Chain(x -> x .^ 3, Dense(2, 50, tanh), Dense(50, 2))
u0 = Float32[2.0; 0.0] |> gdev

prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps)
ps, st = Lux.setup(Random.default_rng(), dudt2)
ps = ComponentArray(ps) |> gdev

function predict_neuralode(p)
    gdev(first(prob_neuralode(u0, p, st)))
end
function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss
end

Zygote.gradient(loss_neuralode, ps)
