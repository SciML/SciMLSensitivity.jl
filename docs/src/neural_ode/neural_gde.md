# Neural Graph Differential Equations

This tutorial has been adapted from [here](https://github.com/CarloLucibello/GraphNeuralNetworks.jl/blob/master/examples/neural_ode_cora.jl).

In this tutorial we will use Graph Differential Equations (GDEs) to perform classification on the [CORA Dataset](https://relational.fit.cvut.cz/dataset/CORA). We shall be using the Graph Neural Networks primitives from the package [GraphNeuralNetworks](https://github.com/CarloLucibello/GraphNeuralNetworks.jl).

```@example graphneuralode_cp
# Load the packages
using GraphNeuralNetworks, DifferentialEquations
using DiffEqFlux: NeuralODE
using GraphNeuralNetworks.GNNGraphs: normalized_adjacency
using Lux, NNlib, Optimisers, Zygote, Random, ComponentArrays
using Lux: AbstractExplicitLayer, glorot_normal, zeros32
import Lux: initialparameters, initialstates
using DiffEqSensitivity
using Statistics: mean
using MLDatasets: Cora
using CUDA
CUDA.allowscalar(false)
device = CUDA.functional() ? gpu : cpu

# Download the dataset
dataset = Cora()

# Preprocess the data and compute adjacency matrix
classes = dataset.metadata["classes"]
g = mldataset2gnngraph(dataset) |> device
onehotbatch(data,labels)= device(labels).==reshape(data, 1,size(data)...)
onecold(y) =  map(argmax,eachcol(y))
X = g.ndata.features
y = onehotbatch(g.ndata.targets, classes) # a dense matrix is not the optimal, but we don't want to use Flux here

Ã = normalized_adjacency(g, add_self_loops=true) |> device

(; train_mask, val_mask, test_mask) = g.ndata
ytrain = y[:,train_mask]

# Model and Data Configuration
nin = size(X, 1)
nhidden = 16
nout = length(classes)
epochs = 20

# Define the graph neural network
struct ExplicitGCNConv{F1,F2,F3} <: AbstractExplicitLayer
    Ã::AbstractMatrix  # nomalized_adjacency matrix
    in_chs::Int
    out_chs::Int
    activation::F1
    init_weight::F2
    init_bias::F3
end

function Base.show(io::IO, l::ExplicitGCNConv)
    print(io, "ExplicitGCNConv($(l.in_chs) => $(l.out_chs)")
    (l.activation == identity) || print(io, ", ", l.activation)
    print(io, ")")
end

function initialparameters(rng::AbstractRNG, d::ExplicitGCNConv)
        return (weight=d.init_weight(rng, d.out_chs, d.in_chs),
                bias=d.init_bias(rng, d.out_chs, 1))
end

function ExplicitGCNConv(Ã, ch::Pair{Int,Int}, activation = identity;
                         init_weight=glorot_normal, init_bias=zeros32) 
    return ExplicitGCNConv{typeof(activation), typeof(init_weight), typeof(init_bias)}(Ã, first(ch), last(ch), activation, 
                                                                                       init_weight, init_bias)
end

function (l::ExplicitGCNConv)(x::AbstractMatrix, ps, st::NamedTuple)
    z = ps.weight * x * l.Ã
    return l.activation.(z .+ ps.bias), st
end

# Define the Neural GDE
function diffeqsol_to_array(x::ODESolution{T, N, <:AbstractVector{<:CuArray}}) where {T, N}
    return dropdims(gpu(x); dims=3)
end
diffeqsol_to_array(x::ODESolution) = dropdims(Array(x); dims=3)

# make NeuralODE work with Lux.Chain
# remove this once https://github.com/SciML/DiffEqFlux.jl/issues/727 is fixed
initialparameters(rng::AbstractRNG, node::NeuralODE) = initialparameters(rng, node.model) 
function initialstates(rng::AbstractRNG, node::NeuralODE)
    if  :layers ∈ propertynames(node.model)
        layers = node.model.layers
        return NamedTuple{keys(layers)}(initialstates.(rng, values(layers)))
    else
        return initialstates(node.model, rng)
    end
end

gnn = Chain(ExplicitGCNConv(Ã, nhidden => nhidden, relu),
            ExplicitGCNConv(Ã, nhidden => nhidden, relu))

node = NeuralODE(gnn, (0.f0, 1.f0), Tsit5(), save_everystep = false,
                 reltol = 1e-3, abstol = 1e-3, save_start = false)                

model = Chain(ExplicitGCNConv(Ã, nin => nhidden, relu),
              node,
              diffeqsol_to_array,
              Dense(nhidden, nout))

# Loss
logitcrossentropy(ŷ, y) = mean(-sum(y .* logsoftmax(ŷ); dims=1))

function loss(x, y, mask, model, ps, st)
    ŷ, st = model(x, ps, st)
    return logitcrossentropy(ŷ[:,mask], y), st
end

function eval_loss_accuracy(X, y, mask, model, ps, st)
    ŷ, _ = model(X, ps, st)
    l = logitcrossentropy(ŷ[:,mask], y[:,mask])
    acc = mean(onecold(ŷ[:,mask]) .== onecold(y[:,mask]))
    return (loss = round(l, digits=4), acc = round(acc*100, digits=2))
end

# Training
function train()
    ## Setup model 
    rng = Random.default_rng()
    Random.seed!(rng, 0)

    ps, st = Lux.setup(rng, model)
    ps = ComponentArray(ps) |> gpu
    st = st |> gpu

    ## Optimizer
    opt = Optimisers.ADAM(0.01f0)
    st_opt = Optimisers.setup(opt,ps)

    ## Training Loop
    for _ in 1:epochs
        (l,st), back = pullback(p->loss(X, ytrain, train_mask, model, p, st), ps)
        gs = back((one(l), nothing))[1]
        st_opt, ps = Optimisers.update(st_opt, ps, gs)
        @show eval_loss_accuracy(X, y, val_mask, model, ps, st)
    end
end

train()
```

# Step by Step Explanation

## Load the Required Packages

```@example graphneuralode
# Load the packages
using GraphNeuralNetworks, DifferentialEquations
using DiffEqFlux: NeuralODE
using GraphNeuralNetworks.GNNGraphs: normalized_adjacency
using Lux, NNlib, Optimisers, Zygote, Random, ComponentArrays
using Lux: AbstractExplicitLayer, glorot_normal, zeros32
import Lux: initialparameters, initialstates
using DiffEqSensitivity
using Statistics: mean
using MLDatasets: Cora
using CUDA
CUDA.allowscalar(false)
device = CUDA.functional() ? gpu : cpu
```

## Load the Dataset

The dataset is available in the desired format in the `MLDatasets` repository. We shall download the dataset from there.

```@example graphneuralode
dataset = Cora()
```

## Preprocessing the Data

Convert the data to `GNNGraph` and get the adjacency matrix from the graph `g`.

```julia
classes = dataset.metadata["classes"]
g = mldataset2gnngraph(dataset) |> device
onehotbatch(data,labels)= device(labels).==reshape(data, 1,size(data)...)
onecold(y) =  map(argmax,eachcol(y))
X = g.ndata.features
y = onehotbatch(g.ndata.targets, classes) # a dense matrix is not the optimal, but we don't want to use Flux here

Ã = normalized_adjacency(g, add_self_loops=true) |> device
```
### Training Data

GNNs operate on an entire graph, so we can't do any sort of minibatching here. We predict the entire dataset but train the model in a semi-supervised learning fashion. 
```@example graphneuralode
(; train_mask, val_mask, test_mask) = g.ndata
ytrain = y[:,train_mask]
```

## Model and Data Configuration

We shall use only 16 hidden state dimensions.

```julia
nin = size(X, 1)
nhidden = 16
nout = length(classes)
epochs = 20
```
## Define the Graph Neural Network

Here we define a type of graph neural networks called `GCNConv`. We use the name `ExplicitGCNConv` to avoid naming conflicts with `GraphNeuralNetworks`. For more informations on defining a layer with `Lux`, please consult to the [doc](http://lux.csail.mit.edu/dev/introduction/overview/#AbstractExplicitLayer-API).


```@example graphneuralode
struct ExplicitGCNConv{F1,F2,F3} <: AbstractExplicitLayer
    Ã::AbstractMatrix  # nomalized_adjacency matrix
    in_chs::Int
    out_chs::Int
    activation::F1
    init_weight::F2
    init_bias::F3
end

function Base.show(io::IO, l::ExplicitGCNConv)
    print(io, "ExplicitGCNConv($(l.in_chs) => $(l.out_chs)")
    (l.activation == identity) || print(io, ", ", l.activation)
    print(io, ")")
end

function initialparameters(rng::AbstractRNG, d::ExplicitGCNConv)
        return (weight=d.init_weight(rng, d.out_chs, d.in_chs),
                bias=d.init_bias(rng, d.out_chs, 1))
end

function ExplicitGCNConv(Ã, ch::Pair{Int,Int}, activation = identity;
                         init_weight=glorot_normal, init_bias=zeros32) 
    return ExplicitGCNConv{typeof(activation), typeof(init_weight), typeof(init_bias)}(Ã, first(ch), last(ch), activation, 
                                                                                       init_weight, init_bias)
end

function (l::ExplicitGCNConv)(x::AbstractMatrix, ps, st::NamedTuple)
    z = ps.weight * x * l.Ã
    return l.activation.(z .+ ps.bias), st
end
```

## Neural Graph Ordinary Differential Equations

Let us now define the final model. We will use two GNN layers for approximating the gradients for the neural ODE. We use one additional `GCNConv` layer to project the data to a latent space and the a `Dense` layer to project it from the latent space to the predictions. Finally a softmax layer gives us the probability of the input belonging to each target category.

```julia
function diffeqsol_to_array(x::ODESolution{T, N, <:AbstractVector{<:CuArray}}) where {T, N}
    return dropdims(gpu(x); dims=3)
end
diffeqsol_to_array(x::ODESolution) = dropdims(Array(x); dims=3)

# make NeuralODE work with Lux.Chain
# remove this once https://github.com/SciML/DiffEqFlux.jl/issues/727 is fixed
initialparameters(rng::AbstractRNG, node::NeuralODE) = initialparameters(rng, node.model) 
function initialstates(rng::AbstractRNG, node::NeuralODE)
    if  :layers ∈ propertynames(node.model)
        layers = node.model.layers
        return NamedTuple{keys(layers)}(initialstates.(rng, values(layers)))
    else
        return initialstates(node.model, rng)
    end
end

gnn = Chain(ExplicitGCNConv(Ã, nhidden => nhidden, relu),
            ExplicitGCNConv(Ã, nhidden => nhidden, relu))

node = NeuralODE(gnn, (0.f0, 1.f0), Tsit5(), save_everystep = false,
                 reltol = 1e-3, abstol = 1e-3, save_start = false)                

model = Chain(ExplicitGCNConv(Ã, nin => nhidden, relu),
              node,
              diffeqsol_to_array,
              Dense(nhidden, nout))
```

## Training Configuration

### Loss Function and Accuracy

We shall be using the standard categorical crossentropy loss function which is used for multiclass classification tasks.

```julia
logitcrossentropy(ŷ, y) = mean(-sum(y .* logsoftmax(ŷ); dims=1))

function loss(x, y, mask, model, ps, st)
    ŷ, st = model(x, ps, st)
    return logitcrossentropy(ŷ[:,mask], y), st
end

function eval_loss_accuracy(X, y, mask, model, ps, st)
    ŷ, _ = model(X, ps, st)
    l = logitcrossentropy(ŷ[:,mask], y[:,mask])
    acc = mean(onecold(ŷ[:,mask]) .== onecold(y[:,mask]))
    return (loss = round(l, digits=4), acc = round(acc*100, digits=2))
end
```

### Setup Model
We need to manually set up our mode with `Lux`, and convert the paramters to `ComponentArray` so that they can work well with sensitivity algorithms.
```
rng = Random.default_rng()
Random.seed!(rng, 0)

ps, st = Lux.setup(rng, model)
ps = ComponentArray(ps) |> gpu
st = st |> gpu
```
### Optimizer

For this task we will be using the `ADAM` optimizer with a learning rate of `0.01`.

```julia
opt = Optimisers.ADAM(0.01f0)
st_opt = Optimisers.setup(opt,ps)
```

## Training Loop

Finally, we use the package `Optimisers` to learn the parameters `ps`. We run the training loop for `epochs` number of iterations.

```julia
for _ in 1:epochs
    (l,st), back = pullback(p->loss(X, ytrain, train_mask, model, p, st), ps)
    gs = back((one(l), nothing))[1]
    st_opt, ps = Optimisers.update(st_opt, ps, gs)
    @show eval_loss_accuracy(X, y, val_mask, model, ps, st)
end
```

## Expected Output

```julia
eval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 1.9064f0, acc = 27.2)
eval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 1.8548f0, acc = 39.0)
eval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 1.7838f0, acc = 43.4)
eval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 1.7028f0, acc = 46.4)
eval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 1.6162f0, acc = 53.8)
eval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 1.5237f0, acc = 59.0)
eval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 1.4241f0, acc = 63.4)
eval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 1.3253f0, acc = 66.6)
eval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 1.2339f0, acc = 69.6)
eval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 1.1489f0, acc = 72.0)
eval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 1.0682f0, acc = 74.4)
eval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 0.995f0, acc = 75.4)
eval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 0.9304f0, acc = 76.2)
eval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 0.8787f0, acc = 76.0)
eval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 0.8413f0, acc = 76.4)
eval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 0.8195f0, acc = 76.8)
eval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 0.8076f0, acc = 77.2)
eval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 0.8022f0, acc = 77.0)
eval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 0.8046f0, acc = 77.2)
eval_loss_accuracy(X, y, val_mask, model, ps, st) = (loss = 0.8182f0, acc = 77.6)
```
