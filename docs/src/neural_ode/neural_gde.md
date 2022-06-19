# Neural Graph Differential Equations

This tutorial has been adapted from [here](https://github.com/CarloLucibello/GraphNeuralNetworks.jl/blob/master/examples/neural_ode_cora.jl).

In this tutorial we will use Graph Differential Equations (GDEs) to perform classification on the [CORA Dataset](https://relational.fit.cvut.cz/dataset/CORA). We shall be using the Graph Neural Networks primitives from the package [GraphNeuralNetworks](https://github.com/CarloLucibello/GraphNeuralNetworks.jl).

```julia
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
(; train_mask, val_mask, test_mask) = g.ndata
ytrain = y[:,train_mask]

Ã = normalized_adjacency(g, add_self_loops=true) |> device

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
```

# Step by Step Explanation

## Load the Required Packages

```julia
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

The dataset is available in the desired format in the GraphNeuralNetworks repository. We shall download the dataset from there.

```julia
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
(; train_mask, val_mask, test_mask) = g.ndata
ytrain = y[:,train_mask]

Ã = normalized_adjacency(g, add_self_loops=true) |> device
```

## Model and Data Configuration

We shall use only 16 hidden state dimensions.

```julia
nin = size(X, 1)
nhidden = 16
nout = length(classes)
epochs = 20
```

## Neural Graph Ordinary Differential Equations

Let us now define the final model. We will use a single layer GNN for approximating the gradients for the neural ODE. We use two additional `GCNConv` layers, one to project the data to a latent space and the other to project it from the latent space to the predictions. Finally a softmax layer gives us the probability of the input belonging to each target category.

```julia
diffeqarray_to_array(x) = reshape(cpu(x), size(x)[1:2])

node = NeuralODE(
    GCNConv(adj_mat, hidden=>hidden),
    (0.f0, 1.f0), Tsit5(), save_everystep = false,
    reltol = 1e-3, abstol = 1e-3, save_start = false
)

model = Flux.Chain(GCNConv(adj_mat, num_features=>hidden, relu),
              Flux.Dropout(0.5),
              node,
              diffeqarray_to_array,
              GCNConv(adj_mat, hidden=>target_catg))
```

## Training Configuration

### Loss Function and Accuracy

We shall be using the standard categorical crossentropy loss function which is used for multiclass classification tasks.

```julia
loss(x, y) = logitcrossentropy(model(x), y)
accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))
```

### Model Parameters

Now we extract the model parameters which we want to learn.

```julia
ps = Flux.params(model, node.p);
```

### Training Data

GNNs operate on an entire graph, so we can't do any sort of minibatching here. We need to pass the entire data in a single pass. So our dataset is an array with a single tuple.

```julia
train_data = [(train_X, train_y)]
```

### Optimizer

For this task we will be using the `ADAM` optimizer with a learning rate of `0.01`.

```julia
opt = ADAM(0.01)
```

### Callback Function

We also define a utility function for printing the accuracy of the model over time.

```julia
evalcb() = @show(accuracy(train_X, train_y))
```

## Training Loop

Finally, with the configuration ready and all the utilities defined we can use the `Flux.train!` function to learn the parameters `ps`. We run the training loop for `epochs` number of iterations.

```julia
for i = 1:epochs
    Flux.train!(loss, ps, train_data, opt, callback=throttle(evalcb, 10))
end
```

## Expected Output

```julia
accuracy(train_X, train_y) = 0.12370753323485968
accuracy(train_X, train_y) = 0.11632200886262925
accuracy(train_X, train_y) = 0.1189069423929099
accuracy(train_X, train_y) = 0.13404726735598227
accuracy(train_X, train_y) = 0.15620384047267355
accuracy(train_X, train_y) = 0.1776218611521418
accuracy(train_X, train_y) = 0.19793205317577547
accuracy(train_X, train_y) = 0.21122599704579026
accuracy(train_X, train_y) = 0.22673559822747416
accuracy(train_X, train_y) = 0.2429837518463811
accuracy(train_X, train_y) = 0.25406203840472674
accuracy(train_X, train_y) = 0.26809453471196454
accuracy(train_X, train_y) = 0.2869276218611521
accuracy(train_X, train_y) = 0.2961595273264402
accuracy(train_X, train_y) = 0.30797636632200887
accuracy(train_X, train_y) = 0.31831610044313147
accuracy(train_X, train_y) = 0.3257016248153619
accuracy(train_X, train_y) = 0.3378877400295421
accuracy(train_X, train_y) = 0.3500738552437223
accuracy(train_X, train_y) = 0.3629985228951256
accuracy(train_X, train_y) = 0.37259970457902514
accuracy(train_X, train_y) = 0.3777695716395864
accuracy(train_X, train_y) = 0.3895864106351551
accuracy(train_X, train_y) = 0.396602658788774
accuracy(train_X, train_y) = 0.4010339734121123
accuracy(train_X, train_y) = 0.40472673559822747
accuracy(train_X, train_y) = 0.41285081240768096
accuracy(train_X, train_y) = 0.422821270310192
accuracy(train_X, train_y) = 0.43057607090103395
accuracy(train_X, train_y) = 0.43833087149187594
accuracy(train_X, train_y) = 0.44645494830132937
accuracy(train_X, train_y) = 0.4538404726735598
accuracy(train_X, train_y) = 0.45901033973412114
accuracy(train_X, train_y) = 0.4630723781388479
accuracy(train_X, train_y) = 0.46971935007385524
accuracy(train_X, train_y) = 0.474519940915805
accuracy(train_X, train_y) = 0.47858197932053176
accuracy(train_X, train_y) = 0.4815361890694239
accuracy(train_X, train_y) = 0.4804283604135894
accuracy(train_X, train_y) = 0.4848596750369276
```
