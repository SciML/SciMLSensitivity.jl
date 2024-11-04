# Faster Neural Ordinary Differential Equations with SimpleChains

[SimpleChains](https://github.com/PumasAI/SimpleChains.jl) has demonstrated performance boosts of ~5x and ~30x when compared to other mainstream deep learning frameworks like Pytorch for the training and evaluation in the specific case of small neural networks. For the nitty-gritty details, as well as, some SciML related videos around the need and applications of such a library, we can refer to this [blogpost](https://julialang.org/blog/2022/04/simple-chains/). As for doing Scientific Machine Learning, how do we even begin with training neural ODEs with any generic deep learning library?

## Training Data

First, we'll need data for training the NeuralODE, which can be obtained by solving the ODE `u' = f(u,p,t)` numerically using the SciML ecosystem in Julia.

```@example sc_neuralode
using SimpleChains,
      StaticArrays, OrdinaryDiffEq, SciMLSensitivity, OptimizationOptimisers, Plots

u0 = @SArray Float32[2.0, 0.0]
datasize = 30
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2], length = datasize)

function trueODE(u, p, t)
    true_A = @SMatrix Float32[-0.1 2.0; -2.0 -0.1]
    ((u .^ 3)'true_A)'
end

prob = ODEProblem(trueODE, u0, tspan)
data = Array(solve(prob, Tsit5(), saveat = tsteps))
```

## Neural Network

Next, we set up a small neural network. It will be trained to output the derivative of the solution at each time step given the value of the solution at the previous time step, and the parameters of the network. Thus, we are treating the neural network as a function `f(u,p,t)`. The difference is that instead of relying on knowing the exact equation for the ODE, we get to solve it only with the data.

```@example sc_neuralode
sc = SimpleChain(static(2),
    Activation(x -> x .^ 3),
    TurboDense{true}(tanh, static(50)),
    TurboDense{true}(identity, static(2)))

p_nn = Array(SimpleChains.init_params(sc))

f(u, p, t) = sc(u, p)
```

## NeuralODE, Prediction and Loss

Now instead of the function `trueODE(u,p,t)` in the first code block, we pass the neural network to the ODE solver. This is our NeuralODE. Now, in order to train it, we obtain predictions from the model and calculate the L2 loss against the data generated numerically previously.

```@example sc_neuralode
prob_nn = ODEProblem(f, u0, tspan)

function predict_neuralode(p)
    Array(solve(prob_nn, Tsit5(); p = p, saveat = tsteps,
        sensealg = QuadratureAdjoint(autojacvec = ZygoteVJP())))
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, data .- pred)
    return loss
end
```

## Training

The next step is to minimize the loss, so that the NeuralODE gets trained. But in order to be able to do that, we have to be able to backpropagate through the NeuralODE model. Here the backpropagation through the neural network is the easy part, and we get that out of the box with any deep learning package(although not as fast as SimpleChains for the small nn case here). But we have to find a way to first propagate the sensitivities of the loss back, first through the ODE solver and then to the neural network.

The adjoint of a neural ODE can be calculated through the various AD algorithms available in SciMLSensitivity.jl. But working with [StaticArrays](https://juliaarrays.github.io/StaticArrays.jl/stable/) in SimpleChains.jl requires a special adjoint method as StaticArrays do not allow any mutation. All the adjoint methods make heavy use of in-place mutation to be performant with the heap allocated normal arrays. For our statically sized, stack allocated StaticArrays, in order to be able to compute the ODE adjoint we need to do everything out of place. Hence, we have specifically used `QuadratureAdjoint(autojacvec=ZygoteVJP())` adjoint algorithm in the solve call inside `predict_neuralode(p)` which computes everything out-of-place when u0 is a StaticArray. Hence, we can move forward with the training of the NeuralODE

```@example sc_neuralode
callback = function (state, l; doplot = true)
    display(l)
    pred = predict_neuralode(state.u)
    plt = scatter(tsteps, data[1, :], label = "data")
    scatter!(plt, tsteps, pred[1, :], label = "prediction")
    if doplot
        display(plot(plt))
    end
    return false
end

optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x),
    Optimization.AutoZygote())
optprob = Optimization.OptimizationProblem(optf, p_nn)

res = Optimization.solve(optprob, Adam(0.05), callback = callback, maxiters = 300)
```
