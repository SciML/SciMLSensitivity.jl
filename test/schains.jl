using OrdinaryDiffEq, DiffEqSensitivity, SimpleChains, Optimization, OptimizationFlux, Zygote, Test

u0 = Float32[2.0, 0.0]
datasize = 30
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2], length = datasize)

function trueODE!(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end

prob = ODEProblem(trueODE!, u0, tspan)
data = Array(solve(prob, Tsit5(), saveat = tsteps))

sc = SimpleChain(
                static(2),
                Activation(x -> x.^3),
                TurboDense{true}(tanh, static(50)),
                TurboDense{true}(identity, static(2))
            )
p_nn = SimpleChains.init_params(sc)

f(u,p,t) = sc(u,p) #oop

prob_nn = ODEProblem(f, u0, tspan)

function predict_neuralode(p)
    Array(solve(prob_nn, Tsit5();p=p,saveat=tsteps,sensealg=QuadratureAdjoint()))
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, data .- pred)
    return loss, pred
end

callback = function (p, l)
    display(l)
    return false
end

l1 = loss_neuralode(p_nn)
optf = Optimization.OptimizationFunction((x,p)->loss_neuralode(x), Optimization.AutoZygote())
optprob = Optimization.OptimizationProblem(optf, p_nn)

res = Optimization.solve(optprob, ADAM(0.05),callback=callback,maxiters=300)

l = loss_neuralode(res.minimizer)

@info:"Regression Test"

@test l < l1
@test l < 4.0

g1 = Zygote.gradient((u, p)->Array(solve(prob_nn,Tsit5();u0=u,p=p,saveat=tsteps,sensealg=QuadratureAdjoint())), u0, p_nn)
g2 = Zygote.gradient((u, p)->Array(solve(prob_nn,Tsit5();u0=u,p=p,saveat=tsteps,sensealg=ForwardDiffSensitivity())), u0, p_nn)

@test g1[1] ≈ g[2] rtol=1e-6
@test g2[1] ≈ g[2] rtol=1e-6
