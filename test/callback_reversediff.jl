using OrdinaryDiffEq, Lux, ComponentArrays, SciMLSensitivity, DiffEqCallbacks, Test
using Optimization, OptimizationOptimisers, Zygote
using Random

Random.seed!(1234)

u0 = Float32[2.0; 0.0]
datasize = 100
tspan = (0.0f0, 10.5f0)
dosetimes = [1.0, 2.0, 4.0, 8.0]

function affect!(integrator)
    integrator.u = integrator.u .+ 1
end

function functionCalling(x, t, integrator)
    # @info "Step: $(t)"
end

cbPreTime = PresetTimeCallback(dosetimes, affect!, save_positions = (false, false))
cbFctCall = FunctionCallingCallback(functionCalling; func_everystep = true,
    func_start = true)

function trueODEfunc(du, u, p, t)
    du .= -u
end
t = range(tspan[1], tspan[2], length = datasize)

prob = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob, Tsit5(), callback = CallbackSet(cbPreTime, cbFctCall),
    saveat = t))
dudt2 = Chain(Dense(2, 50, tanh), Dense(50, 2))
ps, st = Lux.setup(Random.default_rng(), dudt2)
ps = ComponentArray(ps)

function dudt(du, u, p, t)
    du[1:2] .= -u[1:2]
    du[3:end] .= first(dudt2(u[1:2], p, st)) #re(p)(u[3:end])
end
z0 = Float32[u0; u0]
prob = ODEProblem(dudt, z0, tspan)

affect!(integrator) = integrator.u[1:2] .= integrator.u[3:end]
cb = PresetTimeCallback(dosetimes, affect!, save_positions = (false, false))

function predict_n_ode(ps)
    Array(solve(prob, Tsit5(), u0 = z0, p = ps, callback = cb, saveat = t,
        sensealg = ReverseDiffAdjoint()))[1:2, :]
end

function loss_n_ode(ps, _)
    pred = predict_n_ode(ps)
    loss = sum(abs2, ode_data .- pred)
    loss
end
loss_n_ode(ps, nothing)

cb1 = function (p, l)
    @show l
    return false
end

res = solve(OptimizationProblem(OptimizationFunction(loss_n_ode, AutoZygote()), ps),
    Adam(0.005); callback = cb1, maxiters = 250)

@test loss_n_ode(res.u, nothing) < 0.4
