using SciMLSensitivity, OrdinaryDiffEq, DiffEqCallbacks, Lux, ComponentArrays
using Optimization, OptimizationOptimisers, Random, Test, Zygote

function test_hybridNODE(sensealg)
    Random.seed!(12345)
    datalength = 100
    tspan = (0.0, 100.0)
    t = range(tspan[1], tspan[2], length = datalength)
    target = 3.0 * (1:datalength) ./ datalength  # some dummy data to fit to
    cbinput = rand(1, datalength) #some external ODE contribution
    pmodel = Chain(Dense(2, 10, init_weight = zeros32), Dense(10, 2, init_weight = zeros32))
    ps, st = Lux.setup(Xoshiro(0), pmodel)
    ps = ComponentArray{Float64}(ps)
    dudt(u, p, t) = first(pmodel(u, p, st))

    # callback changes the first component of the solution every time
    # t is an integer
    function affect!(integrator, cbinput)
        event_index = round(Int, integrator.t)
        integrator.u[1] += 0.2 * cbinput[event_index]
    end
    callback = PresetTimeCallback(collect(1:datalength), (int) -> affect!(int, cbinput))

    # ODE with Callback
    prob = ODEProblem(dudt, [0.0, 1.0], tspan, ps)

    function predict_n_ode(p)
        arr = Array(solve(prob, Tsit5(),
            p = p, sensealg = sensealg, saveat = 2.0, callback = callback))[1, 2:2:end]
        return arr[1:datalength]
    end

    function loss_n_ode(p, _)
        pred = predict_n_ode(p)
        loss = sum(abs2, target .- pred) ./ datalength
    end

    cb = function (p, l) #callback function to observe training
        @show l
        return false
    end
    @show sensealg
    res = solve(OptimizationProblem(OptimizationFunction(loss_n_ode, AutoZygote()), ps),
        Adam(0.005); callback = cb, maxiters = 200)
    @test loss_n_ode(res.u, nothing) < 0.5
    println("  ")
end

function test_hybridNODE2(sensealg)
    Random.seed!(12345)
    u0 = Float32[2.0; 0.0; 0.0; 0.0]
    tspan = (0.0f0, 1.0f0)

    ## Get goal data
    function trueaffect!(integrator)
        integrator.u[3:4] = -3 * integrator.u[1:2]
    end
    function trueODEfunc(dx, x, p, t)
        @views dx[1:2] .= x[1:2] + x[3:4]
        dx[1] += x[2]
        dx[2] += x[1]
        dx[3:4] .= 0.0f0
    end
    cb_ = PeriodicCallback(trueaffect!, 0.1f0, save_positions = (true, true),
        initial_affect = true)
    prob = ODEProblem(trueODEfunc, u0, tspan)
    sol = solve(prob, Tsit5(), callback = cb_, save_everystep = false, save_start = true)
    ode_data = Array(sol)[1:2, 1:end]'

    ## Make model
    dudt2 = Chain(Dense(4, 50, tanh), Dense(50, 2))
    ps, st = Lux.setup(Xoshiro(0), dudt2)
    ps = ComponentArray{Float32}(ps)

    function affect!(integrator)
        integrator.u[3:4] = -3 * integrator.u[1:2]
    end
    function ODEfunc(dx, x, p, t)
        dx[1:2] .= first(dudt2(x, p, st))
        dx[3:4] .= 0.0f0
    end
    z0 = u0
    prob = ODEProblem(ODEfunc, z0, tspan)
    cb = PeriodicCallback(affect!, 0.1f0, save_positions = (true, true),
        initial_affect = true)

    ## Initialize learning functions
    function predict_n_ode(ps)
        Array(solve(prob, Tsit5(), u0 = z0, p = ps, callback = cb, save_everystep = false,
            save_start = true, sensealg = sensealg))[1:2, :]
    end
    function loss_n_ode(ps, _)
        pred = predict_n_ode(ps)[1:2, 1:end]'
        loss = sum(abs2, ode_data .- pred)
        loss
    end

    cba = function (p, loss)  #callback function to observe training
        @show loss
        return false
    end

    @show sensealg

    res = solve(OptimizationProblem(OptimizationFunction(loss_n_ode, AutoZygote()), ps),
        Adam(0.0025); callback = cba, maxiters = 200)

    @test loss_n_ode(res.u, nothing) < 0.5

    println("  ")
end

mutable struct Affect{T}
    callback_data::T
end
compute_index(t) = round(Int, t) + 1
function (cb::Affect)(integrator)
    indx = compute_index(integrator.t)
    integrator.u .= integrator.u .+
                    @view(cb.callback_data[:, indx, 1]) * (integrator.t - integrator.tprev)
end
function test_hybridNODE3(sensealg)
    u0 = Float32[2.0; 0.0]
    datasize = 101
    tspan = (0.0f0, 10.0f0)

    function trueODEfunc(du, u, p, t)
        du .= -u
    end
    t = range(tspan[1], tspan[2], length = datasize)
    prob = ODEProblem(trueODEfunc, u0, tspan)
    ode_data = Array(solve(prob, Tsit5(), saveat = t))

    true_data = reshape(ode_data, (2, length(t), 1))
    true_data = convert.(Float32, true_data)
    callback_data = true_data * 1.0f-3

    data = (true_data[:, :, 1], callback_data[:, :, 1])
    dudt2 = Chain(Dense(2, 50, tanh), Dense(50, 2))
    ps, st = Lux.setup(Xoshiro(0), dudt2)
    ps = ComponentArray{Float32}(ps)

    function dudt(du, u, p, t)
        du .= first(dudt2(u, p, st))
    end

    z0 = Float32[2.0; 0.0]
    prob = ODEProblem(dudt, z0, tspan)

    function callback_(callback_data)
        affect! = Affect(callback_data)
        condition(u, t, integrator) = integrator.t > 0
        DiscreteCallback(condition, affect!, save_positions = (false, false))
    end

    function predict_n_ode(p, true_data_0, callback_data, sense)
        _prob = remake(prob, p = p, u0 = true_data_0)
        solve(_prob, Tsit5(), callback = callback_(callback_data), saveat = t,
            sensealg = sense)
    end

    function loss_n_ode(p, (true_data, callback_data))
        sol = predict_n_ode(p, (vec(true_data[:, 1, :])), callback_data, sensealg)
        pred = Array(sol)
        loss = sum(abs2, true_data[:, :, 1] .- pred)
        loss
    end

    cba = function (p, loss)  #callback function to observe training
        @show loss
        return false
    end

    @show sensealg

    res = solve(
        OptimizationProblem(OptimizationFunction(loss_n_ode, AutoZygote()), ps,
            data),
        Adam(0.01); maxiters = 1000, callback = cba)
    loss = loss_n_ode(res.u, (true_data, callback_data))

    @test loss < 0.5
end

@testset "PresetTimeCallback: $(sensealg)" for sensealg in [ForwardDiffSensitivity(),
    BacksolveAdjoint(), InterpolatingAdjoint(), QuadratureAdjoint()]
    test_hybridNODE(sensealg)
end

@testset "PeriodicCallback: $(sensealg)" for sensealg in [ReverseDiffAdjoint(),
    BacksolveAdjoint(), InterpolatingAdjoint(), QuadratureAdjoint()]
    test_hybridNODE2(sensealg)
end

@testset "tprevCallback: $(sensealg)" for sensealg in [ReverseDiffAdjoint(),
    BacksolveAdjoint(), InterpolatingAdjoint(), QuadratureAdjoint()]
    test_hybridNODE3(sensealg)
end
