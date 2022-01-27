using DiffEqSensitivity, OrdinaryDiffEq, DiffEqCallbacks, DiffEqFlux, Flux
using Random, Test

function test_hybridNODE(sensealg)
    Random.seed!(12345)
    datalength = 100
    tspan = (0.0,100.0)
    t = range(tspan[1],tspan[2],length=datalength)
    target = 3.0*(1:datalength)./datalength  # some dummy data to fit to
    cbinput = rand(1, datalength) #some external ODE contribution
    pmodel = Chain(
        Dense(2, 10, initW=zeros),
        Dense(10, 2, initW=zeros))
    p, re = Flux.destructure(pmodel)
    dudt(u,p,t) = re(p)(u)

    # callback changes the first component of the solution every time
    # t is an integer
    function affect!(integrator, cbinput)
        event_index = round(Int,integrator.t)
        integrator.u[1] += 0.2*cbinput[event_index]
    end
    callback = PresetTimeCallback(collect(1:datalength),(int)->affect!(int, cbinput))

    # ODE with Callback
    prob = ODEProblem(dudt,[0.0, 1.0],tspan,p)

    function predict_n_ode(p)
        arr = Array(solve(prob, Tsit5(),
            p=p, sensealg=sensealg, saveat=2.0, callback=callback))[1,2:2:end]
        return arr[1:datalength]
    end

    function loss_n_ode()
        pred = predict_n_ode(p)
        loss = sum(abs2,target .- pred)./datalength
    end

    cb = function () #callback function to observe training
        pred = predict_n_ode(p)
        display(loss_n_ode())
    end
    @show sensealg
    Flux.train!(loss_n_ode, Flux.params(p), Iterators.repeated((), 20), ADAM(0.005), cb = cb)
    @test loss_n_ode() < 0.5
    println("  ")
end

function test_hybridNODE2(sensealg)
    Random.seed!(12345)
    u0 = Float32[2.; 0.; 0.; 0.]
    tspan = (0f0,1f0)

    ## Get goal data
    function trueaffect!(integrator)
        integrator.u[3:4] = -3*integrator.u[1:2]
    end
    function trueODEfunc(dx,x,p,t)
        @views dx[1:2] .= x[1:2] + x[3:4]
        dx[1] += x[2]
        dx[2] += x[1]
        dx[3:4] .= 0f0
    end
    cb_ = PeriodicCallback(trueaffect!,0.1f0,save_positions=(true,true),initial_affect=true)
    prob = ODEProblem(trueODEfunc,u0,tspan)
    sol = solve(prob,Tsit5(),callback=cb_,save_everystep=false,save_start=true)
    ode_data = Array(sol)[1:2,1:end]'

    ## Make model
    dudt2 = Chain(Dense(4,50,tanh),
                    Dense(50,2))
    p,re = Flux.destructure(dudt2) # use this p as the initial condition!
    function affect!(integrator)
        integrator.u[3:4] = -3*integrator.u[1:2]
    end
    function ODEfunc(dx,x,p,t)
        dx[1:2] .= re(p)(x)
        dx[3:4] .= 0f0
    end
    z0 = u0
    prob = ODEProblem(ODEfunc,z0,tspan)
    cb = PeriodicCallback(affect!,0.1f0,save_positions=(true,true),initial_affect=true)

    ## Initialize learning functions
    function predict_n_ode()
        _prob = remake(prob,p=p)
        Array(solve(_prob,Tsit5(),u0=z0,p=p,callback=cb,save_everystep=false,save_start=true,sensealg=sensealg))[1:2,:]
    end
    function loss_n_ode()
        pred = predict_n_ode()[1:2,1:end]'
        loss = sum(abs2,ode_data .- pred)
        loss
    end
    loss_n_ode() # n_ode.p stores the initial parameters of the neural ODE
    cba = function ()  #callback function to observe training
        pred = predict_n_ode()[1:2,1:end]'
        display(sum(abs2,ode_data .- pred))
        return false
    end
    cba()

    ## Learn
    ps = Flux.params(p)
    data = Iterators.repeated((), 25)

    @show sensealg

    Flux.train!(loss_n_ode, ps, data, ADAM(0.0025), cb = cba)

    @test loss_n_ode() < 0.5

    println("  ")
end

mutable struct Affect{T}
      callback_data::T
end
function (cb::Affect)(integrator)
    integrator.u .= integrator.u .+ cb.callback_data[:,Int(integrator.t÷60+1),1] * (integrator.t-integrator.tprev)
end
function test_hybridNODE3(sensealg)
    u0 = Float32[2.; 0.]
    datasize = 100
    tspan = (0.0f0,10.5f0)
    
    function trueODEfunc(du,u,p,t)
        du .= -u
    end
    t = range(tspan[1],tspan[2],length=datasize)
    prob = ODEProblem(trueODEfunc,u0,tspan)
    ode_data = Array(solve(prob,Tsit5(),saveat=t))
    
    ode_data = hcat(ode_data,ode_data,ode_data,ode_data)
    true_data = reshape(ode_data,(2,100,4))
    true_data = convert.(Float32,true_data)
    callback_data = true_data .* 1e-3 
    callback_data = convert.(Float32,callback_data)
    train_dataloader = Flux.Data.DataLoader((true_data = true_data,callback_data = callback_data),batchsize=1)
    dudt2 = Chain(Dense(2,50,tanh),
                 Dense(50,2))
    p,re = Flux.destructure(dudt2)
    function dudt(du,u,p,t)
        du .= re(p)(u) 
    end
    z0 = Float32[2.; 0.]
    prob = ODEProblem(dudt,z0,tspan)
    
    
    function callback_(callback_data)
        affect! = Affect(callback_data)
        condition(u,t,integrator) = integrator.t > 0 
        DiscreteCallback(condition,affect!,save_positions=(false,false))
    end
    
    function predict_n_ode(true_data_0,callback_data, sense)
        _prob = remake(prob,p=p)
        Array(solve(_prob,Tsit5(),u0=true_data_0,p=p,callback=callback_(callback_data),saveat=t,sensealg=sense))
    end
    
    function loss_n_ode(true_data,callback_data)
        pred = predict_n_ode((vec(true_data[:,1,:])),callback_data,sensealg)
        loss = Flux.mse(true_data,pred)
        loss
    end

    for (true_data,callback_data) in train_dataloader
        display(loss_n_ode(true_data,callback_data))
    end

    ps = Flux.params(p)
    opt = ADAM(0.01)
    epochs= 4

    function cb1() 
        display(loss_n_ode(train_dataloader.data.true_data,train_dataloader.data.callback_data))
    end
    cb1()

    function train!(loss, ps, data, opt)
        ps = Params(ps)
        for (true_data,callback_data) in data
            gs = gradient(ps) do
                loss(true_data,callback_data)
            end
            Flux.update!(opt, ps, gs)      
        end
    end

    @Flux.epochs epochs train!(loss_n_ode, Params(ps),train_dataloader, opt)
end

@testset "PresetTimeCallback" begin
    test_hybridNODE(ForwardDiffSensitivity())
    test_hybridNODE(BacksolveAdjoint())
    test_hybridNODE(InterpolatingAdjoint())
    test_hybridNODE(QuadratureAdjoint())
end

@testset "PeriodicCallback" begin
    test_hybridNODE2(ReverseDiffAdjoint())
    test_hybridNODE2(BacksolveAdjoint())
    test_hybridNODE2(InterpolatingAdjoint())
    test_hybridNODE2(QuadratureAdjoint())
end

@testset "tprevCallback" begin
    test_hybridNODE3(ReverseDiffAdjoint())
    test_hybridNODE3(BacksolveAdjoint())
    test_hybridNODE3(InterpolatingAdjoint())
    test_hybridNODE3(QuadratureAdjoint())
end