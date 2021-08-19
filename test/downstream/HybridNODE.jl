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
    @test loss_n_ode() < 0.4
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
