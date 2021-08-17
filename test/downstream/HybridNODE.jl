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

@testset "PresetTimeCallback" begin
    test_hybridNODE(ForwardDiffSensitivity())
    test_hybridNODE(BacksolveAdjoint())
    test_hybridNODE(InterpolatingAdjoint())
    test_hybridNODE(QuadratureAdjoint())
end
