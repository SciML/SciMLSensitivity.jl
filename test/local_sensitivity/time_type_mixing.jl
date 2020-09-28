using OrdinaryDiffEq, Zygote, DiffEqSensitivity, Test

# excitation of the system
no_samples = 1000

datasize = 1000
tspan = (0.0f0, 99.9f0)
tsteps = range(tspan[1], tspan[2], length = datasize)

ex = t->sin(t)
u = ex.(tsteps)

f(x) = (atan(8.0 * x - 4.0) + atan(4.0)) / (2.0 * atan(4.0))
function hammerstein_system(u)
    y= zeros(size(u))
    for k in 2:length(u)
        y[k] = 0.2 * f(u[k-1]) + 0.8 * y[k-1]
    end
    return y
end

y = Float32.(hammerstein_system(ex.(tsteps)))

nn_model = FastChain(FastDense(1,50, sigmoid), FastDense(50,50, sigmoid), FastDense(50, 1))

p_model = initial_params(nn_model)

u0 = Float32.([0.0])

function dudt(du, u, p, t)
    du[1] = nn_model(ex(t),p)[1]
end

prob = ODEProblem(dudt,u0,tspan)

function predict_neuralode(p)
    _prob = remake(prob,p=p)
    Array(solve(_prob,Tsit5(), saveat=0.1))
end

loss(p) = sum(abs2,y .- predict_neuralode(p))/length(y)

p_model_ini = copy(p_model)

@test !iszero(Flux.gradient(loss,p_model_ini)[1])
