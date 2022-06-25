using OrdinaryDiffEq, Zygote, SciMLSensitivity, Test

p_model = [1f0]

u0 = Float32.([0.0])

function dudt(du, u, p, t)
    du[1] = p[1]
end

prob = ODEProblem(dudt,u0,(0f0,99.9f0))

function predict_neuralode(p)
    _prob = remake(prob,p=p)
    Array(solve(_prob,Tsit5(), saveat=0.1))
end

loss(p) = sum(abs2,predict_neuralode(p))/length(p)

p_model_ini = copy(p_model)

@test !iszero(Zygote.gradient(loss,p_model_ini)[1])

## https://github.com/SciML/SciMLSensitivity.jl/issues/675

u0 = Float32[2.0; 0.0] # Initial condition
p = [-0.1 2.0; -2.0 -0.1]

datasize = 30 # Number of data points
tspan = (0.0f0, 1.5f0) # Time range
# tsteps = range(tspan[1], tspan[2], length = datasize) # Split time range into equal steps for each data point
tsteps = (rand(datasize) .* (tspan[2] - tspan[1])  .+ tspan[1]) |> sort

function f(du,u,p,t)
    du .= p*u
end

function loss(p)
    prob = ODEProblem(f,u0,tspan,p)
    sol = solve(prob,Tsit5(),saveat=tsteps,sensealg=InterpolatingAdjoint())
    sum(sol)
end

Zygote.gradient(loss, p)[1]

@test !(Zygote.gradient(loss, p)[1] .|> iszero |> all)
