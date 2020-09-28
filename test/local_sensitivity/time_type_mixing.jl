using OrdinaryDiffEq, Zygote, DiffEqSensitivity, Test

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
