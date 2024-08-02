using OrdinaryDiffEq, SciMLSensitivity

function growth(du, u, p, t)
    @. du = p * u * (1 - u)
end
u0 = [0.1]
tspan = (0.0, 2.0)
prob = ODEProblem(growth, u0, tspan, [1.0])
sol = solve(prob, Tsit5(), reltol = 1e-8, abstol = 1e-8)

savetimes = [0.0, 1.0, 1.9]

function f(a)
    _prob = remake(prob, p = [a[1]], saveat = savetimes)
    predicted = solve(_prob, Tsit5(), sensealg = InterpolatingAdjoint(), abstol = 1e-12,
        reltol = 1e-12)
    sum(predicted.u[end])
end

function f2(a)
    _prob = remake(prob, p = [a[1]], saveat = savetimes)
    predicted = solve(_prob, Tsit5(), sensealg = InterpolatingAdjoint(), abstol = 1e-12,
        reltol = 1e-12)
    sum(predicted.u[end])
end

using Zygote
a = ones(3)
@test Zygote.gradient(f, a)[1][1] ≈ Zygote.gradient(f2, a)[1][1]
@test Zygote.gradient(f, a)[1][2] == Zygote.gradient(f2, a)[1][2] == 0
@test Zygote.gradient(f, a)[1][3] == Zygote.gradient(f2, a)[1][3] == 0

# callback in problem construction or in solve call should give same result
# https://github.com/SciML/SciMLSensitivity.jl/issues/1081
odef(du, u, p, t) = du .= u .* p
prob = ODEProblem(odef, [2.0], (0.0, 1.0), [3.0])

let callback_count1 = 0, callback_count2 = 0
    function f1(u0p, adjoint_type)
        condition(u, t, integrator) = t == 0.5
        affect!(integrator) = callback_count1 += 1
        cb = DiscreteCallback(condition, affect!)
        prob = ODEProblem{true}(odef, u0p[1:1], (0.0, 1.0), u0p[2:2]; callback = cb)
        sum(solve(prob, Tsit5(), tstops = [0.5], sensealg = adjoint_type))
    end

    function f2(u0p, adjoint_type)
        condition(u, t, integrator) = t == 0.5
        affect!(integrator) = callback_count2 += 1
        cb = DiscreteCallback(condition, affect!)
        prob = ODEProblem{true}(odef, u0p[1:1], (0.0, 1.0), u0p[2:2])
        sum(solve(prob, Tsit5(), tstops = [0.5], callback = cb, sensealg = adjoint_type))
    end

    @testset "Callback duplication check" begin
        u0p = [2.0, 3.0]
        for adjoint_type in [
            ForwardDiffSensitivity(), ReverseDiffAdjoint(), TrackerAdjoint(),
            BacksolveAdjoint(), InterpolatingAdjoint(), QuadratureAdjoint(), GaussAdjoint()]
            count1 = 0
            count2 = 0
            if adjoint_type == GaussAdjoint()
                @test_broken Zygote.gradient(x -> f1(x, adjoint_type), u0p) ==
                             Zygote.gradient(x -> f2(x, adjoint_type), u0p)
            else
                @test Zygote.gradient(x -> f1(x, adjoint_type), u0p) ==
                      Zygote.gradient(x -> f2(x, adjoint_type), u0p)
                @test callback_count1 == callback_count2
            end
        end
    end
end
