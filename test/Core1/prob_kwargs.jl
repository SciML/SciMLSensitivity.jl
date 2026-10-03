using OrdinaryDiffEq, SciMLSensitivity

function growth(du, u, p, t)
    return @. du = p * u * (1 - u)
end
u0 = [0.1]
tspan = (0.0, 2.0)
prob = ODEProblem(growth, u0, tspan, [1.0])
sol = solve(prob, Tsit5(), reltol = 1.0e-8, abstol = 1.0e-8)

savetimes = [0.0, 1.0, 1.9]

function f(a)
    _prob = remake(prob, p = [a[1]], saveat = savetimes)
    predicted = solve(
        _prob, Tsit5(), sensealg = InterpolatingAdjoint(), abstol = 1.0e-12,
        reltol = 1.0e-12
    )
    return sum(predicted.u[end])
end

function f2(a)
    _prob = remake(prob, p = [a[1]], saveat = savetimes)
    predicted = solve(
        _prob, Tsit5(), sensealg = InterpolatingAdjoint(), abstol = 1.0e-12,
        reltol = 1.0e-12
    )
    return sum(predicted.u[end])
end

using Zygote
a = ones(3)
@test Zygote.gradient(f, a)[1][1] ≈ Zygote.gradient(f2, a)[1][1]
@test Zygote.gradient(f, a)[1][2] == Zygote.gradient(f2, a)[1][2] == 0
@test Zygote.gradient(f, a)[1][3] == Zygote.gradient(f2, a)[1][3] == 0

# callback in problem construction or in solve call should give same result
# https://github.com/SciML/SciMLSensitivity.jl/issues/1081
odef(du, u, p, t) = du[1] = u[1] * p[1]
prob = ODEProblem(odef, [2.0], (0.0, 1.0), [3.0])

# Callback duplication test uses Zygote and has issues on Julia 1.12+
# See: https://github.com/SciML/SciMLSensitivity.jl/issues
if VERSION < v"1.12"
    let callback_count1 = 0, callback_count2 = 0
        function f1(u0p, adjoint_type)
            condition(u, t, integrator) = t == 0.5
            affect!(integrator) = callback_count1 += 1
            cb = DiscreteCallback(condition, affect!)
            prob = ODEProblem{true}(odef, u0p[1:1], (0.0, 1.0), u0p[2:2]; callback = cb)
            return sum(solve(prob, Tsit5(), tstops = [0.5], sensealg = adjoint_type))
        end

        function f2(u0p, adjoint_type)
            condition(u, t, integrator) = t == 0.5
            affect!(integrator) = callback_count2 += 1
            cb = DiscreteCallback(condition, affect!)
            prob = ODEProblem{true}(odef, u0p[1:1], (0.0, 1.0), u0p[2:2])
            return sum(solve(prob, Tsit5(), tstops = [0.5], callback = cb, sensealg = adjoint_type))
        end

        @testset "Callback duplication check" begin
            u0p = [2.0, 3.0]
            for adjoint_type in [
                    ForwardDiffSensitivity(), ReverseDiffAdjoint(), TrackerAdjoint(),
                    BacksolveAdjoint(), InterpolatingAdjoint(), QuadratureAdjoint(), GaussAdjoint(),
                ]
                count1 = 0
                count2 = 0
                @test Zygote.gradient(x -> f1(x, adjoint_type), u0p) ==
                    Zygote.gradient(x -> f2(x, adjoint_type), u0p)
                @test callback_count1 == callback_count2
            end
        end
    end
else
    @info "Skipping callback duplication check on Julia 1.12+ due to Zygote compatibility issues"
end

# An explicitly passed callback argument — including `nothing` or an empty
# `CallbackSet` — replaces the callback stored in `prob.kwargs` under
# `merge_callbacks = false`, both in the primal solve and in the rebuilt
# problem used by the adjoint pass.
using FiniteDiff
@testset "Empty solve callback suppresses problem callback" begin
    lin(u, p, t) = [p[1]]
    cb = DiscreteCallback((u, t, i) -> t == 0.5, i -> (i.u .*= 2))
    prob_cb = ODEProblem(lin, [1.0], (0.0, 1.0), [2.0]; callback = cb)

    function suppressed_loss(p, callback)
        sol = solve(
            prob_cb, Tsit5(); p, callback, merge_callbacks = false,
            sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP()),
            tstops = [0.5], abstol = 1.0e-10, reltol = 1.0e-10
        )
        return only(sol.u[end])^2
    end

    # The suppressed-callback primal is du = p, so u(1) = 1 + p, loss = (1+p)^2.
    prob_nocb = ODEProblem(lin, [1.0], (0.0, 1.0), [2.0])
    fd_grad = only(
        FiniteDiff.finite_difference_gradient(
            p -> only(
                solve(prob_nocb, Tsit5(); p, abstol = 1.0e-10, reltol = 1.0e-10).u[end]
            )^2,
            [2.0]
        )
    )
    @test fd_grad ≈ 6.0 atol = 1.0e-6

    for callback in (nothing, CallbackSet(), CallbackSet(Any[], Any[]))
        value, grad = Zygote.withgradient(p -> suppressed_loss(p, callback), [2.0])
        @test value ≈ 9.0 atol = 1.0e-8
        @test only(grad[1]) ≈ fd_grad atol = 1.0e-6
    end
end
