using OrdinaryDiffEq, SciMLSensitivity, Zygote, Test, Reactant
using Functors: Functors, @functor
using LinearAlgebra: mul!

# Define a custom functor parameter struct that is NOT an AbstractArray
# and does NOT implement SciMLStructures
struct FunctorParams{W <: AbstractMatrix, B <: AbstractVector}
    weights::W
    bias::B
end
@functor FunctorParams

# ODE right-hand side using functor params (out-of-place)
function ode_f(u, p::FunctorParams, t)
    return p.weights * u .+ p.bias
end

# ODE right-hand side using functor params (in-place)
# Note: uses .= instead of mul! because Zygote.Buffer doesn't support mul!
function ode_f!(du, u, p::FunctorParams, t)
    du .= p.weights * u .+ p.bias
    return nothing
end

# Reference: compute gradient using Zygote with plain array parameters
function reference_gradient_oop(u0, tspan, weights, bias)
    flat_p = vcat(vec(weights), bias)
    n = length(u0)
    function ode_flat(u, p, t)
        W = reshape(p[1:(n * n)], n, n)
        b = p[(n * n + 1):end]
        return W * u .+ b
    end
    gs = Zygote.gradient(flat_p) do p
        prob = ODEProblem(ode_flat, u0, tspan, p)
        sol = solve(
            prob, Tsit5(); sensealg = GaussAdjoint(autojacvec = ZygoteVJP()),
            abstol = 1.0e-12, reltol = 1.0e-12
        )
        return sum(abs2, last(sol.u))
    end
    return gs[1]
end

@testset "Functor Parameter Support" begin
    @testset "Trait tests" begin
        @test SciMLSensitivity.supports_functor_params(QuadratureAdjoint()) == false
        @test SciMLSensitivity.supports_functor_params(GaussAdjoint()) == true
        @test SciMLSensitivity.supports_functor_params(
            GaussKronrodAdjoint()
        ) == true
        @test SciMLSensitivity.supports_functor_params(InterpolatingAdjoint()) == false
        @test SciMLSensitivity.supports_functor_params(BacksolveAdjoint()) == false

        @test SciMLSensitivity.supports_structured_vjp(ZygoteVJP()) == true
        @test SciMLSensitivity.supports_structured_vjp(EnzymeVJP()) == true
        @test SciMLSensitivity.supports_structured_vjp(ReactantVJP()) == true
        @test SciMLSensitivity.supports_structured_vjp(ReverseDiffVJP()) == false
        @test SciMLSensitivity.supports_structured_vjp(false) == false
        @test SciMLSensitivity.supports_structured_vjp(true) == false
    end

    # Setup
    n = 2
    u0 = Float64[1.0, 2.0]
    tspan = (0.0, 0.5)
    weights = Float64[-0.5 0.1; -0.1 -0.3]
    bias = Float64[0.1, -0.2]
    p0 = FunctorParams(weights, bias)

    # Reference gradient (ForwardDiff on flat vector)
    ref_grad = reference_gradient_oop(u0, tspan, weights, bias)

    function loss_oop(p_func, sensealg)
        prob = ODEProblem(ode_f, u0, tspan, p_func)
        sol = solve(prob, Tsit5(); sensealg, abstol = 1.0e-12, reltol = 1.0e-12)
        return sum(abs2, last(sol.u))
    end

    function loss_ip(p_func, sensealg)
        prob = ODEProblem(ode_f!, u0, tspan, p_func)
        sol = solve(prob, Tsit5(); sensealg, abstol = 1.0e-12, reltol = 1.0e-12)
        return sum(abs2, last(sol.u))
    end

    function extract_flat_grad(g::FunctorParams)
        return vcat(vec(g.weights), g.bias)
    end
    function extract_flat_grad(g::NamedTuple)
        return vcat(vec(g.weights), g.bias)
    end

    @testset "GaussAdjoint + ZygoteVJP (out-of-place)" begin
        sensealg = GaussAdjoint(autojacvec = ZygoteVJP())
        gs = Zygote.gradient(p0) do p
            loss_oop(p, sensealg)
        end
        g = gs[1]
        @test g !== nothing
        flat_g = extract_flat_grad(g)
        @test isapprox(flat_g, ref_grad, rtol = 1.0e-4)
    end

    @testset "GaussAdjoint + ZygoteVJP (in-place)" begin
        sensealg = GaussAdjoint(autojacvec = ZygoteVJP())
        gs = Zygote.gradient(p0) do p
            loss_ip(p, sensealg)
        end
        g = gs[1]
        @test g !== nothing
        flat_g = extract_flat_grad(g)
        @test isapprox(flat_g, ref_grad, rtol = 1.0e-4)
    end

    @testset "Error tests: unsupported algorithms with functor params" begin
        # InterpolatingAdjoint doesn't support functor params
        @test_throws SciMLSensitivity.AdjointSensitivityParameterCompatibilityError begin
            Zygote.gradient(p0) do p
                prob = ODEProblem(ode_f, u0, tspan, p)
                sol = solve(
                    prob, Tsit5(); sensealg = InterpolatingAdjoint(), abstol = 1.0e-12,
                    reltol = 1.0e-12
                )
                return sum(abs2, last(sol.u))
            end
        end

        # BacksolveAdjoint doesn't support functor params
        @test_throws SciMLSensitivity.AdjointSensitivityParameterCompatibilityError begin
            Zygote.gradient(p0) do p
                prob = ODEProblem(ode_f, u0, tspan, p)
                sol = solve(
                    prob, Tsit5(); sensealg = BacksolveAdjoint(), abstol = 1.0e-12,
                    reltol = 1.0e-12
                )
                return sum(abs2, last(sol.u))
            end
        end

        # QuadratureAdjoint doesn't support functor params
        @test_throws SciMLSensitivity.AdjointSensitivityParameterCompatibilityError begin
            Zygote.gradient(p0) do p
                prob = ODEProblem(ode_f, u0, tspan, p)
                sol = solve(
                    prob, Tsit5(); sensealg = QuadratureAdjoint(), abstol = 1.0e-12,
                    reltol = 1.0e-12
                )
                return sum(abs2, last(sol.u))
            end
        end

        # GaussAdjoint + ReverseDiffVJP doesn't support functor params
        @test_throws ErrorException begin
            Zygote.gradient(p0) do p
                prob = ODEProblem(ode_f, u0, tspan, p)
                sol = solve(
                    prob, Tsit5();
                    sensealg = GaussAdjoint(autojacvec = ReverseDiffVJP()),
                    abstol = 1.0e-12, reltol = 1.0e-12
                )
                return sum(abs2, last(sol.u))
            end
        end
    end
end
