using SciMLSensitivity, OrdinaryDiffEq
using QuadGK, ForwardDiff, LinearAlgebra, Test

# Enzyme tests are skipped on Julia 1.12+ due to compatibility issues
# See: https://github.com/SciML/SciMLSensitivity.jl/issues/1323
if VERSION >= v"1.12"
    @info "Skipping Enzyme closure tests on Julia 1.12+ due to Enzyme compatibility issues"
    @testset "Enzyme Closure Tests (skipped on Julia 1.12+)" begin
        @test_skip false
    end
else
    using Enzyme, Zygote, Reactant

    import Base.zero

    struct Dense{T, F <: Function}
        n_inp::Int
        n_nodes::Int
        W::Matrix{T}
        b::Vector{T}
        activation::F
    end

    function Dense(n_inp, n_nodes, f::Function, T, randfn::Function = rand)
        return Dense(
            n_inp, n_nodes, convert.(T, randfn(n_nodes, n_inp)),
            convert.(T, randfn(n_nodes)), f
        )
    end

    struct NN{T}
        n_inp::Int
        layers::Vector{Dense{T}}
        intermediates::Vector{Vector{T}}
    end

    function NN(n_inp, layers, ::Type{T}) where {T}
        @assert length(layers) >= 1
        @assert n_inp == layers[1].n_inp
        for i in eachindex(layers)[1:(end - 1)]
            @assert layers[i].n_nodes == layers[i + 1].n_inp
        end
        return NN(n_inp, layers, [zeros(T, layer.n_nodes) for layer in layers])
    end

    function paramlength(nn::NN)
        r = 0
        for l in nn.layers
            r = r + length(l.W)
            r = r + length(l.b)
        end
        return r
    end

    function get_params(nn::NN)
        ret = eltype(nn.layers[1].W)[]
        for l in nn.layers
            append!(ret, l.W)
            append!(ret, l.b)
        end
        return ret
    end

    function set_params(nn, params)
        i = 1
        for l in nn.layers
            l.W .= reshape(params[i:(i + length(l.W) - 1)], size(l.W))
            i = i + length(l.W)
            l.b .= params[i:(i + length(l.b) - 1)]
            i = i + length(l.b)
        end
        return
    end

    function Base.zero(nn::NN)
        newnn = deepcopy(nn)
        for l in newnn.layers
            l.W .= 0.0
            l.b .= 0.0
        end
        for inter in newnn.intermediates
            inter .= 0.0
        end
        return newnn
    end

    function applydense!(d::Dense, inp, out)
        mul!(out, d.W, inp, 1.0, 0.0)
        out .+= d.b
        return nothing
    end

    function applyNN!(nn::NN, inp, out)
        applydense!(nn.layers[1], inp, nn.intermediates[1])
        for i in eachindex(nn.layers)[2:end]
            applydense!(nn.layers[i], nn.intermediates[i - 1], nn.intermediates[i])
        end
        out .+= nn.intermediates[end]
        return nothing
    end

    const step = 0.22
    const H_0 = diagm(1:2)
    const H_D = 0.01 * [0.0 1.0; 1.0 0.0]
    const repart = 1:2
    const impart = 3:4

    ##cell

    function make_dfunc(T)
        nn = NN(4, [Dense(4, 10, tanh, T), Dense(10, 4, sin, T)], T)
        plen = paramlength(nn)
        set_params(nn, 1.0e-3 * rand(plen))
        function dfunc(dstate, state, p, t)
            set_params(nn, p)
            scratch = zeros(eltype(dstate), 4)
            dstate[impart] .= -1.0 .*
                (H_0 * state[repart] .+ cos(2.0 * t) .* H_D * state[repart])
            dstate[repart] .= H_0 * state[impart] .+ cos(2.0 * t) .* H_D * state[impart]
            applyNN!(nn, dstate, scratch)
            dstate .+= scratch
            return nothing
        end
        return dfunc, nn
    end

    dfunc, nn = make_dfunc(Float64)
    ##cell initialize and solve
    y0 = [1.0, 0.0, 0.0, 0.0]
    p = get_params(nn)
    tspan = (0, 20.0)

    #test dfunc works
    ds = zero(y0)
    dfunc(ds, y0, p, 0.2) #test dfunc works

    #get solution
    prob = ODEProblem{true}(dfunc, y0, tspan, p)
    sol = solve(prob, Tsit5(), reltol = 1.0e-10)
    ##cell
    const target = zero(y0)
    target[2] = 1.0
    function g(u, p, t)
        return dot(u, target)^2
    end

    function gintegrate(p)
        dfunc, nn = make_dfunc(eltype(p))
        set_params(nn, p)
        prob = ODEProblem{true}(dfunc, y0, tspan, p)
        sol = solve(prob, Tsit5(), abstol = 1.0e-12, reltol = 1.0e-12)
        integral, error = quadgk((t) -> (g(sol(t), p, t)), tspan...)
        return integral
    end
    refdp = ForwardDiff.gradient(gintegrate, p)

    du1, dp1 = adjoint_sensitivities(
        sol, Tsit5(); g,
        sensealg = BacksolveAdjoint(autodiff = true, autojacvec = EnzymeVJP()),
        abstol = 1.0e-12, reltol = 1.0e-12
    )
    @test isapprox(dp1', refdp, atol = 1.0e-5)
    du1r, dp1r = adjoint_sensitivities(
        sol, Tsit5(); g,
        sensealg = BacksolveAdjoint(autodiff = true, autojacvec = ReactantVJP()),
        abstol = 1.0e-12, reltol = 1.0e-12
    )
    @test isapprox(dp1r', refdp, atol = 1.0e-5)
    du2, dp2 = adjoint_sensitivities(
        sol, Tsit5(); g,
        sensealg = GaussAdjoint(autodiff = true, autojacvec = EnzymeVJP()),
        abstol = 1.0e-12, reltol = 1.0e-12
    )
    @test isapprox(dp2', refdp, atol = 1.0e-5)
    du2r, dp2r = adjoint_sensitivities(
        sol, Tsit5(); g,
        sensealg = GaussAdjoint(autodiff = true, autojacvec = ReactantVJP()),
        abstol = 1.0e-12, reltol = 1.0e-12
    )
    @test isapprox(dp2r', refdp, atol = 1.0e-5)
    du3, dp3 = adjoint_sensitivities(
        sol, Tsit5(); g,
        sensealg = QuadratureAdjoint(autodiff = true, autojacvec = EnzymeVJP()),
        abstol = 1.0e-12, reltol = 1.0e-12
    )
    @test isapprox(dp3', refdp, atol = 1.0e-5)
    du3r, dp3r = adjoint_sensitivities(
        sol, Tsit5(); g,
        sensealg = QuadratureAdjoint(autodiff = true, autojacvec = ReactantVJP()),
        abstol = 1.0e-12, reltol = 1.0e-12
    )
    @test isapprox(dp3r', refdp, atol = 1.0e-5)
    du4, dp4 = adjoint_sensitivities(
        sol, Tsit5(); g,
        sensealg = InterpolatingAdjoint(autodiff = true, autojacvec = EnzymeVJP()),
        abstol = 1.0e-12, reltol = 1.0e-12
    )
    @test isapprox(dp4', refdp, atol = 1.0e-5)
    du4r, dp4r = adjoint_sensitivities(
        sol, Tsit5(); g,
        sensealg = InterpolatingAdjoint(autodiff = true, autojacvec = ReactantVJP()),
        abstol = 1.0e-12, reltol = 1.0e-12
    )
    @test isapprox(dp4r', refdp, atol = 1.0e-5)

end  # VERSION < v"1.12" else block
