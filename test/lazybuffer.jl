using LinearAlgebra, OrdinaryDiffEq, Test, PreallocationTools
using Random, FiniteDiff, ForwardDiff, ReverseDiff, SciMLSensitivity, Zygote

# see https://github.com/SciML/PreallocationTools.jl/issues/29
@testset "VJP computation with LazyBuffer" begin
    u0 = rand(2, 2)
    p = rand(2, 2)
    struct foo{T}
        lbc::T
    end

    f = foo(LazyBufferCache())

    function (f::foo)(du, u, p, t)
        tmp = f.lbc[u]
        mul!(tmp, p, u) # avoid tmp = p*u
        @. du = u + tmp
        nothing
    end

    prob = ODEProblem(f, u0, (0.0, 1.0), p)

    function loss(u0, p; sensealg = nothing)
        _prob = remake(prob, u0 = u0, p = p)
        _sol = solve(_prob, Tsit5(), sensealg = sensealg, saveat = 0.1, abstol = 1e-14,
            reltol = 1e-14)
        sum(abs2, _sol)
    end

    loss(u0, p)

    du0 = FiniteDiff.finite_difference_gradient(u0 -> loss(u0, p), u0)
    dp = FiniteDiff.finite_difference_gradient(p -> loss(u0, p), p)
    Fdu0 = ForwardDiff.gradient(u0 -> loss(u0, p), u0)
    Fdp = ForwardDiff.gradient(p -> loss(u0, p), p)
    @test du0≈Fdu0 rtol=1e-8
    @test dp≈Fdp rtol=1e-8

    Zdu0, Zdp = Zygote.gradient(
        (u0, p) -> loss(u0, p;
            sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP())),
        u0, p)
    @test du0≈Zdu0 rtol=1e-8
    @test dp≈Zdp rtol=1e-8
end
