using OrdinaryDiffEq, RecursiveArrayTools, LinearAlgebra
using OrdinaryDiffEqLowOrderRK: BS3, DP5
using Zygote, SciMLSensitivity, Random, Test

struct CustomArray{T, N}
    x::Array{T, N}
end
Base.size(x::CustomArray) = size(x.x)
Base.size(x::CustomArray, d::Integer) = size(x.x, d)
Base.axes(x::CustomArray) = axes(x.x)
Base.ndims(x::CustomArray) = ndims(x.x)
Base.ndims(::Type{<:CustomArray{T, N}}) where {T, N} = N
Base.zero(x::CustomArray) = CustomArray(zero(x.x))
Base.zero(::Type{<:CustomArray{T, N}}) where {T, N} = CustomArray(zero(Array{T, N}))
function Base.similar(x::CustomArray, dims::Union{Integer, AbstractUnitRange}...)
    return CustomArray(similar(x.x, dims...))
end
Base.copyto!(x::CustomArray, y::CustomArray) = CustomArray(copyto!(x.x, y.x))
Base.copy(x::CustomArray) = CustomArray(copy(x.x))
Base.length(x::CustomArray) = length(x.x)
Base.isempty(x::CustomArray) = isempty(x.x)
Base.eltype(x::CustomArray) = eltype(x.x)
Base.fill!(x::CustomArray, y) = CustomArray(fill!(x.x, y))
Base.getindex(x::CustomArray, i) = getindex(x.x, i)
Base.setindex!(x::CustomArray, v, idx) = setindex!(x.x, v, idx)
Base.firstindex(x::CustomArray) = firstindex(x.x)
Base.lastindex(x::CustomArray) = lastindex(x.x)
Base.eachindex(x::CustomArray) = eachindex(x.x)
Base.mapreduce(f, op, x::CustomArray; kwargs...) = mapreduce(f, op, x.x; kwargs...)
Base.any(f::Function, x::CustomArray; kwargs...) = any(f, x.x; kwargs...)
Base.all(f::Function, x::CustomArray; kwargs...) = all(f, x.x; kwargs...)
Base.similar(x::CustomArray, t) = CustomArray(similar(x.x, t))
Base.:(+)(x::CustomArray, y::CustomArray) = CustomArray(x.x + y.x)
Base.:(+)(x::CustomArray, y::AbstractArray) = CustomArray(x.x + y)
Base.:(+)(x::AbstractArray, y::CustomArray) = CustomArray(x + y.x)
Base.:(==)(x::CustomArray, y::CustomArray) = x.x == y.x
Base.:(*)(x::Number, y::CustomArray) = CustomArray(x * y.x)
Base.:(*)(x::CustomArray, y::Number) = CustomArray(x.x * y)
Base.:(/)(x::CustomArray, y::Number) = CustomArray(x.x / y)
LinearAlgebra.norm(x::CustomArray) = norm(x.x)
# Sensitivity / LinearAlgebra paths expect a plain 1-d array from `vec`, not a re-wrapped CustomArray.
LinearAlgebra.vec(x::CustomArray) = vec(x.x)

struct CustomStyle{N} <: Broadcast.BroadcastStyle where {N} end
CustomStyle(::Val{N}) where {N} = CustomStyle{N}()
CustomStyle{M}(::Val{N}) where {N, M} = NoIndexStyle{N}()
Base.BroadcastStyle(::Type{<:CustomArray{T, N}}) where {T, N} = CustomStyle{N}()
function Broadcast.BroadcastStyle(
        ::CustomStyle{N}, ::Broadcast.DefaultArrayStyle{0}
    ) where {N}
    return CustomStyle{N}()
end
function Base.similar(
        bc::Base.Broadcast.Broadcasted{CustomStyle{N}}, ::Type{ElType}
    ) where {N, ElType}
    return CustomArray(similar(Array{ElType, N}, axes(bc)))
end
Base.@propagate_inbounds Base.Broadcast._broadcast_getindex(x::CustomArray, i) = x.x[i]
Base.Broadcast.extrude(x::CustomArray) = x
Base.Broadcast.broadcastable(x::CustomArray) = x

@inline function Base.copyto!(
        dest::CustomArray,
        bc::Base.Broadcast.Broadcasted{
            <:Union{
                Base.Broadcast.AbstractArrayStyle, CustomStyle,
            },
        }
    )
    axes(dest) == axes(bc) || throwdm(axes(dest), axes(bc))
    bc′ = Base.Broadcast.preprocess(dest, bc)
    dest′ = dest.x
    @simd for I in 1:length(dest′)
        @inbounds dest′[I] = bc′[I]
    end
    return dest
end
@inline function Base.copy(bc::Base.Broadcast.Broadcasted{<:CustomStyle})
    bcf = Broadcast.flatten(bc)
    x = find_x(bcf)
    data = similar(x, eltype(bcf[1]))
    @inbounds @simd for I in 1:length(x)
        data[I] = bcf[I]
    end
    return CustomArray(data)
end
find_x(bc::Broadcast.Broadcasted) = find_x(bc.args)
find_x(args::Tuple) = find_x(find_x(args[1]), Base.tail(args))
find_x(x) = x
find_x(::Any, rest) = find_x(rest)
find_x(x::CustomArray, rest) = x.x

RecursiveArrayTools.recursive_unitless_bottom_eltype(x::CustomArray) = eltype(x)
RecursiveArrayTools.recursivecopy!(dest::CustomArray, src::CustomArray) = copyto!(dest, src)
RecursiveArrayTools.recursivecopy(x::CustomArray) = copy(x)
RecursiveArrayTools.recursivefill!(x::CustomArray, a) = fill!(x, a)

Base.show_vector(io::IO, x::CustomArray) = Base.show_vector(io, x.x)

Base.show(io::IO, x::CustomArray) = (print(io, "CustomArray"); show(io, x.x))
function Base.show(io::IO, ::MIME"text/plain", x::CustomArray)
    println(io, Base.summary(x), ":")
    return Base.print_array(io, x.x)
end

ca0 = CustomArray(ones(2))
tspan = (0.0, 1.0)
par = [rand(), rand()]

algs = [Tsit5(), BS3(), Vern9(), DP5()]

for alg in algs
    function cost(p, u0 = ca0; kwargs...)
        prob = ODEProblem(
            (du, u, p, t) -> (du[1] = p[1] * u[1] + p[2] * u[2]; du[2] = p[2] * u[1]),
            u0, tspan, p
        )
        sol = solve(prob, alg; save_everystep = false, kwargs...)
        return 1 - norm(sol.u[end])^2
    end
    @test_nowarn Zygote.gradient(cost, par)
    for p in ([0.2, 0.7], [-0.7, 0.4], [0.2, 0.7])
        initial = copy(ca0)
        parameters = copy(p)
        dp, du0 = Zygote.gradient(p, initial) do q, x
            cost(q, x; abstol = 1.0e-11, reltol = 1.0e-11)
        end
        A = [p[1] p[2]; p[2] 0.0]
        y = exp(A) * initial.x
        directions = ([1.0 0.0; 0.0 0.0], [0.0 1.0; 1.0 0.0])
        expected_dp = map(directions) do E
            # The upper-right block is the Frechet derivative of exp(A) along E.
            dy = exp([A E; zeros(2, 2) A])[1:2, 3:4] * initial.x
            -2dot(y, dy)
        end
        @test dp ≈ collect(expected_dp) rtol = 1.0e-7 atol = 1.0e-8
        @test du0.x ≈ -2exp(A)' * y rtol = 1.0e-7 atol = 1.0e-8
        @test initial.x == ca0.x == ones(2)
        @test p == parameters
    end
end
