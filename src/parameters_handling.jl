# NOTE: `fmap` can handle all these cases without us defining them, but it often makes the
#       code type unstable. So we define them here to make the code type stable.
# Handle Non-Array Parameters in a Generic Fashion
"""
    recursive_copyto!(y, x)

`y[:] .= vec(x)` for generic `x` and `y`. This is used to handle non-array parameters!
"""
recursive_copyto!(y::AbstractArray, x::AbstractArray) = copyto!(y, x)
recursive_copyto!(y::AbstractArray, x::Number) = y .= x
recursive_copyto!(y::Tuple, x::Tuple) = map(recursive_copyto!, y, x)
function recursive_copyto!(y::NamedTuple{F}, x::NamedTuple{F}) where {F}
    map(recursive_copyto!, values(y), values(x))
end
recursive_copyto!(y::T, x::T) where {T} = fmap(recursive_copyto!, y, x)
recursive_copyto!(y, ::Nothing) = y
recursive_copyto!(::Nothing, ::Nothing) = nothing

"""
    neg!(x)

`x .*= -1` for generic `x`. This is used to handle non-array parameters!
"""
recursive_neg!(x::AbstractArray) = (x .*= -1)
recursive_neg!(x::Tuple) = map(recursive_neg!, x)
recursive_neg!(x::NamedTuple{F}) where {F} = NamedTuple{F}(map(recursive_neg!, values(x)))
recursive_neg!(x) = fmap(recursive_neg!, x)
recursive_neg!(::Nothing) = nothing

"""
    recursive_sub!(y, x)

`y .-= x` for generic `x` and `y`. This is used to handle non-array parameters!
"""
recursive_sub!(y::AbstractArray, x::AbstractArray) = axpy!(-1, x, y)
recursive_sub!(y::Tuple, x::Tuple) = map(recursive_sub!, y, x)
function recursive_sub!(y::NamedTuple{F}, x::NamedTuple{F}) where {F}
    NamedTuple{F}(map(recursive_sub!, values(y), values(x)))
end
recursive_sub!(y::T, x::T) where {T} = fmap(recursive_sub!, y, x)
recursive_sub!(y, ::Nothing) = y
recursive_sub!(::Nothing, ::Nothing) = nothing

"""
    recursive_add!(y, x)

`y .+= x` for generic `x` and `y`. This is used to handle non-array parameters!
"""
recursive_add!(y::AbstractArray, x::AbstractArray) = y .+= x
recursive_add!(y::Tuple, x::Tuple) = recursive_add!.(y, x)
function recursive_add!(y::NamedTuple{F}, x::NamedTuple{F}) where {F}
    NamedTuple{F}(recursive_add!(values(y), values(x)))
end
recursive_add!(y::T, x::T) where {T} = fmap(recursive_add!, y, x)
recursive_add!(y, ::Nothing) = y
recursive_add!(::Nothing, ::Nothing) = nothing

"""
    allocate_vjp(λ, x)
    allocate_vjp(x)

`similar(λ, size(x))` for generic `x`. This is used to handle non-array parameters!
"""
allocate_vjp(λ::AbstractArray, x::AbstractArray) = similar(λ, size(x))
allocate_vjp(λ::AbstractArray, x::Tuple) = allocate_vjp.((λ,), x)
function allocate_vjp(λ::AbstractArray, x::NamedTuple{F}) where {F}
    NamedTuple{F}(allocate_vjp.((λ,), values(x)))
end
allocate_vjp(λ::AbstractArray, x) = fmap(Base.Fix1(allocate_vjp, λ), x)

allocate_vjp(x::AbstractArray) = similar(x)
allocate_vjp(x::Tuple) = allocate_vjp.(x)
allocate_vjp(x::NamedTuple{F}) where {F} = NamedTuple{F}(allocate_vjp.(values(x)))
allocate_vjp(x) = fmap(allocate_vjp, x)

"""
    allocate_zeros(x)

`zero.(x)` for generic `x`. This is used to handle non-array parameters!
"""
allocate_zeros(x::AbstractArray) = zero.(x)
allocate_zeros(x::Tuple) = allocate_zeros.(x)
allocate_zeros(x::NamedTuple{F}) where {F} = NamedTuple{F}(allocate_zeros.(values(x)))
allocate_zeros(x) = fmap(allocate_zeros, x)

"""
    recursive_adjoint(y)

`adjoint(y)` for generic `y`. This is used to handle non-array parameters!
"""
recursive_adjoint(y::AbstractArray) = adjoint(y)
recursive_adjoint(y::Tuple) = recursive_adjoint.(y)
recursive_adjoint(y::NamedTuple{F}) where {F} = NamedTuple{F}(recursive_adjoint.(values(y)))
recursive_adjoint(y) = fmap(recursive_adjoint, y)
