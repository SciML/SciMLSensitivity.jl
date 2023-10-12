# NOTE: `fmap` can handle all these cases without us defining them, but it often makes the
#       code type unstable. So we define them here to make the code type stable.
# Handle Non-Array Parameters in a Generic Fashion
"""
    recursive_copyto!(y, x)

`y[:] .= vec(x)` for generic `x` and `y`. This is used to handle non-array parameters!
"""
recursive_copyto!(y::AbstractArray, x::AbstractArray) = copyto!(y, x)
recursive_copyto!(y::Tuple, x::Tuple) = map(recursive_copyto!, y, x)
recursive_copyto!(y::NamedTuple{F}, x::NamedTuple{F}) where {F} =
    map(recursive_copyto!, values(y), values(x))
recursive_copyto!(y::T, x::T) where {T} = fmap(recursive_copyto!, y, x)

"""
    neg!(x)

`x .*= -1` for generic `x`. This is used to handle non-array parameters!
"""
recursive_neg!(x::AbstractArray) = (x .*= -1)
recursive_neg!(x::Tuple) = map(recursive_neg!, x)
recursive_neg!(x::NamedTuple{F}) where {F} = NamedTuple{F}(map(recursive_neg!, values(x)))
recursive_neg!(x) = fmap(recursive_neg!, x)

"""
    recursive_sub!(y, x)

`y .-= x` for generic `x` and `y`. This is used to handle non-array parameters!
"""
recursive_sub!(y::AbstractArray, x::AbstractArray) = (y .-= x)
recursive_sub!(y::Tuple, x::Tuple) = map(recursive_sub!, y, x)
recursive_sub!(y::NamedTuple{F}, x::NamedTuple{F}) where {F} =
    NamedTuple{F}(map(recursive_sub!, values(y), values(x)))
recursive_sub!(y::T, x::T) where {T} = fmap(recursive_sub!, y, x)

"""
    allocate_vjp(λ, x)

`similar(λ, size(x))` for generic `x`. This is used to handle non-array parameters!
"""
allocate_vjp(λ::AbstractArray, x::AbstractArray) = similar(λ, size(x))
allocate_vjp(λ::AbstractArray, x::Tuple) = allocate_vjp.((λ,), x)
allocate_vjp(λ::AbstractArray, x::NamedTuple{F}) where {F} =
    NamedTuple{F}(allocate_vjp.((λ,), values(x)))
allocate_vjp(λ::AbstractArray, x) = fmap(Base.Fix1(allocate_vjp, λ), x)
