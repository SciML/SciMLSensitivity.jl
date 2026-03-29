# NOTE: `fmap` can handle all these cases without us defining them, but it often makes the
#       code type unstable. So we define them here to make the code type stable.
# Handle Non-Array Parameters in a Generic Fashion

# Base cases for non-differentiable Functors leaf types (Symbol, String, Char, etc.)
# that fmap will pass to the mapped function. Without these, the generic fallbacks
# `f(x) = fmap(f, x)` infinitely recurse on leaves since fmap calls f(leaf) again.
# Number gets proper zero/arithmetic; non-numeric leaves are returned unchanged.
const _NonDiffLeaf = Union{Symbol, AbstractString, AbstractChar}
"""
    recursive_copyto!(y, x)

`y[:] .= vec(x)` for generic `x` and `y`. This is used to handle non-array parameters!
"""
recursive_copyto!(y::AbstractArray, x::AbstractArray) = copyto!(y, x)
recursive_copyto!(y::AbstractArray, x::Number) = y .= x
recursive_copyto!(y::Tuple, x::Tuple) = map(recursive_copyto!, y, x)
function recursive_copyto!(y::NamedTuple{F}, x::NamedTuple{F}) where {F}
    return map(recursive_copyto!, values(y), values(x))
end
recursive_copyto!(y::T, x::T) where {T} = fmap(recursive_copyto!, y, x)
recursive_copyto!(y, ::Nothing) = y
recursive_copyto!(::Nothing, ::Nothing) = nothing
function recursive_copyto!(y::T, x::NamedTuple) where {T}
    return fmap(recursive_copyto!, y, x)
end
recursive_copyto!(y::Number, x::Number) = x
recursive_copyto!(y::T, x::T) where {T <: _NonDiffLeaf} = y

"""
    neg!(x)

`x .*= -1` for generic `x`. This is used to handle non-array parameters!
"""
recursive_neg!(x::AbstractArray) = (x .*= -1)
recursive_neg!(x::Tuple) = map(recursive_neg!, x)
recursive_neg!(x::NamedTuple{F}) where {F} = NamedTuple{F}(map(recursive_neg!, values(x)))
recursive_neg!(x) = fmap(recursive_neg!, x)
recursive_neg!(::Nothing) = nothing
recursive_neg!(x::Number) = -x
recursive_neg!(x::_NonDiffLeaf) = x

"""
    recursive_sub!(y, x)

`y .-= x` for generic `x` and `y`. This is used to handle non-array parameters!
"""
recursive_sub!(y::AbstractArray, x::AbstractArray) = axpy!(-1, x, y)
recursive_sub!(y::Tuple, x::Tuple) = map(recursive_sub!, y, x)
function recursive_sub!(y::NamedTuple{F}, x::NamedTuple{F}) where {F}
    return NamedTuple{F}(map(recursive_sub!, values(y), values(x)))
end
recursive_sub!(y::T, x::T) where {T} = fmap(recursive_sub!, y, x)
recursive_sub!(y, ::Nothing) = y
recursive_sub!(::Nothing, ::Nothing) = nothing
recursive_sub!(y::Number, x::Number) = y - x
recursive_sub!(y::T, x::T) where {T <: _NonDiffLeaf} = y

"""
    recursive_add!(y, x)

`y .+= x` for generic `x` and `y`. This is used to handle non-array parameters!
"""
recursive_add!(y::AbstractArray, x::AbstractArray) = y .+= x
recursive_add!(y::Tuple, x::Tuple) = recursive_add!.(y, x)
function recursive_add!(y::NamedTuple{F}, x::NamedTuple{F}) where {F}
    return NamedTuple{F}(recursive_add!(values(y), values(x)))
end
recursive_add!(y::T, x::T) where {T} = fmap(recursive_add!, y, x)
recursive_add!(y, ::Nothing) = y
recursive_add!(::Nothing, ::Nothing) = nothing
recursive_add!(y::Number, x::Number) = y + x
recursive_add!(y::T, x::T) where {T <: _NonDiffLeaf} = y

"""
    allocate_vjp(λ, x)
    allocate_vjp(x)

`similar(λ, size(x))` for generic `x`. This is used to handle non-array parameters!
"""
function allocate_vjp(λ::AbstractArray{T}, x::AbstractArray) where {T}
    return fill!(similar(λ, size(x)), zero(T))
end
allocate_vjp(λ::AbstractArray, x::Tuple) = allocate_vjp.((λ,), x)
function allocate_vjp(λ::AbstractArray, x::NamedTuple{F}) where {F}
    return NamedTuple{F}(allocate_vjp.((λ,), values(x)))
end
allocate_vjp(λ::AbstractArray, x) = fmap(Base.Fix1(allocate_vjp, λ), x)
allocate_vjp(λ::AbstractArray{T}, x::Number) where {T} = zero(T)
allocate_vjp(::AbstractArray, x::_NonDiffLeaf) = x

# ---------------------------------------------
# fix 3: make allocate_vjp safe on "no params"
allocate_vjp(x::Nothing) = nothing
allocate_vjp(x::SciMLBase.NullParameters) = nothing
allocate_vjp(::AbstractArray, ::Nothing) = nothing
allocate_vjp(::AbstractArray, ::SciMLBase.NullParameters) = nothing
# ---------------------------------------------
allocate_vjp(x::AbstractArray) = zero(x) # similar(x)
allocate_vjp(x::Tuple) = allocate_vjp.(x)
allocate_vjp(x::NamedTuple{F}) where {F} = NamedTuple{F}(allocate_vjp.(values(x)))
allocate_vjp(x) = fmap(allocate_vjp, x)
allocate_vjp(x::Number) = zero(x)
allocate_vjp(x::_NonDiffLeaf) = x

"""
    allocate_zeros(x)

`zero.(x)` for generic `x`. This is used to handle non-array parameters!
"""
#---------------------------------------------
# fix 2:
allocate_zeros(::Nothing) = nothing
allocate_zeros(::SciMLBase.NullParameters) = SciMLBase.NullParameters()
#---------------------------------------------
allocate_zeros(x::AbstractArray) = zero.(x)
allocate_zeros(x::Tuple) = allocate_zeros.(x)
allocate_zeros(x::NamedTuple{F}) where {F} = NamedTuple{F}(allocate_zeros.(values(x)))
allocate_zeros(x) = fmap(allocate_zeros, x)
allocate_zeros(x::Number) = zero(x)
allocate_zeros(x::_NonDiffLeaf) = x

"""
    recursive_adjoint(y)

`adjoint(y)` for generic `y`. This is used to handle non-array parameters!
"""
#---------------------------------------------
# fix 3: make recursive_adjoint safe on "no params"
recursive_adjoint(::Nothing) = nothing
#---------------------------------------------
recursive_adjoint(y::AbstractArray) = adjoint(y)
recursive_adjoint(y::Tuple) = recursive_adjoint.(y)
recursive_adjoint(y::NamedTuple{F}) where {F} = NamedTuple{F}(recursive_adjoint.(values(y)))
recursive_adjoint(y) = fmap(recursive_adjoint, y)
recursive_adjoint(y::Number) = adjoint(y)
recursive_adjoint(y::_NonDiffLeaf) = y
