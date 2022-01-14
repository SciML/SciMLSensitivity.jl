abstract type AbstractPullbackMultiplyOperator{T} <: DiffEqBase.AbstractDiffEqLinearOperator{T} end

# TODO: Other AD choices
PullbackMultiplyOperator(::ZygoteVJP, f, u, p=nothing, t=nothing) =
    ZygotePullbackMultiplyOperator(f, u, p, t)

function PullbackMultiplyOperator(vjp_choice::Bool, f, u, p=nothing, t=nothing)
    if vjp_choice
        return PullbackMultiplyOperator(ZygoteVJP(), f, u, p, t)
    else
        return VecJacOperator(f, u, p; autodiff=false)
    end
end

struct ZygotePullbackMultiplyOperator{T,F,S} <: AbstractPullbackMultiplyOperator{T}
    back::F
    s::S

    function ZygotePullbackMultiplyOperator(f, u, p=nothing, t=nothing)
        val, back = Zygote.pullback(x -> f(x, p, t), u)
        s = size(val)
        return new{typeof(u),typeof(back),typeof(s)}(back, s)
    end
end

Base.size(p::ZygotePullbackMultiplyOperator) = (prod(p.s), prod(p.s))
Base.size(p::ZygotePullbackMultiplyOperator, i::Int64) = size(p)[i]

Base.eltype(::ZygotePullbackMultiplyOperator{T}) where {T} = T

function LinearAlgebra.mul!(
    du::AbstractVector,
    P::ZygotePullbackMultiplyOperator,
    x::AbstractVector,
)
    du .= vec(P * x)
end

function Base.:*(P::ZygotePullbackMultiplyOperator, x::AbstractVector)
    return P.back(reshape(x, P.s))[1]
end

SciMLBase.isinplace(::ZygotePullbackMultiplyOperator, ::Int64) = false
