# Piracy that used to be requires, allowing ReverseDiff.jl to be specialized for SciML

DiffEqBase.value(x::ReverseDiff.TrackedReal) = x.value
DiffEqBase.value(x::ReverseDiff.TrackedArray) = x.value

DiffEqBase.promote_u0(u0::ReverseDiff.TrackedArray, p::ReverseDiff.TrackedArray, t0) = u0
DiffEqBase.promote_u0(u0::AbstractArray{<:ReverseDiff.TrackedReal}, p::ReverseDiff.TrackedArray, t0) = u0
DiffEqBase.promote_u0(u0::ReverseDiff.TrackedArray, p::AbstractArray{<:ReverseDiff.TrackedReal}, t0) = u0
DiffEqBase.promote_u0(u0::AbstractArray{<:ReverseDiff.TrackedReal}, p::AbstractArray{<:ReverseDiff.TrackedReal}, t0) = u0
DiffEqBase.promote_u0(u0, p::ReverseDiff.TrackedArray, t0) = ReverseDiff.track(u0)
DiffEqBase.promote_u0(u0, p::AbstractArray{<:ReverseDiff.TrackedReal}, t0) = eltype(p).(u0)

# Support adaptive with non-tracked time
@inline function DiffEqBase.ODE_DEFAULT_NORM(u::ReverseDiff.TrackedArray, t) where {N}
    sqrt(sum(abs2, DiffEqBase.value(u)) / length(u))
end
@inline function DiffEqBase.ODE_DEFAULT_NORM(u::AbstractArray{<:ReverseDiff.TrackedReal,N}, t) where {N}
    sqrt(sum(x -> DiffEqBase.ODE_DEFAULT_NORM(x[1], x[2]), zip((DiffEqBase.value(x) for x in u), Iterators.repeated(t))) / length(u))
end
@inline function DiffEqBase.ODE_DEFAULT_NORM(u::Array{<:ReverseDiff.TrackedReal,N}, t) where {N}
    sqrt(sum(x -> DiffEqBase.ODE_DEFAULT_NORM(x[1], x[2]), zip((DiffEqBase.value(x) for x in u), Iterators.repeated(t))) / length(u))
end
@inline DiffEqBase.ODE_DEFAULT_NORM(u::ReverseDiff.TrackedReal, t) = abs(value(u))

# Support TrackedReal time, don't drop tracking on the adaptivity there
@inline function DiffEqBase.ODE_DEFAULT_NORM(u::ReverseDiff.TrackedArray, t::ReverseDiff.TrackedReal) where {N}
    sqrt(sum(abs2, u) / length(u))
end
@inline function DiffEqBase.ODE_DEFAULT_NORM(u::AbstractArray{<:ReverseDiff.TrackedReal,N}, t::ReverseDiff.TrackedReal) where {N}
    sqrt(sum(x -> DiffEqBase.ODE_DEFAULT_NORM(x[1], x[2]), zip(u, Iterators.repeated(t))) / length(u))
end
@inline function DiffEqBase.ODE_DEFAULT_NORM(u::Array{<:ReverseDiff.TrackedReal,N}, t::ReverseDiff.TrackedReal) where {N}
    sqrt(sum(x -> DiffEqBase.ODE_DEFAULT_NORM(x[1], x[2]), zip(u, Iterators.repeated(t))) / length(u))
end
@inline DiffEqBase.ODE_DEFAULT_NORM(u::ReverseDiff.TrackedReal, t::ReverseDiff.TrackedReal) = abs(u)

function DiffEqBase.solve_up(prob::DiffEqBase.DEProblem, sensealg::Union{DiffEqBase.AbstractSensitivityAlgorithm,Nothing}, u0::ReverseDiff.TrackedArray, p::ReverseDiff.TrackedArray, args...; kwargs...)
    ReverseDiff.track(solve_up, prob, sensealg, u0, p, args...; kwargs...)
end

function DiffEqBase.solve_up(prob::DiffEqBase.DEProblem, sensealg::Union{DiffEqBase.AbstractSensitivityAlgorithm,Nothing}, u0, p::ReverseDiff.TrackedArray, args...; kwargs...)
    ReverseDiff.track(solve_up, prob, sensealg, u0, p, args...; kwargs...)
end

function DiffEqBase.solve_up(prob::DiffEqBase.DEProblem, sensealg::Union{DiffEqBase.AbstractSensitivityAlgorithm,Nothing}, u0::ReverseDiff.TrackedArray, p, args...; kwargs...)
    ReverseDiff.track(solve_up, prob, sensealg, u0, p, args...; kwargs...)
end

@inline function DiffEqNoiseProcess.wiener_randn(rng::Random.AbstractRNG, proto::ReverseDiff.TrackedArray)
    ReverseDiff.track(convert.(eltype(proto.value), randn(rng, size(proto))))
end
@inline function DiffEqNoiseProcess.wiener_randn!(rng::AbstractRNG, rand_vec::Array{<:ReverseDiff.TrackedReal})
    rand_vec .= ReverseDiff.track.(randn.((rng,), typeof.(DiffEqBase.value.(rand_vec))))
end
@inline function DiffEqNoiseProcess.wiener_randn!(rng::AbstractRNG, rand_vec::AbstractArray{<:ReverseDiff.TrackedReal})
    rand_vec .= ReverseDiff.track.(randn.((rng,), typeof.(DiffEqBase.value.(rand_vec))))
end