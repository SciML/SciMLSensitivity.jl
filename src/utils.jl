function isfunctor(x)
    return !(x isa AbstractArray) && !isscimlstructure(x) && !isempty(Functors.children(x))
end

function to_nt(s::T) where {T}
    return NamedTuple{propertynames(s)}(map(x -> getproperty(s, x), propertynames(s)))
end

"""
    solution_parameter_cotangent(Δ)

The parameter cotangent a solution tangent `Δ` carries in `Δ.prob.p`, or `nothing`.
Reverse rules on solution indexing (`sol[sym]` for an observed variable) put the observed
function's parameter cotangent there, since that part of the gradient is not in `Δ.u`.
"""
function solution_parameter_cotangent(Δ)
    Δp = unthunk(tangent_field(tangent_field(Δ, :prob), :p))
    return Δp isa AbstractZero ? nothing : Δp
end

# Read one field of a structural tangent, tolerating any tangent shape. A `Tangent` is lenient
# (returns `ZeroTangent()` for an absent field); a raw `NamedTuple` is not, so guard it; an
# `AbstractZero` (or anything else) has no fields, so it contributes nothing. A solution-shaped
# cotangent lands here too: its `p` is a parameter object rather than a gradient.
tangent_field(::Union{ZeroTangent, NoTangent}, ::Symbol) = ZeroTangent()
tangent_field(t::Tangent, f::Symbol) = getproperty(t, f)
tangent_field(t::NamedTuple, f::Symbol) = haskey(t, f) ? getfield(t, f) : ZeroTangent()
tangent_field(::Any, ::Symbol) = ZeroTangent()

"""
    tunable_cotangent(p, Δp, tunables, repack)

Project the structural parameter cotangent `Δp` (nested `Tangent`s or `NamedTuple`s
mirroring `p`, or an array for array parameters) onto the tunable portion of `p`. Returns
a cotangent aligned with `tunables = canonicalize(Tunable(), p)[1]`, or `nothing` when
there is no contribution.
"""
function tunable_cotangent(p, Δp, tunables, repack)
    (Δp === nothing || p === nothing || p isa SciMLBase.NullParameters) && return nothing
    isscimlstructure(p) || return nothing
    Δfull = fill_cotangent(repack(zero(tunables)), Δp)
    return canonicalize(Tunable(), Δfull)[1]
end

# Write the entries of a structural cotangent into a copy of the parameter object whose
# tunable portion is zero, so that `canonicalize` can read that portion back out. Only the
# tunable portion is read, so the other fields keep whatever the copy holds.
fill_cotangent(x, Δ) = typeof(Δ) === typeof(x) ? Δ : x  # a cotangent given as a parameter object
fill_cotangent(x::AbstractArray, Δ::AbstractArray) = reshape(Δ, size(x))
fill_cotangent(x::Number, Δ::Number) = Δ
function fill_cotangent(x, Δ::Union{Tangent{<:Any, <:NamedTuple}, NamedTuple})
    fnames = fieldnames(typeof(x))
    if x isa SciMLBase.DespecializedParameters
        # The wrapper forwards `getproperty`, so a cotangent taken through it may carry the
        # wrapped object's fields directly instead of under `params`.
        inner = haskey(Δ, :params) ? unthunk(Δ[:params]) : Δ
        return SciMLBase.DespecializedParameters(
            fill_cotangent(SciMLBase.unwrap_parameters(x), inner)
        )
    elseif !(:params in fnames) && length(Δ) == 1 && haskey(Δ, :params)
        # The cotangent was taken through a `DespecializedParameters` wrapper the solver put
        # around `p`; `p` itself is the wrapped object, an array for array parameters.
        return fill_cotangent(x, unthunk(Δ[:params]))
    end
    isempty(fnames) && return x
    vals = map(fnames) do k
        v = getfield(x, k)
        haskey(Δ, k) ? fill_cotangent(v, unthunk(Δ[k])) : v
    end
    return ConstructionBase.constructorof(typeof(x))(vals...)
end

"""
    _get_sensitivity_vjp_verbose(verbose)

Extract the verbosity setting for sensitivity VJP choice warnings.

Returns `true` if warnings should be displayed, `false` if they should be silenced.
Handles:

  - `Bool`: used directly
  - `NonlinearVerbosity` (or similar types with `sensitivity_vjp_choice` field): checks the toggle
  - Other types (e.g. `SciMLLogging` presets): defaults to `true` for backward compatibility
"""
function _get_sensitivity_vjp_verbose(verbose)
    verbose isa Bool && return verbose
    if hasproperty(verbose, :sensitivity_vjp_choice)
        toggle = getproperty(verbose, :sensitivity_vjp_choice)
        return verbosity_to_bool(toggle)
    end
    return true
end
