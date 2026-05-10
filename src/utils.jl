function isfunctor(x)
    return !(x isa AbstractArray) && !isscimlstructure(x) && !isempty(Functors.children(x))
end

function to_nt(s::T) where {T}
    return NamedTuple{propertynames(s)}(map(x -> getproperty(s, x), propertynames(s)))
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
