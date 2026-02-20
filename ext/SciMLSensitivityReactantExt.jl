module SciMLSensitivityReactantExt

using SciMLSensitivity: SciMLSensitivity, SciMLBase
using Reactant: Reactant, ConcreteRArray
using Enzyme: Enzyme
import SciMLSensitivity: get_paramjac_config, reactant_run_ad, ReactantVJP, ReactantLoaded

# Creates a VJP kernel closure that captures the ODE function `raw_f`.
# The function must be captured (not passed as argument) because Reactant
# can only trace through functions whose identity is encoded in the closure type.
function _make_vjp_kernel(raw_f, isinplace::Bool)
    return function(dy_buf, u, p, t, λ)
        du_shadow = zero(u)
        dp_shadow = zero(p)
        dλ_seed = copy(λ)

        if isinplace
            Enzyme.autodiff(
                Enzyme.Reverse,
                Enzyme.Const(raw_f),
                Enzyme.Const,
                Enzyme.Duplicated(dy_buf, dλ_seed),
                Enzyme.Duplicated(u, du_shadow),
                Enzyme.Duplicated(p, dp_shadow),
                Enzyme.Const(t)
            )
        else
            # For out-of-place functions: f(u, p, t) -> dy
            # We wrap it so Enzyme can differentiate through the copyto!
            function oop_wrapper(dy_buf, u, p, t)
                result = raw_f(u, p, t)
                copyto!(dy_buf, result)
                return nothing
            end

            Enzyme.autodiff(
                Enzyme.Reverse,
                Enzyme.Const(oop_wrapper),
                Enzyme.Const,
                Enzyme.Duplicated(dy_buf, dλ_seed),
                Enzyme.Duplicated(u, du_shadow),
                Enzyme.Duplicated(p, dp_shadow),
                Enzyme.Const(t)
            )
        end

        return dy_buf, du_shadow, dp_shadow
    end
end

function get_paramjac_config(::ReactantLoaded, ::ReactantVJP, pf, p, f, y, _t)
    if p === nothing || p isa SciMLBase.NullParameters
        return (nothing, pf)
    end

    # Extract the raw ODE function for direct use in the kernel
    raw_f = SciMLBase.unwrapped_f(f)
    iip = SciMLBase.isinplace(f)

    # Create a VJP kernel that captures raw_f in its closure type
    vjp_kernel = _make_vjp_kernel(raw_f, iip)

    # Create example ConcreteRArrays for compilation
    dy_buf = ConcreteRArray(zero(y))
    u_ra = ConcreteRArray(zero(y))
    p_ra = ConcreteRArray(zero(p))
    t_ra = Reactant.to_rarray(Float64(_t))
    λ_ra = ConcreteRArray(zero(y))

    # Pre-compile the VJP kernel once. Reactant traces through the closure
    # (including Enzyme.autodiff) and compiles to XLA/HLO. The compiled function
    # is cached and reused for every subsequent VJP call during the solve.
    # Allow scalar indexing during tracing so that ODE functions with scalar
    # operations (e.g. x, y = u) can be compiled.
    compiled_fn = Reactant.@allowscalar Reactant.compile(
        vjp_kernel, (dy_buf, u_ra, p_ra, t_ra, λ_ra))

    return (compiled_fn, pf)
end

function reactant_run_ad(paramjac_config::Tuple, y, p, t, λ)
    compiled_fn, pf = paramjac_config

    if compiled_fn === nothing
        error("ReactantVJP does not support NullParameters")
    end

    # collect ensures SubArrays/views are converted to Array for ConcreteRArray
    y_arr = collect(y)
    p_arr = collect(p)
    λ_arr = collect(λ)
    y_ra = ConcreteRArray(y_arr)
    p_ra = ConcreteRArray(p_arr)
    λ_ra = ConcreteRArray(λ_arr)
    dy_ra = ConcreteRArray(zero(y_arr))
    t_ra = Reactant.to_rarray(Float64(t))

    dy, y_grad, p_grad = compiled_fn(dy_ra, y_ra, p_ra, t_ra, λ_ra)

    return Array(dy), Array(y_grad), Array(p_grad)
end

end # module
