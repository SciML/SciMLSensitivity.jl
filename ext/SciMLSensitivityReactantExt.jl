module SciMLSensitivityReactantExt

using SciMLSensitivity: SciMLSensitivity, SciMLBase, FakeIntegrator
using Reactant: Reactant, ConcreteRArray
using Enzyme: Enzyme
import SciMLSensitivity: get_paramjac_config, reactant_run_ad, ReactantVJP, ReactantLoaded,
    get_cb_paramjac_config, reactant_run_cb_ad

# =============================================================================
# ODE VJP kernels
# =============================================================================

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

    # Pre-allocate cached buffers to avoid per-call allocations in reactant_run_ad.
    # These are reused via copyto! instead of allocating new arrays each VJP call.
    y_cache = zero(y)
    p_cache = zero(p)
    λ_cache = zero(y)
    dy_cache = zero(y)

    return (compiled_fn, pf, y_cache, p_cache, λ_cache, dy_cache)
end

function reactant_run_ad(paramjac_config::Tuple, y, p, t, λ)
    compiled_fn, pf, y_cache, p_cache, λ_cache, dy_cache = paramjac_config

    if compiled_fn === nothing
        error("ReactantVJP does not support NullParameters")
    end

    # Copy into pre-allocated buffers (handles SubArrays/views without allocating)
    copyto!(y_cache, y)
    copyto!(p_cache, p)
    copyto!(λ_cache, λ)
    fill!(dy_cache, zero(eltype(dy_cache)))

    y_ra = ConcreteRArray(y_cache)
    p_ra = ConcreteRArray(p_cache)
    λ_ra = ConcreteRArray(λ_cache)
    dy_ra = ConcreteRArray(dy_cache)
    t_ra = Reactant.to_rarray(Float64(t))

    dy, y_grad, p_grad = compiled_fn(dy_ra, y_ra, p_ra, t_ra, λ_ra)

    return Array(dy), Array(y_grad), Array(p_grad)
end

# =============================================================================
# Callback VJP kernels
# =============================================================================

# State mode kernel: differentiates affect!(fakeinteg) w.r.t. u and p,
# extracting fakeinteg.u as the output (like CallbackAffectWrapper).
# raw_affect and event_idx are captured in the closure for Reactant tracing.
function _make_cb_state_vjp_kernel(raw_affect, event_idx)
    has_event_idx = event_idx !== nothing

    function cb_state_fn(out_buf, u, p, t, tprev)
        fakeinteg = FakeIntegrator(copy(u), copy(p), t, tprev)
        if has_event_idx
            raw_affect(fakeinteg, event_idx)
        else
            raw_affect(fakeinteg)
        end
        copyto!(out_buf, fakeinteg.u)
        return nothing
    end

    return function(out_buf, u, p, t, tprev, λ)
        du_shadow = zero(u)
        dp_shadow = zero(p)
        dλ_seed = copy(λ)

        Enzyme.autodiff(
            Enzyme.Reverse,
            Enzyme.Const(cb_state_fn),
            Enzyme.Const,
            Enzyme.Duplicated(out_buf, dλ_seed),
            Enzyme.Duplicated(u, du_shadow),
            Enzyme.Duplicated(p, dp_shadow),
            Enzyme.Const(t),
            Enzyme.Const(tprev)
        )

        return out_buf, du_shadow, dp_shadow
    end
end

# Param mode kernel: differentiates affect!(fakeinteg) w.r.t. u and p,
# extracting fakeinteg.p as the output (like CallbackAffectPWrapper).
function _make_cb_param_vjp_kernel(raw_affect, event_idx)
    has_event_idx = event_idx !== nothing

    function cb_param_fn(out_buf, u, p, t, tprev)
        fakeinteg = FakeIntegrator(copy(u), copy(p), t, tprev)
        if has_event_idx
            raw_affect(fakeinteg, event_idx)
        else
            raw_affect(fakeinteg)
        end
        copyto!(out_buf, fakeinteg.p)
        return nothing
    end

    return function(out_buf, u, p, t, tprev, λ)
        du_shadow = zero(u)
        dp_shadow = zero(p)
        dλ_seed = copy(λ)

        Enzyme.autodiff(
            Enzyme.Reverse,
            Enzyme.Const(cb_param_fn),
            Enzyme.Const,
            Enzyme.Duplicated(out_buf, dλ_seed),
            Enzyme.Duplicated(u, du_shadow),
            Enzyme.Duplicated(p, dp_shadow),
            Enzyme.Const(t),
            Enzyme.Const(tprev)
        )

        return out_buf, du_shadow, dp_shadow
    end
end

function get_cb_paramjac_config(
        ::ReactantLoaded, ::ReactantVJP, raw_affect, event_idx, y, p, _t, mode)
    if mode === :state
        kernel = _make_cb_state_vjp_kernel(raw_affect, event_idx)
        out_example = ConcreteRArray(zero(y))
        λ_example = ConcreteRArray(zero(y))
        out_cache = zero(y)
        λ_cache = zero(y)
    else # :param
        kernel = _make_cb_param_vjp_kernel(raw_affect, event_idx)
        out_example = ConcreteRArray(zero(p))
        λ_example = ConcreteRArray(zero(p))
        out_cache = zero(p)
        λ_cache = zero(p)
    end

    u_ra = ConcreteRArray(zero(y))
    p_ra = ConcreteRArray(zero(p))
    t_ra = Reactant.to_rarray(Float64(_t))
    tprev_ra = Reactant.to_rarray(Float64(_t))

    compiled_fn = Reactant.@allowscalar Reactant.compile(
        kernel, (out_example, u_ra, p_ra, t_ra, tprev_ra, λ_example))

    y_cache = zero(y)
    p_cache = zero(p)

    return (compiled_fn, nothing, y_cache, p_cache, λ_cache, out_cache)
end

function reactant_run_cb_ad(paramjac_config::Tuple, y, p, t, tprev, λ)
    compiled_fn, _, y_cache, p_cache, λ_cache, out_cache = paramjac_config

    if compiled_fn === nothing
        error("ReactantVJP callback config not initialized")
    end

    # Copy into pre-allocated buffers
    copyto!(y_cache, y)
    copyto!(p_cache, p)
    copyto!(λ_cache, λ)
    fill!(out_cache, zero(eltype(out_cache)))

    y_ra = ConcreteRArray(y_cache)
    p_ra = ConcreteRArray(p_cache)
    λ_ra = ConcreteRArray(λ_cache)
    out_ra = ConcreteRArray(out_cache)
    t_ra = Reactant.to_rarray(Float64(t))
    tprev_ra = Reactant.to_rarray(Float64(tprev))

    _out, y_grad, p_grad = compiled_fn(out_ra, y_ra, p_ra, t_ra, tprev_ra, λ_ra)

    return Array(_out), Array(y_grad), Array(p_grad)
end

end # module
