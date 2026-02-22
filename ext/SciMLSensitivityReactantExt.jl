module SciMLSensitivityReactantExt

using SciMLSensitivity: SciMLSensitivity, SciMLBase, FakeIntegrator
using Reactant: Reactant, ConcreteRArray
using Enzyme: Enzyme
import SciMLSensitivity: get_paramjac_config, reactant_run_ad!, ReactantVJP, ReactantLoaded,
    get_cb_paramjac_config, reactant_run_cb_ad!

# Helper: conditionally wrap Reactant.compile in @allowscalar
function _reactant_compile(kernel, args, allow_scalar::Bool)
    if allow_scalar
        return Reactant.@allowscalar Reactant.compile(kernel, args)
    else
        return Reactant.compile(kernel, args)
    end
end

# =============================================================================
# ODE VJP kernels
# =============================================================================

# Creates a VJP kernel closure that captures the ODE function `raw_f`.
# The function must be captured (not passed as argument) because Reactant
# can only trace through functions whose identity is encoded in the closure type.
function _make_vjp_kernel(raw_f, isinplace::Bool)
    return function (dy_buf, u, p, t, λ)
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

# Variant for NullParameters: params are Const (not differentiated).
# Returns only (dy_buf, du_shadow) since there is no param gradient.
function _make_vjp_kernel_nullparams(raw_f, isinplace::Bool)
    return function (dy_buf, u, t, λ)
        du_shadow = zero(u)
        dλ_seed = copy(λ)

        if isinplace
            Enzyme.autodiff(
                Enzyme.Reverse,
                Enzyme.Const(raw_f),
                Enzyme.Const,
                Enzyme.Duplicated(dy_buf, dλ_seed),
                Enzyme.Duplicated(u, du_shadow),
                Enzyme.Const(SciMLBase.NullParameters()),
                Enzyme.Const(t)
            )
        else
            function oop_wrapper(dy_buf, u, t)
                result = raw_f(u, SciMLBase.NullParameters(), t)
                copyto!(dy_buf, result)
                return nothing
            end

            Enzyme.autodiff(
                Enzyme.Reverse,
                Enzyme.Const(oop_wrapper),
                Enzyme.Const,
                Enzyme.Duplicated(dy_buf, dλ_seed),
                Enzyme.Duplicated(u, du_shadow),
                Enzyme.Const(t)
            )
        end

        return dy_buf, du_shadow
    end
end

function get_paramjac_config(::ReactantLoaded, vjp::ReactantVJP, pf, p, f, y, _t)
    # Extract the raw ODE function for direct use in the kernel
    raw_f = SciMLBase.unwrapped_f(f)
    iip = SciMLBase.isinplace(f)

    # Steady-state problems pass _t = nothing; use 0.0 as a placeholder.
    t_val = _t === nothing ? 0.0 : Float64(_t)

    if p === nothing || p isa SciMLBase.NullParameters
        # NullParameters: compile a kernel that doesn't differentiate w.r.t. p.
        vjp_kernel = _make_vjp_kernel_nullparams(raw_f, iip)

        dy_buf = ConcreteRArray(zero(y))
        u_ra = ConcreteRArray(zero(y))
        t_ra = Reactant.to_rarray(t_val; track_numbers = true)
        λ_ra = ConcreteRArray(zero(y))

        compiled_fn = _reactant_compile(
            vjp_kernel, (dy_buf, u_ra, t_ra, λ_ra), vjp.allow_scalar
        )

        y_cache = zero(y)
        λ_cache = zero(y)
        dy_cache = zero(y)
        dy_result = zero(y)
        ygrad_result = zero(y)

        # :nullparams tag in slot 2 signals reactant_run_ad! to use the nullparams path
        return (
            compiled_fn, :nullparams, y_cache, nothing, λ_cache, dy_cache,
            dy_result, ygrad_result, nothing,
        )
    end

    # Create a VJP kernel that captures raw_f in its closure type
    vjp_kernel = _make_vjp_kernel(raw_f, iip)

    # Create example ConcreteRArrays for compilation
    dy_buf = ConcreteRArray(zero(y))
    u_ra = ConcreteRArray(zero(y))
    p_ra = ConcreteRArray(zero(p))
    t_ra = Reactant.to_rarray(t_val; track_numbers = true)
    λ_ra = ConcreteRArray(zero(y))

    # Pre-compile the VJP kernel once. Reactant traces through the closure
    # (including Enzyme.autodiff) and compiles to XLA/HLO. The compiled function
    # is cached and reused for every subsequent VJP call during the solve.
    compiled_fn = _reactant_compile(
        vjp_kernel, (dy_buf, u_ra, p_ra, t_ra, λ_ra), vjp.allow_scalar
    )

    # Pre-allocate cached buffers to avoid per-call allocations.
    # Input caches: reused via copyto! to handle SubArrays/views.
    # Result caches: landing zone for device-to-host transfer from ConcreteRArray.
    y_cache = zero(y)
    p_cache = zero(p)
    λ_cache = zero(y)
    dy_cache = zero(y)
    dy_result = zero(y)
    ygrad_result = zero(y)
    pgrad_result = zero(p)

    return (
        compiled_fn, pf, y_cache, p_cache, λ_cache, dy_cache,
        dy_result, ygrad_result, pgrad_result,
    )
end

function reactant_run_ad!(dλ, dgrad, dy, paramjac_config::Tuple, y, p, t, λ)
    compiled_fn, tag, y_cache, p_cache, λ_cache, dy_cache,
        dy_result, ygrad_result, pgrad_result = paramjac_config

    if tag === :nullparams
        # NullParameters path: kernel takes (dy_buf, u, t, λ) — no p argument
        copyto!(y_cache, y)
        copyto!(λ_cache, λ)
        fill!(dy_cache, zero(eltype(dy_cache)))

        y_ra = ConcreteRArray(y_cache)
        λ_ra = ConcreteRArray(λ_cache)
        dy_ra = ConcreteRArray(dy_cache)
        t_val = t === nothing ? 0.0 : Float64(t)
        t_ra = Reactant.to_rarray(t_val; track_numbers = true)

        dy_out, y_grad = compiled_fn(dy_ra, y_ra, t_ra, λ_ra)

        copyto!(dy_result, dy_out)
        copyto!(ygrad_result, y_grad)
        dy !== nothing && copyto!(dy, dy_result)
        dλ !== nothing && copyto!(dλ, ygrad_result)
        # dgrad stays untouched — no param gradient for NullParameters
        return nothing
    end

    # Normal path with parameters
    copyto!(y_cache, y)
    copyto!(p_cache, p)
    copyto!(λ_cache, λ)
    fill!(dy_cache, zero(eltype(dy_cache)))

    y_ra = ConcreteRArray(y_cache)
    p_ra = ConcreteRArray(p_cache)
    λ_ra = ConcreteRArray(λ_cache)
    dy_ra = ConcreteRArray(dy_cache)
    t_val = t === nothing ? 0.0 : Float64(t)
    t_ra = Reactant.to_rarray(t_val; track_numbers = true)

    dy_out, y_grad, p_grad = compiled_fn(dy_ra, y_ra, p_ra, t_ra, λ_ra)

    copyto!(dy_result, dy_out)
    copyto!(ygrad_result, y_grad)
    copyto!(pgrad_result, p_grad)
    dy !== nothing && copyto!(dy, dy_result)
    dλ !== nothing && copyto!(dλ, ygrad_result)
    dgrad !== nothing && copyto!(dgrad, pgrad_result)
    return nothing
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

    return function (out_buf, u, p, t, tprev, λ)
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

    return function (out_buf, u, p, t, tprev, λ)
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
        ::ReactantLoaded, vjp::ReactantVJP, raw_affect, event_idx, y, p, _t, mode
    )
    if mode === :state
        kernel = _make_cb_state_vjp_kernel(raw_affect, event_idx)
        out_example = ConcreteRArray(zero(y))
        λ_example = ConcreteRArray(zero(y))
        out_cache = zero(y)
        λ_cache = zero(y)
        out_result = zero(y)
        ygrad_result = zero(y)
    else # :param
        kernel = _make_cb_param_vjp_kernel(raw_affect, event_idx)
        out_example = ConcreteRArray(zero(p))
        λ_example = ConcreteRArray(zero(p))
        out_cache = zero(p)
        λ_cache = zero(p)
        out_result = zero(p)
        ygrad_result = zero(y)
    end

    cb_t_val = _t === nothing ? 0.0 : Float64(_t)
    u_ra = ConcreteRArray(zero(y))
    p_ra = ConcreteRArray(zero(p))
    t_ra = Reactant.to_rarray(cb_t_val; track_numbers = true)
    tprev_ra = Reactant.to_rarray(cb_t_val; track_numbers = true)

    compiled_fn = _reactant_compile(
        kernel, (out_example, u_ra, p_ra, t_ra, tprev_ra, λ_example), vjp.allow_scalar
    )

    y_cache = zero(y)
    p_cache = zero(p)
    pgrad_result = zero(p)

    return (
        compiled_fn, nothing, y_cache, p_cache, λ_cache, out_cache,
        out_result, ygrad_result, pgrad_result,
    )
end

function reactant_run_cb_ad!(dλ, dgrad, dy, paramjac_config::Tuple, y, p, t, tprev, λ)
    compiled_fn, _, y_cache, p_cache, λ_cache, out_cache,
        out_result, ygrad_result, pgrad_result = paramjac_config

    if compiled_fn === nothing
        error("ReactantVJP callback config not initialized")
    end

    # Copy into pre-allocated input buffers
    copyto!(y_cache, y)
    copyto!(p_cache, p)
    copyto!(λ_cache, λ)
    fill!(out_cache, zero(eltype(out_cache)))

    y_ra = ConcreteRArray(y_cache)
    p_ra = ConcreteRArray(p_cache)
    λ_ra = ConcreteRArray(λ_cache)
    out_ra = ConcreteRArray(out_cache)
    t_val = t === nothing ? 0.0 : Float64(t)
    tprev_val = tprev === nothing ? 0.0 : Float64(tprev)
    t_ra = Reactant.to_rarray(t_val; track_numbers = true)
    tprev_ra = Reactant.to_rarray(tprev_val; track_numbers = true)

    _out, y_grad, p_grad = compiled_fn(out_ra, y_ra, p_ra, t_ra, tprev_ra, λ_ra)

    # Device-to-host into pre-allocated result caches, then into caller's buffers
    copyto!(out_result, _out)
    copyto!(ygrad_result, y_grad)
    copyto!(pgrad_result, p_grad)
    dy !== nothing && copyto!(dy, out_result)
    dλ !== nothing && copyto!(dλ, ygrad_result)
    dgrad !== nothing && copyto!(dgrad, pgrad_result)
    return nothing
end

end # module
