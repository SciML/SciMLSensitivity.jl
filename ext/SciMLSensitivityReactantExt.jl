module SciMLSensitivityReactantExt

using SciMLSensitivity: SciMLSensitivity, SciMLBase, FakeIntegrator
using Reactant: Reactant, ConcreteRArray
using Enzyme: Enzyme
using ForwardDiff: ForwardDiff
import SciMLSensitivity: get_paramjac_config, reactant_run_ad!, reactant_run_dual_ad!,
    ReactantVJP, ReactantLoaded, ReactantVJPConfig, ReactantDualTag,
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
    if isinplace
        # Wrap in SciMLBase.Void so the in-place function returns nothing
        # (matching EnzymeVJP's approach). Pass as Duplicated so Enzyme can
        # differentiate through mutable state captured in the closure.
        void_f = SciMLBase.Void(raw_f)
        f_shadow = Enzyme.make_zero(void_f)
        return function (dy_buf, u, p, t, λ)
            du_shadow = zero(u)
            dp_shadow = zero(p)
            dλ_seed = copy(λ)

            Enzyme.autodiff(
                Enzyme.Reverse,
                Enzyme.Duplicated(void_f, f_shadow),
                Enzyme.Const,
                Enzyme.Duplicated(dy_buf, dλ_seed),
                Enzyme.Duplicated(u, du_shadow),
                Enzyme.Duplicated(p, dp_shadow),
                Enzyme.Const(t)
            )

            return dy_buf, du_shadow, dp_shadow
        end
    else
        # For out-of-place functions: use a fixed wrapper (Const) and pass
        # raw_f as a Duplicated argument so Enzyme tracks its captured state.
        f_shadow = Enzyme.make_zero(raw_f)
        return function (dy_buf, u, p, t, λ)
            du_shadow = zero(u)
            dp_shadow = zero(p)
            dλ_seed = copy(λ)

            function oop_wrapper(f, dy_buf, u, p, t)
                result = f(u, p, t)
                copyto!(dy_buf, result)
                return nothing
            end

            Enzyme.autodiff(
                Enzyme.Reverse,
                Enzyme.Const(oop_wrapper),
                Enzyme.Const,
                Enzyme.Duplicated(raw_f, f_shadow),
                Enzyme.Duplicated(dy_buf, dλ_seed),
                Enzyme.Duplicated(u, du_shadow),
                Enzyme.Duplicated(p, dp_shadow),
                Enzyme.Const(t)
            )

            return dy_buf, du_shadow, dp_shadow
        end
    end
end

# Variant for NullParameters: params are Const (not differentiated).
# Returns only (dy_buf, du_shadow) since there is no param gradient.
function _make_vjp_kernel_nullparams(raw_f, isinplace::Bool)
    if isinplace
        void_f = SciMLBase.Void(raw_f)
        f_shadow = Enzyme.make_zero(void_f)
        return function (dy_buf, u, t, λ)
            du_shadow = zero(u)
            dλ_seed = copy(λ)

            Enzyme.autodiff(
                Enzyme.Reverse,
                Enzyme.Duplicated(void_f, f_shadow),
                Enzyme.Const,
                Enzyme.Duplicated(dy_buf, dλ_seed),
                Enzyme.Duplicated(u, du_shadow),
                Enzyme.Const(SciMLBase.NullParameters()),
                Enzyme.Const(t)
            )

            return dy_buf, du_shadow
        end
    else
        f_shadow = Enzyme.make_zero(raw_f)
        return function (dy_buf, u, t, λ)
            du_shadow = zero(u)
            dλ_seed = copy(λ)

            function oop_wrapper(f, dy_buf, u, t)
                result = f(u, SciMLBase.NullParameters(), t)
                copyto!(dy_buf, result)
                return nothing
            end

            Enzyme.autodiff(
                Enzyme.Reverse,
                Enzyme.Const(oop_wrapper),
                Enzyme.Const,
                Enzyme.Duplicated(raw_f, f_shadow),
                Enzyme.Duplicated(dy_buf, dλ_seed),
                Enzyme.Duplicated(u, du_shadow),
                Enzyme.Const(t)
            )

            return dy_buf, du_shadow
        end
    end
end

# =============================================================================
# Compilation helpers
# =============================================================================

function _compile_float_kernel(raw_f, iip, vjp, y, p, t_val)
    vjp_kernel = _make_vjp_kernel(raw_f, iip)
    dy_buf = ConcreteRArray(zero(y))
    u_ra = ConcreteRArray(zero(y))
    p_ra = ConcreteRArray(zero(p))
    t_ra = Reactant.to_rarray(t_val; track_numbers = true)
    λ_ra = ConcreteRArray(zero(y))
    return _reactant_compile(vjp_kernel, (dy_buf, u_ra, p_ra, t_ra, λ_ra), vjp.allow_scalar)
end

function _compile_float_kernel_nullparams(raw_f, iip, vjp, y, t_val)
    vjp_kernel = _make_vjp_kernel_nullparams(raw_f, iip)
    dy_buf = ConcreteRArray(zero(y))
    u_ra = ConcreteRArray(zero(y))
    t_ra = Reactant.to_rarray(t_val; track_numbers = true)
    λ_ra = ConcreteRArray(zero(y))
    return _reactant_compile(vjp_kernel, (dy_buf, u_ra, t_ra, λ_ra), vjp.allow_scalar)
end

# =============================================================================
# get_paramjac_config — returns ReactantVJPConfig
# =============================================================================

function get_paramjac_config(
        ::ReactantLoaded, vjp::ReactantVJP, pf, p, f, y, _t;
        numindvar = nothing, alg = nothing
    )
    raw_f = SciMLBase.unwrapped_f(f)
    iip = SciMLBase.isinplace(f)
    t_val = _t === nothing ? 0.0 : Float64(_t)
    N = length(y)
    is_nullparams = (p === nothing || p isa SciMLBase.NullParameters)
    NP = is_nullparams ? 0 : length(p)

    float_kernel = if is_nullparams
        _compile_float_kernel_nullparams(raw_f, iip, vjp, y, t_val)
    else
        _compile_float_kernel(raw_f, iip, vjp, y, p, t_val)
    end

    float_caches = if is_nullparams
        (
            y_cache = zero(y),
            λ_cache = zero(y),
            dy_cache = zero(y),
            dy_result = zero(y),
            ygrad_result = zero(y),
        )
    else
        (
            y_cache = zero(y),
            p_cache = zero(p),
            λ_cache = zero(y),
            dy_cache = zero(y),
            dy_result = zero(y),
            ygrad_result = zero(y),
            pgrad_result = zero(p),
        )
    end

    # Pre-allocate dual caches if a stiff solver will push ForwardDiff.Dual
    # numbers through the VJP path. The adjoint state dimension determines the
    # chunk size: for InterpolatingAdjoint it is numindvar + numparams.
    dual_caches, CS = if alg !== nothing && numindvar !== nothing &&
            SciMLBase.forwarddiffs_model(alg)
        adjoint_dim = numindvar + NP
        _CS = ForwardDiff.pickchunksize(adjoint_dim)
        dc = if is_nullparams
            (
                y_float_cache = Vector{Float64}(undef, N),
                λ_val_cache = Vector{Float64}(undef, N),
                λ_part_cache = Vector{Float64}(undef, N),
                dλ_val = Vector{Float64}(undef, N),
                dy_val = Vector{Float64}(undef, N),
                dλ_parts = Matrix{Float64}(undef, N, _CS),
                dy_parts = Matrix{Float64}(undef, N, _CS),
                single_dy = Vector{Float64}(undef, N),
                single_ygrad = Vector{Float64}(undef, N),
            )
        else
            (
                y_float_cache = Vector{Float64}(undef, N),
                λ_val_cache = Vector{Float64}(undef, N),
                λ_part_cache = Vector{Float64}(undef, N),
                dλ_val = Vector{Float64}(undef, N),
                dy_val = Vector{Float64}(undef, N),
                pgrad_val = Vector{Float64}(undef, NP),
                dλ_parts = Matrix{Float64}(undef, N, _CS),
                dy_parts = Matrix{Float64}(undef, N, _CS),
                pgrad_parts = Matrix{Float64}(undef, NP, _CS),
                single_dy = Vector{Float64}(undef, N),
                single_ygrad = Vector{Float64}(undef, N),
                single_pgrad = Vector{Float64}(undef, NP),
            )
        end
        dc, _CS
    else
        nothing, 0
    end

    return ReactantVJPConfig(
        float_kernel, nothing, float_caches, dual_caches,
        is_nullparams, CS
    )
end

# =============================================================================
# reactant_run_ad! — dispatches on ReactantVJPConfig
# =============================================================================

function reactant_run_ad!(dλ, dgrad, dy, config::ReactantVJPConfig, y, p, t, λ)
    _has_duals = eltype(λ) <: ForwardDiff.Dual ||
        (dλ !== nothing && eltype(dλ) <: ForwardDiff.Dual)

    if _has_duals
        return reactant_run_dual_ad!(dλ, dgrad, dy, config, y, p, t, λ)
    end

    if config.is_nullparams
        _run_float_nullparams!(dλ, dgrad, dy, config, y, t, λ)
    else
        _run_float_params!(dλ, dgrad, dy, config, y, p, t, λ)
    end
    return nothing
end

function _run_float_nullparams!(dλ, dgrad, dy, config, y, t, λ)
    fc = config.float_caches
    _run_single_float_call!(
        fc.ygrad_result, fc.dy_result, nothing,
        config, y, nothing, t, λ
    )
    dy !== nothing && copyto!(dy, fc.dy_result)
    dλ !== nothing && copyto!(dλ, fc.ygrad_result)
    return nothing
end

function _run_float_params!(dλ, dgrad, dy, config, y, p, t, λ)
    fc = config.float_caches
    _run_single_float_call!(
        fc.ygrad_result, fc.dy_result, fc.pgrad_result,
        config, y, p, t, λ
    )
    dy !== nothing && copyto!(dy, fc.dy_result)
    dλ !== nothing && copyto!(dλ, fc.ygrad_result)
    dgrad !== nothing && copyto!(dgrad, fc.pgrad_result)
    return nothing
end

# =============================================================================
# reactant_run_dual_ad! — handles ForwardDiff.Dual inputs via linearity of VJP
#
# Key insight: In the adjoint ODE, only λ (and possibly t) carry Dual perturbations.
# y and p are Float64 (from the interpolated forward solution).
# Since the VJP is LINEAR in λ:
#   VJP(f, y, p, t, λ_val + Σεᵢ*λ_partᵢ) = VJP(f, y, p, t, λ_val) + Σεᵢ*VJP(f, y, p, t, λ_partᵢ)
#
# So we call the existing Float64 kernel (1 + chunk_size) times:
#   once for the value, once per partial direction.
# =============================================================================

function reactant_run_dual_ad!(dλ, dgrad, dy, config::ReactantVJPConfig, y, p, t, λ)
    # Determine the Dual type and chunk size from actual inputs
    DualType = if eltype(λ) <: ForwardDiff.Dual
        eltype(λ)
    elseif dλ !== nothing && eltype(dλ) <: ForwardDiff.Dual
        eltype(dλ)
    else
        error("reactant_run_dual_ad! called but no Dual types found")
    end
    CS = ForwardDiff.npartials(DualType)

    # Extract Float64 values from inputs
    t_float = t isa ForwardDiff.Dual ? ForwardDiff.value(t) : t
    N = length(y)

    dc = config.dual_caches
    # Use pre-allocated caches if available and chunk size matches
    if dc !== nothing && config.chunk_size == CS
        _reactant_run_dual_preallocated!(
            dλ, dgrad, dy, config, dc, y, p,
            t_float, λ, DualType, N, CS
        )
    else
        _reactant_run_dual_fallback!(
            dλ, dgrad, dy, config, y, p,
            t_float, λ, DualType, N, CS
        )
    end

    return nothing
end

# Fast path: use pre-allocated dual caches
function _reactant_run_dual_preallocated!(
        dλ, dgrad, dy, config, dc, y, p, t_float, λ,
        ::Type{DualType}, N, CS
    ) where {DualType}
    y_float = dc.y_float_cache
    if eltype(y) <: ForwardDiff.Dual
        for i in 1:N
            y_float[i] = ForwardDiff.value(y[i])
        end
    else
        copyto!(y_float, y)
    end

    λ_val = dc.λ_val_cache
    if eltype(λ) <: ForwardDiff.Dual
        for i in 1:N
            λ_val[i] = ForwardDiff.value(λ[i])
        end
    else
        copyto!(λ_val, λ)
    end

    # --- Call 1: Float64 kernel with λ_value ---
    dλ_val = dc.dλ_val
    dy_v = dc.dy_val
    pgrad_v = config.is_nullparams ? nothing : dc.pgrad_val
    _run_single_float_call!(dλ_val, dy_v, pgrad_v, config, y_float, p, t_float, λ_val)

    # --- Calls 2..CS+1: Float64 kernel with each λ_partial direction ---
    dλ_parts = dc.dλ_parts
    dy_parts = dc.dy_parts
    pgrad_parts = config.is_nullparams ? nothing : dc.pgrad_parts
    fill!(dλ_parts, zero(Float64))
    fill!(dy_parts, zero(Float64))
    pgrad_parts !== nothing && fill!(pgrad_parts, zero(Float64))

    λ_part = dc.λ_part_cache
    single_dy = dc.single_dy
    single_ygrad = dc.single_ygrad
    single_pgrad = config.is_nullparams ? nothing : dc.single_pgrad

    if eltype(λ) <: ForwardDiff.Dual
        for j in 1:CS
            for i in 1:N
                λ_part[i] = ForwardDiff.partials(λ[i], j)
            end

            _run_single_float_call!(
                single_ygrad, single_dy, single_pgrad,
                config, y_float, p, t_float, λ_part
            )

            for i in 1:N
                dλ_parts[i, j] = single_ygrad[i]
                dy_parts[i, j] = single_dy[i]
            end
            if !config.is_nullparams
                NP = length(p)
                for i in 1:NP
                    pgrad_parts[i, j] = single_pgrad[i]
                end
            end
        end
    end

    _reconstruct_dual_outputs!(
        dλ, dy, dgrad, dλ_val, dy_v, pgrad_v,
        dλ_parts, dy_parts, pgrad_parts, DualType, config.is_nullparams
    )
    return nothing
end

# Fallback path: allocate temporaries when pre-allocated caches don't match
function _reactant_run_dual_fallback!(
        dλ, dgrad, dy, config, y, p, t_float, λ,
        ::Type{DualType}, N, CS
    ) where {DualType}
    y_float = Vector{Float64}(undef, N)
    if eltype(y) <: ForwardDiff.Dual
        for i in 1:N
            y_float[i] = ForwardDiff.value(y[i])
        end
    else
        copyto!(y_float, y)
    end

    λ_val = Vector{Float64}(undef, N)
    if eltype(λ) <: ForwardDiff.Dual
        for i in 1:N
            λ_val[i] = ForwardDiff.value(λ[i])
        end
    else
        copyto!(λ_val, λ)
    end

    dλ_val_buf = Vector{Float64}(undef, N)
    dy_val_buf = Vector{Float64}(undef, N)
    NP = config.is_nullparams ? 0 : length(p)
    pgrad_val_buf = config.is_nullparams ? nothing : Vector{Float64}(undef, NP)
    _run_single_float_call!(
        dλ_val_buf, dy_val_buf, pgrad_val_buf,
        config, y_float, p, t_float, λ_val
    )

    dλ_parts = zeros(Float64, N, CS)
    dy_parts = zeros(Float64, N, CS)
    pgrad_parts = config.is_nullparams ? nothing : zeros(Float64, NP, CS)

    λ_part = Vector{Float64}(undef, N)
    s_dy = Vector{Float64}(undef, N)
    s_ygrad = Vector{Float64}(undef, N)
    s_pgrad = config.is_nullparams ? nothing : Vector{Float64}(undef, NP)

    if eltype(λ) <: ForwardDiff.Dual
        for j in 1:CS
            for i in 1:N
                λ_part[i] = ForwardDiff.partials(λ[i], j)
            end

            _run_single_float_call!(
                s_ygrad, s_dy, s_pgrad,
                config, y_float, p, t_float, λ_part
            )

            for i in 1:N
                dλ_parts[i, j] = s_ygrad[i]
                dy_parts[i, j] = s_dy[i]
            end
            if !config.is_nullparams
                for i in 1:NP
                    pgrad_parts[i, j] = s_pgrad[i]
                end
            end
        end
    end

    _reconstruct_dual_outputs!(
        dλ, dy, dgrad, dλ_val_buf, dy_val_buf, pgrad_val_buf,
        dλ_parts, dy_parts, pgrad_parts, DualType, config.is_nullparams
    )
    return nothing
end

# Type-parameterized helper for Dual reconstruction. N is a type parameter, so
# Val(N) is resolved at compile time for ntuple.
function _reconstruct_dual_outputs!(
        dλ, dy, dgrad, dλ_val, dy_val, pgrad_val,
        dλ_parts, dy_parts, pgrad_parts,
        ::Type{ForwardDiff.Dual{Tag, V, N}}, is_nullparams
    ) where {Tag, V, N}
    if dλ !== nothing
        for i in eachindex(dλ_val)
            p = ForwardDiff.Partials(ntuple(j -> dλ_parts[i, j], Val(N)))
            dλ[i] = ForwardDiff.Dual{Tag, Float64, N}(dλ_val[i], p)
        end
    end
    if dy !== nothing
        for i in eachindex(dy_val)
            p = ForwardDiff.Partials(ntuple(j -> dy_parts[i, j], Val(N)))
            dy[i] = ForwardDiff.Dual{Tag, Float64, N}(dy_val[i], p)
        end
    end
    if dgrad !== nothing && !is_nullparams
        for i in eachindex(dgrad)
            p = ForwardDiff.Partials(ntuple(j -> pgrad_parts[i, j], Val(N)))
            dgrad[i] = ForwardDiff.Dual{Tag, Float64, N}(pgrad_val[i], p)
        end
    end
    return nothing
end

# Helper: run a single Float64 kernel call, writing results into provided buffers.
# ygrad_out receives ∂L/∂u, dy_out receives f(u,p,t), pgrad_out receives ∂L/∂p.
function _run_single_float_call!(
        ygrad_out, dy_out, pgrad_out, config, y_float, p,
        t_float, λ_float
    )
    fc = config.float_caches
    copyto!(fc.y_cache, y_float)
    copyto!(fc.λ_cache, λ_float)
    fill!(fc.dy_cache, zero(Float64))

    y_ra = ConcreteRArray(fc.y_cache)
    λ_ra = ConcreteRArray(fc.λ_cache)
    dy_ra = ConcreteRArray(fc.dy_cache)
    t_val = t_float === nothing ? 0.0 : Float64(t_float)
    t_ra = Reactant.to_rarray(t_val; track_numbers = true)

    if config.is_nullparams
        dy_result, y_grad = config.float_kernel(dy_ra, y_ra, t_ra, λ_ra)
        copyto!(dy_out, Array(dy_result))
        copyto!(ygrad_out, Array(y_grad))
    else
        copyto!(fc.p_cache, p)
        p_ra = ConcreteRArray(fc.p_cache)
        dy_result, y_grad, p_grad = config.float_kernel(dy_ra, y_ra, p_ra, t_ra, λ_ra)
        copyto!(dy_out, Array(dy_result))
        copyto!(ygrad_out, Array(y_grad))
        copyto!(pgrad_out, Array(p_grad))
    end
    return nothing
end

# =============================================================================
# Callback VJP kernels
# =============================================================================

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

    copyto!(out_result, Array(_out))
    copyto!(ygrad_result, Array(y_grad))
    copyto!(pgrad_result, Array(p_grad))
    dy !== nothing && copyto!(dy, out_result)
    dλ !== nothing && copyto!(dλ, ygrad_result)
    dgrad !== nothing && copyto!(dgrad, pgrad_result)
    return nothing
end

end # module
