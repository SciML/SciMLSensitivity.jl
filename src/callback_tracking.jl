"""
Appends a tracking process to determine the time of the callback to be used in
the reverse pass. The rationale is explain in:

https://github.com/SciML/SciMLSensitivity.jl/issues/4
"""
track_callbacks(cb, t, u, p, sensealg) = track_callbacks(CallbackSet(cb), t, u, p, sensealg)
function track_callbacks(cb::CallbackSet, t, u, p, sensealg)
    return CallbackSet(
        map(cb -> _track_callback(cb, t, u, p, sensealg), cb.continuous_callbacks),
        map(cb -> _track_callback(cb, t, u, p, sensealg), cb.discrete_callbacks)
    )
end

mutable struct ImplicitCorrection{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, RefType}
    gt_val::T1
    gu_val::T2
    gt::T3
    gu::T4
    gt_conf::T5
    gu_conf::T6
    condition::T7
    Lu_left::T8
    Lu_right::T9
    dy_left::T10
    dy_right::T11
    cur_time::RefType # initialized as "dummy" Ref that gets overwritten by Ref of loss
    terminated::Bool
end

ImplicitCorrection(cb::DiscreteCallback, t, u, p, sensealg) = nothing
function ImplicitCorrection(cb, t, u, p, sensealg)
    condition = cb.condition

    gt_val = similar(u, 1)
    gt_val .= 0
    gu_val = zero(u)

    fakeinteg = FakeIntegrator(u, p, t, t)
    gt, gu = build_condition_wrappers(cb, condition, u, t, fakeinteg)

    gt_conf = build_deriv_config(sensealg, gt, gt_val, t)
    gu_conf = build_grad_config(sensealg, gu, u, p)

    dy_left = zero(u)
    dy_right = zero(u)

    Lu_left = zero(u)
    Lu_right = zero(u)

    cur_time = Ref(1) # initialize the Ref, set to Ref of loss below
    terminated = false

    return ImplicitCorrection(
        gt_val, gu_val, gt, gu, gt_conf, gu_conf, condition, Lu_left,
        Lu_right, dy_left, dy_right, cur_time, terminated
    )
end

struct TrackedAffect{T, T2, T3, T4, T5, T6}
    event_times::Vector{T}
    tprev::Vector{T}
    uleft::Vector{T2}
    pleft::Vector{T3}
    affect!::T4
    correction::T5
    event_idx::Vector{T6}
end

TrackedAffect(t::Number, u, p, affect!::Nothing, correction) = nothing
function TrackedAffect(t::Number, u, p, affect!, correction)
    return TrackedAffect(
        Vector{typeof(t)}(undef, 0), Vector{typeof(t)}(undef, 0),
        Vector{typeof(u)}(undef, 0), Vector{typeof(p)}(undef, 0), affect!,
        correction,
        Vector{Int}(undef, 0)
    )
end

function Base.hasproperty(f::TrackedAffect, s::Symbol)
    if hasfield(TrackedAffect, s)
        return true
    else
        _affect = getfield(f, :affect!)
        return hasfield(typeof(_affect), s)
    end
end

function Base.getproperty(f::TrackedAffect, s::Symbol)
    if hasfield(TrackedAffect, s)
        return getfield(f, s)
    else
        _affect = getfield(f, :affect!)
        return getfield(_affect, s)
    end
end

function Base.setproperty!(f::TrackedAffect, s::Symbol, value)
    if hasfield(TrackedAffect, s)
        return setfield!(f, s, value)
    else
        _affect = getfield(f, :affect!)
        return setfield!(_affect, s, value)
    end
end

function (f::TrackedAffect)(integrator, event_idx = nothing)
    uleft = deepcopy(integrator.u)
    pleft = deepcopy(integrator.p)
    if event_idx === nothing
        f.affect!(integrator)
    else
        f.affect!(integrator, event_idx)
    end
    return if integrator.u_modified
        if isempty(f.event_times)
            push!(f.event_times, integrator.t)
            push!(f.tprev, integrator.tprev)
            push!(f.uleft, uleft)
            push!(f.pleft, pleft)
            if event_idx !== nothing
                push!(f.event_idx, event_idx)
            end
        else
            if !maximum(.≈(integrator.t, f.event_times, rtol = 0.0, atol = 1.0e-14))
                push!(f.event_times, integrator.t)
                push!(f.tprev, integrator.tprev)
                push!(f.uleft, uleft)
                push!(f.pleft, pleft)
                if event_idx !== nothing
                    push!(f.event_idx, event_idx)
                end
            end
        end
    end
end

function _track_callback(cb::DiscreteCallback, t, u, p, sensealg)
    correction = ImplicitCorrection(cb, t, u, p, sensealg)
    return DiscreteCallback(
        cb.condition,
        TrackedAffect(t, u, p, cb.affect!, correction),
        cb.initialize,
        cb.finalize,
        cb.save_positions
    )
end

function _track_callback(cb::ContinuousCallback, t, u, p, sensealg)
    correction = ImplicitCorrection(cb, t, u, p, sensealg)
    return ContinuousCallback(
        cb.condition,
        TrackedAffect(t, u, p, cb.affect!, correction),
        TrackedAffect(t, u, p, cb.affect_neg!, correction),
        cb.initialize,
        cb.finalize,
        cb.idxs,
        cb.rootfind, cb.interp_points,
        cb.save_positions,
        cb.dtrelax, cb.abstol, cb.reltol, cb.repeat_nudge
    )
end

function _track_callback(cb::VectorContinuousCallback, t, u, p, sensealg)
    correction = ImplicitCorrection(cb, t, u, p, sensealg)
    return VectorContinuousCallback(
        cb.condition,
        TrackedAffect(t, u, p, cb.affect!, correction),
        TrackedAffect(t, u, p, cb.affect_neg!, correction),
        cb.len, cb.initialize, cb.finalize, cb.idxs,
        cb.rootfind, cb.interp_points,
        collect(cb.save_positions),
        cb.dtrelax, cb.abstol, cb.reltol, cb.repeat_nudge
    )
end

struct FakeIntegrator{uType, P, tType, tprevType}
    u::uType
    p::P
    t::tType
    tprev::tprevType
end

function Base.getproperty(fi::FakeIntegrator, s::Symbol)
    s === :tdir && return sign(fi.t - fi.tprev)
    return getfield(fi, s)
end

struct CallbackSensitivityFunction{
        fType, Alg <: AbstractOverloadingSensitivityAlgorithm,
        C <: AdjointDiffCache, pType,
    } <: SensitivityFunction
    f::fType
    sensealg::Alg
    diffcache::C
    prob::pType
end

getprob(S::CallbackSensitivityFunction) = S.prob
inplace_sensitivity(S::CallbackSensitivityFunction) = true

struct CallbackSensitivityFunctionPSwap{
        fType, Alg <: AbstractOverloadingSensitivityAlgorithm,
        C <: AdjointDiffCache, pType,
    } <: SensitivityFunction
    f::fType
    sensealg::Alg
    diffcache::C
    prob::pType
end

getprob(S::CallbackSensitivityFunctionPSwap) = S.prob
inplace_sensitivity(S::CallbackSensitivityFunctionPSwap) = true

"""
Sets up callbacks for the adjoint pass. This is a version that has an effect
at each event of the forward pass and defines the reverse pass values via the
vjps as described in https://arxiv.org/pdf/1905.10403.pdf Equation 13.

For more information, see https://github.com/SciML/SciMLSensitivity.jl/issues/4
"""
function setup_reverse_callbacks(cb, sensealg, dgdu, dgdp, cur_time, terminated)
    return setup_reverse_callbacks(CallbackSet(cb), sensealg, dgdu, dgdp, cur_time, terminated)
end
function setup_reverse_callbacks(
        cb::CallbackSet, sensealg, dgdu, dgdp, cur_time,
        terminated
    )
    cb = CallbackSet(
        _setup_reverse_callbacks.(
            cb.continuous_callbacks,
            (sensealg,),
            (dgdu,),
            (dgdp,),
            (cur_time,), (terminated,)
        )...,
        reverse(
            _setup_reverse_callbacks.(
                cb.discrete_callbacks,
                (sensealg,),
                (dgdu,),
                (dgdp,), (cur_time,), (terminated,)
            )
        )...
    )
    return cb
end

function _setup_reverse_callbacks(
        cb::Union{
            ContinuousCallback, DiscreteCallback,
            VectorContinuousCallback,
        }, sensealg,
        dgdu,
        dgdp,
        loss_ref, terminated
    )
    return _setup_reverse_callbacks(cb, cb.affect!, sensealg, dgdu, dgdp, loss_ref, terminated)
end

function _setup_reverse_callbacks(
        cb::Union{
            ContinuousCallback, DiscreteCallback,
            VectorContinuousCallback,
        },
        affect::TrackedAffect, sensealg, dgdu,
        dgdp,
        loss_ref, terminated
    )
    if cb isa Union{ContinuousCallback, VectorContinuousCallback} && cb.affect! !== nothing
        cb.affect!.correction.cur_time = loss_ref # set cur_time
        cb.affect!.correction.terminated = terminated # flag if time evolution was terminated by callback
    end

    dgdp !== nothing &&
        error("We have not yet implemented custom adjoint rules to support parameter-dependent loss functions for hybrid systems.")

    # ReverseLossCallback adds gradients before and after the callback if save_positions is (true, true).
    # This, however, means that we must check the save_positions setting within the callback.
    # if save_positions = [1,1] is true the loss gradient is accumulated correctly before and after callback.
    # if save_positions = [0,0] no extra gradient is added.
    # if save_positions = [0,1] the gradient contribution is added before the callback but no additional gradient is added afterwards.
    # if save_positions = [1,0] the gradient contribution is added before, and in principle we would need to correct the adjoint state again. Therefore,

    cb.save_positions == [1, 0] && error("save_positions=[1,0] is currently not supported.")
    # Callbacks require ReverseDiffVJP or EnzymeVJP for their own VJP computations.
    # The ODE adjoint may use a different autojacvec (even numerical/false), but the
    # callback affect functions (CallbackAffectWrapper) are separate and typically work
    # with ReverseDiff even when the ODE function doesn't.
    cb_autojacvec = if sensealg.autojacvec isa Union{ReverseDiffVJP, EnzymeVJP}
        sensealg.autojacvec
    elseif sensealg.autojacvec isa ReactantVJP
        # ReactantVJP delegates to EnzymeVJP for callback affect VJPs.
        # Callbacks are called infrequently (only at event times) so there is
        # no benefit from Reactant compilation here.
        EnzymeVJP()
    else
        @warn "autojacvec=$(sensealg.autojacvec) is not compatible with callbacks, using ReverseDiffVJP() for callback VJPs"
        ReverseDiffVJP()
    end
    cb_sensealg = setvjp(sensealg, cb_autojacvec)

    # event times
    times = if cb isa DiscreteCallback
        cb.affect!.event_times
    else
        [cb.affect!.event_times; cb.affect_neg!.event_times]
    end

    # precompute w and wp to generate a tape
    cb_diffcaches = get_cb_diffcaches(cb, cb_autojacvec)

    function affect!(integrator)
        indx, pos_neg = get_indx(cb, integrator.t)
        tprev = get_tprev(cb, indx, pos_neg)
        event_idx = cb isa VectorContinuousCallback ? get_event_idx(cb, indx, pos_neg) :
            nothing

        # update diffcache here to use the correct precompiled callback tape
        w, wp = setup_w_wp(cb, cb_autojacvec, pos_neg, event_idx, tprev)
        diffcaches = cb_diffcaches[pos_neg, event_idx]

        S = integrator.f.f # get the sensitivity function

        # Create a fake sensitivity function to do the vjps
        fakeS = CallbackSensitivityFunction(w, cb_sensealg, diffcaches[1], integrator.sol.prob)

        du = first(get_tmp_cache(integrator))
        λ, grad, y, dλ, dgrad, dy = split_states(du, integrator.u, integrator.t, S)

        if sensealg isa GaussAdjoint
            dgrad = integrator.f.f.integrating_cb.affect!.accumulation_cache
            recursive_copyto!(dgrad, 0)
        end

        # if save_positions[2] = false, then the right limit is not saved. Thus, for
        # the QuadratureAdjoint we would need to lift y from the left to the right limit.
        # However, one also needs to update dgrad later on.
        if (sensealg isa QuadratureAdjoint && !cb.save_positions[2]) ||
                (sensealg isa InterpolatingAdjoint && ischeckpointing(sensealg))
            w(y, y, integrator.p, integrator.t)
        end

        if cb isa Union{ContinuousCallback, VectorContinuousCallback}
            # correction of the loss function sensitivity for continuous callbacks
            # wrt dependence of event time t on parameters and initial state.
            # Must be handled here because otherwise it is unclear if continuous or
            # discrete callback was triggered.
            (; correction) = cb.affect!
            (; dy_right, Lu_right) = correction
            # compute #f(xτ_right,p_right,τ(x₀,p))
            compute_f!(dy_right, S, y, integrator)
            # if callback did not terminate the time evolution, we have to compute one more correction term.
            if cb.save_positions[2] && !correction.terminated
                loss_indx = correction.cur_time[] + 1
                loss_correction!(Lu_right, y, integrator, dgdu, dgdp, loss_indx)
            else
                Lu_right .*= false
            end
        end

        update_p = copy_to_integrator!(cb, y, integrator.p, indx, pos_neg)

        if sensealg isa BacksolveAdjoint
            # reshape u and du (y and dy) to match forward pass (e.g., for matrices as initial conditions). Only needed for BacksolveAdjoint
            _y = S.y
            if eltype(y) <: ForwardDiff.Dual # handle implicit solvers
                copyto!(vec(_y), ForwardDiff.value.(y))
            else
                copyto!(vec(_y), y)
            end
            y = _y
        end

        if cb isa Union{ContinuousCallback, VectorContinuousCallback}
            # compute the correction of the right limit (with left state limit inserted into dgdt)
            (; dy_left, cur_time) = correction
            compute_f!(dy_left, S, y, integrator)
            dgdt(dy_left, correction, sensealg, y, integrator, tprev, event_idx)
            if !correction.terminated
                implicit_correction!(Lu_right, dλ, λ, dy_right, correction)
                correction.terminated = false # additional callbacks might have happened which didn't terminate the time evolution
            end
        end

        if update_p
            # changes in parameters
            if !(sensealg isa QuadratureAdjoint)
                # reinit diffcache struct
                #diffcache(t2)
                fakeSp = CallbackSensitivityFunctionPSwap(
                    wp, cb_sensealg, diffcaches[2],
                    integrator.sol.prob
                )
                #vjp with Jacobin given by dw/dp before event and vector given by grad

                if sensealg isa GaussAdjoint
                    vecjacobian!(
                        nothing, y,
                        integrator.f.f.integrating_cb.affect!.integrand_values.integrand,
                        integrator.p, integrator.t, fakeSp; dgrad = dgrad, dy = nothing
                    )
                    integrator.f.f.integrating_cb.affect!.integrand_values.integrand .= dgrad
                else
                    vecjacobian!(
                        nothing, y, grad, integrator.p, integrator.t, fakeSp;
                        dgrad = dgrad, dy = nothing
                    )
                    grad .= dgrad
                end
            end
        end

        vecjacobian!(dλ, y, λ, integrator.p, integrator.t, fakeS; dgrad, dy)

        # Since we differentiated the function that changes `p`, need to fix it again
        update_p = copy_to_integrator!(cb, y, integrator.p, indx, pos_neg)

        dgrad !== nothing && !(sensealg isa QuadratureAdjoint) && (dgrad .*= -1)

        if cb isa Union{ContinuousCallback, VectorContinuousCallback}
            # second correction to correct for left limit
            (; Lu_left) = correction
            implicit_correction!(Lu_left, dλ, dy_left, correction)
            dλ .+= Lu_left - Lu_right

            if cb.save_positions[1] == true
                # if the callback saved the first position, we need to implicitly correct this value as well
                loss_indx = correction.cur_time[]
                implicit_correction!(
                    Lu_left, dy_left, correction, y, integrator, dgdu,
                    dgdp,
                    loss_indx
                )
                dλ .+= Lu_left
            end
        end

        λ .= dλ

        return if sensealg isa GaussAdjoint
            @assert integrator.f.f isa ODEGaussAdjointSensitivityFunction
            integrator.f.f.integrating_cb.affect!.integrand_values.integrand .-= dgrad

            #recursive_add!(integrator.f.f.integrating_cb.affect!.integrand_values.integrand,dgrad)
        elseif !(sensealg isa QuadratureAdjoint)
            grad .-= dgrad
        end
    end

    return PresetTimeCallback(
        times,
        affect!,
        save_positions = (false, false)
    )
end

function _setup_reverse_callbacks(
        cb::Union{
            ContinuousCallback, DiscreteCallback,
            VectorContinuousCallback,
        },
        affect, sensealg,
        dgdu,
        dgdp,
        loss_ref, terminated
    )
    # return cb if affect is not a TrackedAffect
    return cb
end

mutable struct CallbackAffectWrapper{cbType, AJV, EI, T} <: Function
    cb::cbType
    autojacvec::AJV
    pos_neg::Bool
    event_idx::EI
    tprev::T
end

function (ff::CallbackAffectWrapper)(du, u, p, t)
    _affect! = get_affect!(ff.cb, ff.pos_neg)
    fakeinteg = get_FakeIntegrator(ff.autojacvec, u, p, t, ff.tprev)
    if ff.cb isa VectorContinuousCallback
        _affect!(fakeinteg, ff.event_idx)
    else
        _affect!(fakeinteg)
    end
    du .= fakeinteg.u
    return nothing
end

mutable struct CallbackAffectPWrapper{cbType, AJV, EI, T} <: Function
    cb::cbType
    autojacvec::AJV
    pos_neg::Bool
    event_idx::EI
    tprev::T
end

function (ff::CallbackAffectPWrapper)(dp, u, p, t)
    _affect! = get_affect!(ff.cb, ff.pos_neg)
    fakeinteg = get_FakeIntegrator(ff.autojacvec, u, p, t, ff.tprev)
    if ff.cb isa VectorContinuousCallback
        _affect!(fakeinteg, ff.event_idx)
    else
        _affect!(fakeinteg)
    end
    dp .= fakeinteg.p
    return nothing
end

function setup_w_wp(
        cb::Union{DiscreteCallback, ContinuousCallback, VectorContinuousCallback},
        autojacvec::Union{ReverseDiffVJP, EnzymeVJP}, pos_neg, event_idx,
        tprev
    )
    w = CallbackAffectWrapper(cb, autojacvec, pos_neg, event_idx, tprev)
    wp = CallbackAffectPWrapper(cb, autojacvec, pos_neg, event_idx, tprev)
    return w, wp
end

function get_FakeIntegrator(autojacvec::ReverseDiffVJP, u, p, t, tprev)
    return FakeIntegrator([x for x in u], [x for x in p], t, tprev)
end
get_FakeIntegrator(autojacvec::EnzymeVJP, u, p, t, tprev) = FakeIntegrator(u, p, t, tprev)

function _get_wp_paramjac_config(autojacvec::EnzymeVJP, _p, wp, y, __p, _t)
    return (zero(y), zero(_p), zero(_p), zero(_p), zero(y))
end

function _get_wp_paramjac_config(autojacvec::ReverseDiffVJP, _p, wp, y, __p, _t)
    if _p === nothing || _p isa SciMLBase.NullParameters
        tunables, repack = _p, identity
    else
        tunables, repack, aliases = canonicalize(Tunable(), _p)
    end
    tunables_inner = tunables
    tape = ReverseDiff.GradientTape((y, tunables_inner, [_t])) do u, p, t
        dp1 = similar(p, length(p))
        dp1 .= false
        wp(dp1, u, repack(p), first(t))
        return vec(dp1)
    end
    if compile_tape(autojacvec)
        return ReverseDiff.compile(tape)
    else
        return tape
    end
end

function get_cb_diffcaches(
        cb::Union{
            DiscreteCallback, ContinuousCallback,
            VectorContinuousCallback,
        },
        autojacvec
    )
    _dc = []
    if cb isa DiscreteCallback
        pos_negs = (true,)
    else
        pos_negs = (true, false)
    end
    if cb isa VectorContinuousCallback
        event_idxs = 1:(cb.len)
    else
        event_idxs = (nothing,)
    end
    for event_idx in event_idxs
        for pos_neg in pos_negs
            if (pos_neg && !isempty(cb.affect!.event_times)) ||
                    (!pos_neg && !isempty(cb.affect_neg!.event_times))
                if pos_neg && !isempty(cb.affect!.event_times)
                    y = cb.affect!.uleft[end]
                    _p = cb.affect!.pleft[end]
                    _t = cb.affect!.tprev[end]
                else
                    y = cb.affect_neg!.uleft[end]
                    _p = cb.affect_neg!.pleft[end]
                    _t = cb.affect_neg!.tprev[end]
                end

                w, wp = setup_w_wp(cb, autojacvec, pos_neg, event_idx, _t)

                paramjac_config = get_paramjac_config(
                    autojacvec, _p, w, y, _p, _t;
                    numindvar = length(y), alg = nothing,
                    isinplace = true, isRODE = false,
                    _W = nothing
                )
                pf = get_pf(autojacvec; _f = w, isinplace = true, isRODE = false)
                if autojacvec isa EnzymeVJP
                    paramjac_config = (paramjac_config..., Enzyme.make_zero(pf), nothing)
                end

                diffcache_w = AdjointDiffCache(
                    nothing, pf, nothing, nothing, nothing,
                    nothing, nothing, nothing, paramjac_config,
                    nothing, nothing, nothing, nothing, nothing,
                    nothing, nothing, nothing, false,
                    nothing, identity
                )

                paramjac_config = _get_wp_paramjac_config(
                    autojacvec, _p, wp, y, _p, _t
                )
                pf = get_pf(autojacvec; _f = wp, isinplace = true, isRODE = false)
                if autojacvec isa EnzymeVJP
                    paramjac_config = (paramjac_config..., Enzyme.make_zero(pf), nothing)
                end

                diffcache_wp = AdjointDiffCache(
                    nothing, pf, nothing, nothing, nothing,
                    nothing, nothing, nothing, paramjac_config,
                    nothing, nothing, nothing, nothing, nothing,
                    nothing, nothing, nothing, false,
                    nothing, identity
                )

                push!(_dc, (pos_neg, event_idx) => (diffcache_w, diffcache_wp))
            end
        end
    end
    return Dict(_dc)
end

get_indx(cb::DiscreteCallback, t) = (searchsortedfirst(cb.affect!.event_times, t), true)
function get_indx(cb::Union{ContinuousCallback, VectorContinuousCallback}, t)
    return if !isempty(cb.affect!.event_times) || !isempty(cb.affect_neg!.event_times)
        indx = searchsortedfirst(cb.affect!.event_times, t)
        indx_neg = searchsortedfirst(cb.affect_neg!.event_times, t)
        if !isempty(cb.affect!.event_times) &&
                cb.affect!.event_times[min(indx, length(cb.affect!.event_times))] == t
            return indx, true
        elseif !isempty(cb.affect_neg!.event_times) &&
                cb.affect_neg!.event_times[
                min(
                    indx_neg,
                    length(cb.affect_neg!.event_times)
                ),
            ] ==
                t
            return indx_neg, false
        else
            error("Event was triggered but no corresponding event in ContinuousCallback was found. Please report this error.")
        end
    else
        error("No event was recorded. Please report this error.")
    end
end

get_tprev(cb::DiscreteCallback, indx, bool) = cb.affect!.tprev[indx]
function get_tprev(cb::Union{ContinuousCallback, VectorContinuousCallback}, indx, bool)
    if bool
        return cb.affect!.tprev[indx]
    else
        return cb.affect_neg!.tprev[indx]
    end
end

function get_event_idx(cb::VectorContinuousCallback, indx, bool)
    if bool
        return cb.affect!.event_idx[indx]
    else
        return cb.affect_neg!.event_idx[indx]
    end
end

function copy_to_integrator!(cb::DiscreteCallback, y, p, indx, bool)
    # For BacksolveAdjoint, y is a view to the integrator state;
    # for the other methods, it's the S.y cache
    copyto!(y, cb.affect!.uleft[indx])
    update_p = (p != cb.affect!.pleft[indx])
    update_p && copyto!(p, cb.affect!.pleft[indx])
    return update_p
end

function copy_to_integrator!(
        cb::Union{ContinuousCallback, VectorContinuousCallback},
        y,
        p,
        indx,
        bool
    )
    if bool
        copyto!(y, cb.affect!.uleft[indx])
        update_p = (p != cb.affect!.pleft[indx])
        update_p && copyto!(p, cb.affect!.pleft[indx])
    else
        copyto!(y, cb.affect_neg!.uleft[indx])
        update_p = (p != cb.affect_neg!.pleft[indx])
        update_p && copyto!(p, cb.affect_neg!.pleft[indx])
    end
    return update_p
end

function compute_f!(dy, S, y, integrator)
    p, t = integrator.p, integrator.t

    if inplace_sensitivity(S)
        S.f(dy, y, p, t)
    else
        dy[:] .= S.f(y, p, t)
    end
    return nothing
end

function dgdt(dy, correction, sensealg, y, integrator, tprev, event_idx)
    # dy refers to f evaluated on left limit
    (; gt_val, gu_val, gt, gu, gt_conf, gu_conf, condition) = correction

    p, t = integrator.p, integrator.t

    fakeinteg = FakeIntegrator([x for x in y], p, t, tprev)

    # derivative and gradient of condition with respect to time and state, respectively
    gt.u = y
    gt.integrator = fakeinteg

    gu.t = t
    gu.integrator = fakeinteg

    # for VectorContinuousCallback we also need to set the event_idx.
    if gt isa VectorConditionTimeWrapper
        gt.event_idx = event_idx
        gu.event_idx = event_idx

        # safety check: evaluate condition to check if several conditions were true.
        # This is currently not supported
        condition(gt.out_cache, y, t, integrator)
        gt.out_cache .= abs.(gt.out_cache) .< 1000 * eps(eltype(gt.out_cache))
        (sum(gt.out_cache) != 1 || gt.out_cache[event_idx] != 1) &&
            error("Either several events were triggered or `event_idx` was falsely identified. Output of conditions $(gt.out_cache)")
    end

    derivative!(gt_val, gt, t, sensealg, gt_conf)
    gradient!(gu_val, gu, y, sensealg, gu_conf)

    gt_val .+= dot(gu_val, dy)
    @. gt_val = inv(gt_val) # allocates?

    @. gu_val *= -gt_val
    return nothing
end

function loss_correction!(Lu, y, integrator, dgdu, dgdp, indx)
    # ∂L∂t correction should be added if L depends explicitly on time.
    p, t = integrator.p, integrator.t
    dgdu(Lu, y, p, t, indx)
    return nothing
end

function implicit_correction!(Lu, dλ, λ, dy, correction)
    (; gu_val) = correction

    # remove gradients from adjoint state to compute correction factor
    @. dλ = λ - Lu
    Lu .= dot(dλ, dy) * gu_val

    return nothing
end

function implicit_correction!(Lu, λ, dy, correction)
    (; gu_val) = correction

    Lu .= dot(λ, dy) * gu_val

    return nothing
end

function implicit_correction!(Lu, dy, correction, y, integrator, dgdu, dgdp, indx)
    (; gu_val) = correction

    p, t = integrator.p, integrator.t

    # loss function gradient (not condition!)
    # ∂L∂t correction should be added, also ∂L∂p is missing.
    # correct adjoint
    dgdu(Lu, y, p, t, indx)

    Lu .= dot(Lu, dy) * gu_val

    # note that we don't add the gradient Lu here again to the correction because it will be added by the ReverseLossCallback.
    return nothing
end

# ConditionTimeWrapper: Wrapper for implicit correction for ContinuousCallback
# VectorConditionTimeWrapper: Wrapper for implicit correction for VectorContinuousCallback
function build_condition_wrappers(cb::ContinuousCallback, condition, u, t, fakeinteg)
    gt = ConditionTimeWrapper(condition, u, fakeinteg)
    gu = ConditionUWrapper(condition, t, fakeinteg)
    return gt, gu
end
function build_condition_wrappers(cb::VectorContinuousCallback, condition, u, t, fakeinteg)
    out = similar(u, cb.len) # create a cache for condition function (out,u,t,integrator)
    out .= 0
    gt = VectorConditionTimeWrapper(condition, u, fakeinteg, 1, out)
    gu = VectorConditionUWrapper(condition, t, fakeinteg, 1, out)
    return gt, gu
end
mutable struct ConditionTimeWrapper{F, uType, Integrator} <: Function
    f::F
    u::uType
    integrator::Integrator
end
(ff::ConditionTimeWrapper)(t) = [ff.f(ff.u, t, ff.integrator)]
mutable struct ConditionUWrapper{F, tType, Integrator} <: Function
    f::F
    t::tType
    integrator::Integrator
end
(ff::ConditionUWrapper)(u) = ff.f(u, ff.t, ff.integrator)
mutable struct VectorConditionTimeWrapper{F, uType, Integrator, outType} <: Function
    f::F
    u::uType
    integrator::Integrator
    event_idx::Int
    out_cache::outType
end
function (ff::VectorConditionTimeWrapper)(t)
    (
        out = zeros(typeof(t), length(ff.out_cache));
        ff.f(out, ff.u, t, ff.integrator);
        [
            out[ff.event_idx],
        ]
    )
end

mutable struct VectorConditionUWrapper{F, tType, Integrator, outType} <: Function
    f::F
    t::tType
    integrator::Integrator
    event_idx::Int
    out_cache::outType
end
function (ff::VectorConditionUWrapper)(u)
    (
        out = similar(u, length(ff.out_cache));
        ff.f(out, u, ff.t, ff.integrator); out[ff.event_idx]
    )
end

DiffEqBase.terminate!(i::FakeIntegrator) = nothing

# get the affect function of the callback. For example, allows us to get the `f` in PeriodicCallback without the integrator.tstops handling.
get_affect!(cb::DiscreteCallback, bool) = get_affect!(cb.affect!)
function get_affect!(cb::Union{ContinuousCallback, VectorContinuousCallback}, bool)
    return bool ? get_affect!(cb.affect!) : get_affect!(cb.affect_neg!)
end
get_affect!(affect!::TrackedAffect) = get_affect!(affect!.affect!)
get_affect!(affect!) = affect!
get_affect!(affect!::DiffEqCallbacks.PeriodicCallbackAffect) = affect!.affect!
