module SciMLSensitivityMooncakeExt

using SciMLSensitivity: SciMLSensitivity, FakeIntegrator
using Mooncake: Mooncake
import Mooncake: MinimalCtx, ForwardMode, Dual, CoDual, frule!!, rrule!!, @is_primitive,
    @zero_derivative
import SciMLSensitivity: get_paramjac_config, get_cb_paramjac_config, mooncake_run_ad,
    MooncakeVJP, MooncakeLoaded,
    DiffEqBase, MooncakeAdjoint, _init_originator_gradient,
    ReverseDiffAdjoint, TrackerAdjoint, ForwardSensitivity,
    EnzymeAdjoint, ZygoteAdjoint
using SciMLSensitivity: SciMLBase, SciMLStructures, canonicalize, Tunable, isscimlstructure,
    SciMLStructuresCompatibilityError, convert_tspan,
    has_continuous_callback,
    unwrapped_f, state_values, current_time
using SciMLSensitivity: FunctionWrappersWrappers, ODEFunction
using SciMLBase: remake, solve
using ChainRulesCore: NoTangent, ZeroTangent, Tangent, unthunk
using Accessors: @reset

# Mirrors `Mooncake.__call_rule` (src/utils.jl) instead of depending on it directly: that's
# a double-underscore internal helper, not part of Mooncake's rule-writing API. On Julia
# >= 1.11 it's just `rule(args...)`; on 1.10 it routes through `Base.inferencebarrier` and
# a `@noinline` erased call to dodge a real compiler invalidation bug (see Mooncake's own
# comment for the MWE). SciMLSensitivity still supports 1.10, so this keeps the workaround
# instead of silently dropping it.
@static if VERSION < v"1.11-"
    @noinline _call_built_rule_erased(rule, args) = rule(args...)
    @inline _call_built_rule(rule, args) = _call_built_rule_erased(Base.inferencebarrier(rule), args)
else
    @inline _call_built_rule(rule, args) = rule(args...)
end

# `automatic_sensealg_choice` is a discrete algorithm choice, never a differentiable
# function of u0/p, but its body has a try/catch Mooncake's forward-mode compiler can't
# trace. Zero-derivative keeps Mooncake from ever looking inside.
@zero_derivative(
    MinimalCtx, Tuple{typeof(SciMLSensitivity.automatic_sensealg_choice), Vararg},
)

# Mooncake-native gradient for the DAE/ODE init path. Avoids pulling Zygote
# into the rrule when the user is differentiating with Mooncake. The default
# (Zygote-based) implementation lives in src/concrete_solve.jl.
function _init_originator_gradient(
        ::SciMLBase.MooncakeOriginator, f, tunables,
    )
    rule = Mooncake.build_rrule(f, tunables)
    _, (_, igs) = Mooncake.value_and_gradient!!(rule, f, tunables)
    return igs
end

function get_paramjac_config(::MooncakeLoaded, ::MooncakeVJP, pf, p, f, y, _t)
    dy_mem = zero(y)
    λ_mem = zero(y)
    cache = Mooncake.prepare_pullback_cache(pf, dy_mem, y, p, _t)
    # Pre-allocate buffer for tangent_to_primal!! conversion of struct-based
    # array types (e.g. ComponentArray) whose Mooncake tangent is Mooncake.Tangent.
    # (Mooncake.Config(friendly_tangents=true) would avoid this, but currently
    # fails on complex closure types captured by pf.)
    p_grad_buf = p isa AbstractArray && !(p isa Array) ? similar(p) : nothing
    return cache, pf, λ_mem, dy_mem, p_grad_buf
end

"""
    get_cb_paramjac_config(::MooncakeLoaded, ::MooncakeVJP, raw_affect, event_idx, y, p, _t, mode)

Build a Mooncake pullback cache for a tracked callback affect function. Mirrors
the `get_cb_paramjac_config(::ReactantLoaded, ::ReactantVJP, ...)` entry point:
`raw_affect` is extracted upfront (`get_affect!(cb, pos_neg)` at the call site)
so the Mooncake-traced closure does not need to recursively unwrap
`TrackedAffect`, which would otherwise trip on the `Base.argument_datatype`
ccall surfaced by that dispatch.

`mode === :state` builds a cache for the state-affect closure (state-sized
output); `mode === :param` builds one for the parameter-affect closure
(parameter-sized output) so its Mooncake cotangent/output buffers match the
flat tunables shape rather than the state shape. The returned 5-tuple has the
same layout as `get_paramjac_config(::MooncakeLoaded, ::MooncakeVJP, ...)` so
`_vecjacobian!(::MooncakeVJP)` / `mooncake_run_ad` can consume it unchanged.
"""
function get_cb_paramjac_config(
        ::MooncakeLoaded, ::MooncakeVJP, raw_affect, event_idx, y, p, _t, mode
    )
    has_event_idx = event_idx !== nothing
    tprev0 = _t

    if mode === :state
        pf = let raw = raw_affect, ev = event_idx, tprev = tprev0, has_ev = has_event_idx
            (out, u, p, t) -> begin
                fakeinteg = FakeIntegrator(copy(u), copy(p), t, tprev)
                if has_ev
                    raw(fakeinteg, ev)
                else
                    raw(fakeinteg)
                end
                copyto!(out, fakeinteg.u)
                return out
            end
        end
        out_sample = y
    elseif mode === :param
        pf = let raw = raw_affect, ev = event_idx, tprev = tprev0, has_ev = has_event_idx
            (out, u, p, t) -> begin
                fakeinteg = FakeIntegrator(copy(u), copy(p), t, tprev)
                if has_ev
                    raw(fakeinteg, ev)
                else
                    raw(fakeinteg)
                end
                copyto!(out, fakeinteg.p)
                return out
            end
        end
        out_sample = p
    else
        error("get_cb_paramjac_config: unknown mode $(mode); expected :state or :param")
    end

    dy_mem = zero(out_sample)
    λ_mem = zero(out_sample)
    cache = Mooncake.prepare_pullback_cache(pf, dy_mem, y, p, _t)
    p_grad_buf = p isa AbstractArray && !(p isa Array) ? similar(p) : nothing
    return cache, pf, λ_mem, dy_mem, p_grad_buf
end

function mooncake_run_ad(paramjac_config::Tuple, y, p, t, λ)
    cache, pf, λ_mem, dy_mem, p_grad_buf = paramjac_config
    λ_mem .= λ
    # The Mooncake cache is built with flat tunables (Vector), but callers like
    # _vecjacobian! and vec_pjac! may pass the full structured parameter object.
    # Extract flat tunables when p is a structured type to match the cache.
    _p = if !(p isa AbstractArray) && p !== nothing && !(p isa SciMLBase.NullParameters)
        first(canonicalize(Tunable(), p))
    else
        p
    end
    dy, _ = Mooncake.value_and_pullback!!(cache, λ_mem, pf, dy_mem, y, _p, t)
    y_grad = cache.tangents[3]
    p_grad_raw = cache.tangents[4]
    p_grad = if p_grad_buf !== nothing && p_grad_raw isa Mooncake.Tangent
        Mooncake.tangent_to_primal!!(p_grad_buf, p_grad_raw)
    else
        p_grad_raw
    end
    return dy, y_grad, p_grad
end

function SciMLBase._concrete_solve_adjoint(
        prob::Union{
            SciMLBase.AbstractDiscreteProblem,
            SciMLBase.AbstractODEProblem,
            SciMLBase.AbstractDAEProblem,
            SciMLBase.AbstractDDEProblem,
            SciMLBase.AbstractSDEProblem,
            SciMLBase.AbstractSDDEProblem,
            SciMLBase.AbstractRODEProblem,
        },
        alg, sensealg::MooncakeAdjoint,
        u0, p, originator::SciMLBase.ADOriginator,
        args...;
        kwargs...
    )
    if !(p === nothing || p isa SciMLBase.NullParameters)
        if !isscimlstructure(p)
            throw(SciMLStructuresCompatibilityError())
        end
    end

    if p === nothing || p isa SciMLBase.NullParameters
        tunables, repack = p, identity
    else
        tunables, repack, _ = canonicalize(Tunable(), p)
    end

    function mooncake_adjoint_forwardpass(_u0, _p)
        if (
                convert_tspan(sensealg) === nothing &&
                    ((haskey(kwargs, :callback) && has_continuous_callback(kwargs[:callback])))
            ) ||
                (convert_tspan(sensealg) !== nothing && convert_tspan(sensealg))
            _tspan = convert.(eltype(_p), prob.tspan)
        else
            _tspan = prob.tspan
        end

        if DiffEqBase.isinplace(prob)
            if prob.f isa ODEFunction &&
                    (
                    prob.f.f isa FunctionWrappersWrappers.FunctionWrappersWrapper ||
                        SciMLBase.specialization(prob.f) === SciMLBase.AutoSpecialize
                )
                f = ODEFunction{DiffEqBase.isinplace(prob), SciMLBase.FullSpecialize}(unwrapped_f(prob.f))
                _prob = remake(
                    prob, f = f, u0 = _u0, p = _p, tspan = _tspan, callback = nothing
                )
            else
                _prob = remake(prob, u0 = _u0, p = _p, tspan = _tspan, callback = nothing)
            end
        else
            _prob = remake(
                prob, u0 = _u0, p = SciMLStructures.replace(Tunable(), p, _p),
                tspan = _tspan, callback = nothing
            )
        end

        kwargs_filtered = NamedTuple(filter(x -> x[1] != :sensealg, kwargs))
        sol = solve(
            _prob, alg, args...; sensealg = DiffEqBase.SensitivityADPassThrough(),
            kwargs_filtered...
        )
        sol = SciMLBase.sensitivity_solution(sol, state_values(sol), current_time(sol))
        @reset sol.prob = prob
        return sol
    end

    # `_concrete_solve_adjoint` must return `(primal, pullback)`, with the seed for `pullback`
    # not known until later -- but Mooncake's public `value_and_pullback!!` only offers an
    # eager, single-call form (build + run forward + apply seed, all at once). So we split it
    # by hand: build the rule and run the forward pass now, and defer applying the seed until
    # `mooncake_adjoint_backpass` is actually invoked.
    rule = Mooncake.build_rrule(mooncake_adjoint_forwardpass, u0, tunables)
    fx = (
        CoDual(mooncake_adjoint_forwardpass, Mooncake.zero_tangent(mooncake_adjoint_forwardpass)),
        CoDual(u0, Mooncake.zero_tangent(u0)),
        CoDual(tunables, Mooncake.zero_tangent(tunables)),
    )
    fx_fwds = Mooncake.tuple_map(Mooncake.to_fwds, fx)
    out, pb!! = _call_built_rule(rule, fx_fwds)

    function mooncake_adjoint_backpass(ybar)
        # Convert the incoming ChainRules-style cotangent into the Mooncake tangent type of
        # the primal output (the same conversion `@from_rrule`/`@from_chainrules` use), run
        # the deferred Mooncake pullback, then convert the resulting Mooncake tangents back
        # to ChainRules tangents for the caller.
        ȳ = Mooncake.mooncake_tangent(Mooncake.primal(out), ybar)
        dfargs = pb!!(Mooncake.rdata(ȳ))
        _, u0bar_mc, pbar_mc = Mooncake.tuple_map(
            (f, r) -> Mooncake.tangent(Mooncake.fdata(Mooncake.tangent(f)), r), fx, dfargs
        )
        _u0bar = Mooncake.to_cr_tangent(u0bar_mc)
        pbar = Mooncake.to_cr_tangent(pbar_mc)

        return if originator isa SciMLBase.TrackerOriginator ||
                originator isa SciMLBase.ReverseDiffOriginator
            (
                NoTangent(), NoTangent(), _u0bar, pbar, NoTangent(),
                ntuple(_ -> NoTangent(), length(args))...,
            )
        else
            (
                NoTangent(), NoTangent(), NoTangent(),
                _u0bar, pbar, NoTangent(),
                ntuple(_ -> NoTangent(), length(args))...,
            )
        end
    end

    out_primal = Mooncake.primal(out)
    u = state_values(out_primal)
    return SciMLBase.sensitivity_solution(out_primal, u, current_time(out_primal)),
        mooncake_adjoint_backpass
end

# ============================================================================
# Mooncake-native `solve_up` rule for `MooncakeAdjoint`
#
# The generic `@from_rrule`-based `solve_up` primitive (`DiffEqBaseMooncakeExt.jl`) round-trips
# through `mooncake_tangent`/`to_cr_tangent`, which has no case for a struct as deeply nested
# as `ODESolution` and throws `ArgumentError: ... does not currently have a method of
# mooncake_tangent`. `MooncakeAdjoint`'s gradient never needs to leave Mooncake's native
# representation, so this rule bypasses ChainRules entirely: a *nested* `Mooncake.build_rrule`
# call (not a foreign-AD-backed pullback like `_MooncakeOverAnotherADSensealg` below), reusing
# the incoming `u0`/`p` CoDuals' fdata directly so in-place accumulation lands in the same,
# potentially-aliased buffers the surrounding reverse pass expects (see the "Aliasing
# Invariant" in docs/src/understanding_mooncake/rule_system.md).
function _solve_up_mooncake_native_forwardpass(prob, alg_and_rest, kwargs, _u0, _p)
    _prob = remake(prob; u0 = _u0, p = _p)
    sol = solve(
        _prob, alg_and_rest...; sensealg = DiffEqBase.SensitivityADPassThrough(), kwargs...
    )
    return SciMLBase.sensitivity_solution(sol, state_values(sol), current_time(sol))
end

function _solve_up_mooncake_native(
        prob, sensealg, u0::CoDual, p::CoDual, alg_and_rest...; kwargs...
    )
    forwardpass(_u0, _p) = _solve_up_mooncake_native_forwardpass(
        prob, alg_and_rest, kwargs, _u0, _p
    )
    rule = Mooncake.build_rrule(forwardpass, Mooncake.primal(u0), Mooncake.primal(p))
    fx_fwds = (
        CoDual(forwardpass, Mooncake.fdata(Mooncake.zero_tangent(forwardpass))),
        u0, p,
    )
    out, pb!! = _call_built_rule(rule, fx_fwds)
    function native_pb!!(y_rdata)
        _, u0_rdata, p_rdata = pb!!(y_rdata)
        return u0_rdata, p_rdata
    end
    return Mooncake.primal(out), Mooncake.tangent(out), native_pb!!
end

function rrule!!(
        f::CoDual{typeof(DiffEqBase.solve_up)},
        prob::CoDual{<:DiffEqBase.AbstractDEProblem},
        sensealg::CoDual{<:MooncakeAdjoint},
        u0::CoDual, p::CoDual, alg_and_rest::CoDual...,
    )
    fargs = (f, prob, sensealg, u0, p, alg_and_rest...)
    primals = Mooncake.tuple_map(Mooncake.primal, fargs)
    lazy_rdata = Mooncake.tuple_map(Mooncake.lazy_zero_rdata, primals)
    y_primal, y_fdata, native_pb!! = _solve_up_mooncake_native(
        primals[2], primals[3], u0, p, primals[6:end]...
    )

    function pb!!(y_rdata)
        u0_rdata, p_rdata = native_pb!!(y_rdata)
        return (
            Mooncake.instantiate(lazy_rdata[1]),
            Mooncake.instantiate(lazy_rdata[2]),
            Mooncake.instantiate(lazy_rdata[3]),
            u0_rdata,
            p_rdata,
            Mooncake.tuple_map(Mooncake.instantiate, lazy_rdata[6:end])...,
        )
    end

    return CoDual(y_primal, y_fdata), pb!!
end

function rrule!!(
        ::CoDual{typeof(Core.kwcall)},
        kwargs::CoDual{<:NamedTuple},
        f::CoDual{typeof(DiffEqBase.solve_up)},
        prob::CoDual{<:DiffEqBase.AbstractDEProblem},
        sensealg::CoDual{<:MooncakeAdjoint},
        u0::CoDual, p::CoDual, alg_and_rest::CoDual...,
    )
    fargs = (f, prob, sensealg, u0, p, alg_and_rest...)
    primals = Mooncake.tuple_map(Mooncake.primal, fargs)
    lazy_rdata = Mooncake.tuple_map(Mooncake.lazy_zero_rdata, primals)
    kwargs_p = Base.structdiff(Mooncake.primal(kwargs), NamedTuple{(:originator,)})
    y_primal, y_fdata, native_pb!! = _solve_up_mooncake_native(
        primals[2], primals[3], u0, p, primals[6:end]...; kwargs_p...
    )
    kwargs_rdata = Mooncake.rdata(Mooncake.zero_tangent(Mooncake.primal(kwargs)))

    function pb!!(y_rdata)
        u0_rdata, p_rdata = native_pb!!(y_rdata)
        args_rdata = (
            Mooncake.instantiate(lazy_rdata[1]),
            Mooncake.instantiate(lazy_rdata[2]),
            Mooncake.instantiate(lazy_rdata[3]),
            u0_rdata,
            p_rdata,
            Mooncake.tuple_map(Mooncake.instantiate, lazy_rdata[6:end])...,
        )
        return Mooncake.NoRData(), kwargs_rdata, args_rdata...
    end

    return CoDual(y_primal, y_fdata), pb!!
end

# Mooncake stacked over ReverseDiffAdjoint/TrackerAdjoint/ForwardSensitivity
# (SciML/SciMLSensitivity.jl#1510, chalk-lab/Mooncake.jl#1208).
#
# These sensealgs need to know which AD is driving them (`originator`), normally
# auto-detected via a `@mooncake_overlay`. That detection never fires under Mooncake:
# `MooncakeOriginator`/`ChainRulesOriginator` are zero-field structs, which Julia constant-
# folds away before any rule dispatch happens. So these rules skip detection and construct
# `MooncakeOriginator()` directly, dispatching into the existing `MooncakeOriginator`
# methods (#1420, src/concrete_solve.jl) that re-solve with a plain primal Mooncake can wrap
# in a `CoDual`. More specific than the generic rule's `Union{Nothing,
# AbstractSensitivityAlgorithm}` signature, so dispatch prefers these three sensealgs and
# falls back to the generic rule for everything else.
const _MooncakeOverAnotherADSensealg = Union{
    ReverseDiffAdjoint, TrackerAdjoint, ForwardSensitivity,
}

function _solve_up_mooncake_over_another_ad(prob, sensealg, u0, p, alg_and_kwargs...; kwargs...)
    return DiffEqBase._solve_adjoint(
        prob, sensealg, u0, p, SciMLBase.MooncakeOriginator(), alg_and_kwargs...; kwargs...
    )
end

# `cr_dfargs` is shaped like `_concrete_solve_adjoint`'s arg list
# (`prob̄, alḡ, sensealḡ, ū0, p̄, originator̄, tail̄...`), not `solve_up`'s. Reorders/pads it to
# `fargs = (f, prob, sensealg, u0, p, alg_and_rest...)`, dropping `originator` (a zero-field
# marker, never differentiable). `alg_and_rest` can be empty even though `_concrete_solve_adjoint`
# always has an `alg` slot -- `_solve_adjoint` takes the same no-tail branch either way.
function _match_fargs_cotangents(cr_dfargs, alg_and_rest)
    alg_and_rest_cotangents = isempty(alg_and_rest) ? () : (cr_dfargs[2], cr_dfargs[7:end]...)
    return (
        NoTangent(), cr_dfargs[1], cr_dfargs[3], cr_dfargs[4], cr_dfargs[5],
        alg_and_rest_cotangents...,
    )
end

function rrule!!(
        f::CoDual{typeof(DiffEqBase.solve_up)},
        prob::CoDual{<:SciMLBase.AbstractDEProblem},
        sensealg::CoDual{<:_MooncakeOverAnotherADSensealg},
        u0::CoDual, p::CoDual, alg_and_rest::CoDual...,
    )
    fargs = (f, prob, sensealg, u0, p, alg_and_rest...)
    primals = Mooncake.tuple_map(Mooncake.primal, fargs)
    lazy_rdata = Mooncake.tuple_map(Mooncake.lazy_zero_rdata, primals)
    y_primal, cr_pb = _solve_up_mooncake_over_another_ad(primals[2:end]...)
    y_fdata = Mooncake.fdata(Mooncake.zero_tangent(y_primal))

    function pb!!(y_rdata)
        cr_tangent = Mooncake.to_cr_tangent(Mooncake.tangent(y_fdata, y_rdata))
        cr_dfargs = _match_fargs_cotangents(cr_pb(cr_tangent), alg_and_rest)
        return Mooncake.tuple_map(fargs, lazy_rdata, cr_dfargs) do x, l_rdata, cr_dx
            return Mooncake.increment_and_get_rdata!(
                Mooncake.tangent(x), Mooncake.instantiate(l_rdata), cr_dx
            )
        end
    end

    return CoDual(y_primal, y_fdata), pb!!
end

function rrule!!(
        ::CoDual{typeof(Core.kwcall)},
        kwargs::CoDual{<:NamedTuple},
        f::CoDual{typeof(DiffEqBase.solve_up)},
        prob::CoDual{<:SciMLBase.AbstractDEProblem},
        sensealg::CoDual{<:_MooncakeOverAnotherADSensealg},
        u0::CoDual, p::CoDual, alg_and_rest::CoDual...,
    )
    fargs = (f, prob, sensealg, u0, p, alg_and_rest...)
    primals = Mooncake.tuple_map(Mooncake.primal, fargs)
    lazy_rdata = Mooncake.tuple_map(Mooncake.lazy_zero_rdata, primals)
    # `originator` is dropped from the forwarded kwargs -- we supply the
    # correct one ourselves above, unconditionally, since this rule only ever
    # fires under Mooncake.
    kwargs_p = Base.structdiff(Mooncake.primal(kwargs), NamedTuple{(:originator,)})
    y_primal, cr_pb = _solve_up_mooncake_over_another_ad(primals[2:end]...; kwargs_p...)
    y_fdata = Mooncake.fdata(Mooncake.zero_tangent(y_primal))
    kwargs_rdata = Mooncake.rdata(Mooncake.zero_tangent(Mooncake.primal(kwargs)))

    function pb!!(y_rdata)
        cr_tangent = Mooncake.to_cr_tangent(Mooncake.tangent(y_fdata, y_rdata))
        cr_dfargs = _match_fargs_cotangents(cr_pb(cr_tangent), alg_and_rest)
        args_rdata = Mooncake.tuple_map(fargs, lazy_rdata, cr_dfargs) do x, l_rdata, cr_dx
            return Mooncake.increment_and_get_rdata!(
                Mooncake.tangent(x), Mooncake.instantiate(l_rdata), cr_dx
            )
        end
        return Mooncake.NoRData(), kwargs_rdata, args_rdata...
    end

    return CoDual(y_primal, y_fdata), pb!!
end

# `EnzymeAdjoint`/`ZygoteAdjoint` under Mooncake: confirmed broken by direct testing, for
# two different reasons. `ZygoteAdjoint` fails inside Zygote's own tracing of solve()'s
# internals (a mutating-array operation Zygote can't differentiate, unrelated to Mooncake).
# `EnzymeAdjoint` fails on a type mismatch inside Mooncake's own rule machinery. Neither
# currently reaches this extension at all -- both fall through to the generic
# `@from_rrule`-based solve_up primitive and crash with a confusing, deep stacktrace.
# Erroring here instead, before any of that runs, gives a clear message up front.
const _MooncakeUnsupportedAnotherADSensealg = Union{EnzymeAdjoint, ZygoteAdjoint}

function _mooncake_another_ad_backend_unsupported(sensealg)
    error(
        "solve() with $(typeof(sensealg)) is not currently supported under Mooncake " *
            "(confirmed broken by direct testing -- not just untested). Use " *
            "MooncakeAdjoint() instead."
    )
end

function rrule!!(
        ::CoDual{typeof(DiffEqBase.solve_up)},
        ::CoDual{<:DiffEqBase.AbstractDEProblem},
        sensealg::CoDual{<:_MooncakeUnsupportedAnotherADSensealg},
        ::CoDual, ::CoDual, ::CoDual...,
    )
    return _mooncake_another_ad_backend_unsupported(Mooncake.primal(sensealg))
end

function rrule!!(
        ::CoDual{typeof(Core.kwcall)},
        ::CoDual{<:NamedTuple},
        ::CoDual{typeof(DiffEqBase.solve_up)},
        ::CoDual{<:DiffEqBase.AbstractDEProblem},
        sensealg::CoDual{<:_MooncakeUnsupportedAnotherADSensealg},
        ::CoDual, ::CoDual, ::CoDual...,
    )
    return _mooncake_another_ad_backend_unsupported(Mooncake.primal(sensealg))
end

# Forward-mode primitive for `DiffEqBase._solve_adjoint` under ReverseDiffAdjoint and
# TrackerAdjoint (SciML/SciMLSensitivity.jl#1427) -- just an explicit error, see the
# message below for why. Registered on `_solve_adjoint` rather than `solve_up`, since the
# rrule above calls `_solve_adjoint` directly and `solve_up` is never reached on this
# path. Deliberately narrower than `_MooncakeOverAnotherADSensealg` (excludes
# ForwardSensitivity, which gets its own error below for a different reason).
const _MooncakeDelegatesToAnotherAD = Union{ReverseDiffAdjoint, TrackerAdjoint}

function _mooncake_solve_adjoint_forward_mode_unsupported(sensealg)
    error(
        "value_and_hvp!! can't compute a Hessian-vector product for solve() with " *
            "$(typeof(sensealg)) under Mooncake. $(typeof(sensealg)) gets its gradient by " *
            "calling ReverseDiff.jl or Tracker.jl, and Mooncake can't safely forward-mode " *
            "differentiate through another AD package's internals " *
            "(SciML/SciMLSensitivity.jl#1427). Use MooncakeAdjoint() if you need an HVP " *
            "today. A real fix means writing a Mooncake-native adjoint rule for " *
            "$(typeof(sensealg)) (a continuous adjoint method using only " *
            "Mooncake.value_and_gradient!!/value_and_hvp!! on the ODE right-hand side), so " *
            "Mooncake's forward-over-reverse machinery can differentiate it the way it " *
            "already does for MooncakeAdjoint."
    )
end

@is_primitive(
    MinimalCtx, ForwardMode,
    Tuple{
        typeof(DiffEqBase._solve_adjoint), SciMLBase.AbstractDEProblem,
        _MooncakeDelegatesToAnotherAD, Any, Any, SciMLBase.MooncakeOriginator, Vararg,
    },
)
function frule!!(
        ::Dual{typeof(DiffEqBase._solve_adjoint)},
        ::Dual{<:SciMLBase.AbstractDEProblem},
        sensealg::Dual{<:_MooncakeDelegatesToAnotherAD},
        ::Dual, ::Dual,
        ::Dual{<:SciMLBase.MooncakeOriginator},
        ::Dual...,
    )
    return _mooncake_solve_adjoint_forward_mode_unsupported(Mooncake.primal(sensealg))
end

@is_primitive(
    MinimalCtx, ForwardMode,
    Tuple{
        typeof(Core.kwcall), NamedTuple, typeof(DiffEqBase._solve_adjoint),
        SciMLBase.AbstractDEProblem, _MooncakeDelegatesToAnotherAD, Any, Any,
        SciMLBase.MooncakeOriginator, Vararg,
    },
)
function frule!!(
        ::Dual{typeof(Core.kwcall)},
        ::Dual{<:NamedTuple},
        ::Dual{typeof(DiffEqBase._solve_adjoint)},
        ::Dual{<:SciMLBase.AbstractDEProblem},
        sensealg::Dual{<:_MooncakeDelegatesToAnotherAD},
        ::Dual, ::Dual,
        ::Dual{<:SciMLBase.MooncakeOriginator},
        ::Dual...,
    )
    return _mooncake_solve_adjoint_forward_mode_unsupported(Mooncake.primal(sensealg))
end

# ForwardSensitivity's HVP is also unsupported, but not for the ReverseDiff/Tracker
# reason above -- it doesn't delegate to another AD package at all. Plain Mooncake
# tracing here hits a missing rule for FunctionWrapper's .obj access first, and past
# that, a deeper wall: tracing into ForwardSensitivity's own ForwardDiff.jl Jacobian
# machinery inside solve_call hits the same type-prediction mismatch as
# chalk-lab/Mooncake.jl#1209.
function _mooncake_forward_sensitivity_hvp_unsupported()
    error(
        "value_and_hvp!! can't compute a Hessian-vector product for solve() with " *
            "ForwardSensitivity() under Mooncake. ForwardSensitivity computes its " *
            "gradient using ForwardDiff.jl internally, and forward-mode-differentiating " *
            "that with Mooncake hits a type-prediction mismatch deep in solve_call " *
            "(the same class of issue as chalk-lab/Mooncake.jl#1209). Use MooncakeAdjoint() " *
            "if you need an HVP today. See SciML/SciMLSensitivity.jl#1427 for what was tried."
    )
end

@is_primitive(
    MinimalCtx, ForwardMode,
    Tuple{
        typeof(DiffEqBase._solve_adjoint), SciMLBase.AbstractDEProblem,
        ForwardSensitivity, Any, Any, SciMLBase.MooncakeOriginator, Vararg,
    },
)
function frule!!(
        ::Dual{typeof(DiffEqBase._solve_adjoint)},
        ::Dual{<:SciMLBase.AbstractDEProblem},
        ::Dual{<:ForwardSensitivity},
        ::Dual, ::Dual,
        ::Dual{<:SciMLBase.MooncakeOriginator},
        ::Dual...,
    )
    return _mooncake_forward_sensitivity_hvp_unsupported()
end

@is_primitive(
    MinimalCtx, ForwardMode,
    Tuple{
        typeof(Core.kwcall), NamedTuple, typeof(DiffEqBase._solve_adjoint),
        SciMLBase.AbstractDEProblem, ForwardSensitivity, Any, Any,
        SciMLBase.MooncakeOriginator, Vararg,
    },
)
function frule!!(
        ::Dual{typeof(Core.kwcall)},
        ::Dual{<:NamedTuple},
        ::Dual{typeof(DiffEqBase._solve_adjoint)},
        ::Dual{<:SciMLBase.AbstractDEProblem},
        ::Dual{<:ForwardSensitivity},
        ::Dual, ::Dual,
        ::Dual{<:SciMLBase.MooncakeOriginator},
        ::Dual...,
    )
    return _mooncake_forward_sensitivity_hvp_unsupported()
end

end
