mutable struct GaussIntegrand{pType, uType, lType, rateType, S, PF, PJC, PJT, DGP,
    G, SAlg <: AbstractGAdjoint, HType}
    sol::S
    p::pType
    y::uType
    λ::lType
    pf::PF
    f_cache::rateType
    pJ::PJT
    paramjac_config::PJC
    sensealg::SAlg
    dgdp_cache::DGP
    dgdp::G
    dde_history::HType  # History wrapper for DDEs, nothing for ODEs
end

struct ODEGaussAdjointSensitivityFunction{C <: AdjointDiffCache,
    Alg <: AbstractGAdjoint,
    uType, SType, CPS, pType,
    fType,
    GI <: GaussIntegrand,
    ICB} <: SensitivityFunction
    diffcache::C
    sensealg::Alg
    discrete::Bool
    y::uType
    sol::SType
    checkpoint_sol::CPS
    prob::pType
    f::fType
    GaussInt::GI
    integrating_cb::ICB
end

mutable struct GaussCheckpointSolution{S, I, T, T2}
    cpsol::S # solution in a checkpoint interval
    intervals::I # checkpoint intervals
    cursor::Int # sol.prob.tspan = intervals[cursor]
    tols::T
    tstops::T2 # for callbacks
end

function ODEGaussAdjointSensitivityFunction(
        g, sensealg, gaussint, discrete, sol, dgdu, dgdp,
        f, alg,
        checkpoints, integrating_cb, tols, tstops = nothing;
        tspan = reverse(sol.prob.tspan))
    checkpointing = ischeckpointing(sensealg, sol)
    (checkpointing && checkpoints === nothing) &&
        error("checkpoints must be passed when checkpointing is enabled.")
    checkpoint_sol = if checkpointing
        intervals = map(tuple, @view(checkpoints[1:(end - 1)]), @view(checkpoints[2:end]))
        interval_end = intervals[end][end]
        tspan[1] > interval_end && push!(intervals, (interval_end, tspan[1]))
        cursor = lastindex(intervals)
        interval = intervals[cursor]
        if tstops === nothing
            cpsol = solve(remake(sol.prob, tspan = interval, u0 = sol(interval[1])),
                sol.alg; dense = true, tols...)
            gaussint.sol = cpsol
        else
            if maximum(interval[1] .< tstops .< interval[2])
                # callback might have changed p
                _p = Gaussreset_p(sol.prob.kwargs[:callback], interval)
                #cpsol = solve(remake(sol.prob, tspan = interval, u0 = sol(interval[1])),
                #    tstops = tstops,
                #    p = _p, sol.alg; tols...)

                cpsol = solve(remake(sol.prob, tspan = interval, u0 = sol(interval[1])),
                    dense = true,
                    p = _p, sol.alg; tols...)
                gaussint.sol = cpsol
            else
                #cpsol = solve(remake(sol.prob, tspan = interval, u0 = sol(interval[1])),
                #    tstops = tstops, sol.alg; tols...)
                cpsol = solve(remake(sol.prob, tspan = interval, u0 = sol(interval[1])),
                    sol.alg; dense = true, tols...)
                gaussint.sol = cpsol
            end
        end
        GaussCheckpointSolution(cpsol, intervals, cursor, tols, tstops)
    else
        nothing
    end
    diffcache,
    y = adjointdiffcache(
        g, sensealg, discrete, sol, dgdu, dgdp, f, alg;
        quad = true)
    return ODEGaussAdjointSensitivityFunction(diffcache, sensealg, discrete,
        y, sol, checkpoint_sol, sol.prob, f, gaussint, integrating_cb)
end

function Gaussfindcursor(intervals, t)
    # equivalent with `findfirst(x->x[1] <= t <= x[2], intervals)`
    lt(x, t) = <(x[2], t)
    return searchsortedfirst(intervals, t, lt = lt)
end

# u = λ'
function (S::ODEGaussAdjointSensitivityFunction)(du, u, p, t)
    (; sol, checkpoint_sol, discrete, prob, f) = S
    #f = sol.prob.f
    λ, grad, y, dλ, dgrad, dy = split_states(du, u, t, S)

    vecjacobian!(dλ, y, λ, p, t, S)

    dλ .*= -one(eltype(λ))
    discrete || accumulate_cost!(dλ, y, p, t, S)
    return nothing
end

function (S::ODEGaussAdjointSensitivityFunction)(du, u, p, t, W)
    (; sol, checkpoint_sol, discrete, prob, f) = S

    λ, grad, y, dλ, dgrad, dy = split_states(du, u, t, S)

    vecjacobian!(dλ, y, λ, p, t, S, W = W)

    dλ .*= -one(eltype(λ))

    discrete || accumulate_cost!(dλ, y, p, t, S)
    return nothing
end

function (S::ODEGaussAdjointSensitivityFunction)(u, p, t)
    (; sol, checkpoint_sol, discrete, prob) = S
    f = sol.prob.f

    λ, grad, y, dgrad, dy = split_states(u, t, S)

    dy, dλ, dgrad = vecjacobian(y, λ, p, t, S; dgrad = dgrad, dy = dy)
    dλ *= (-one(eltype(λ)))

    if !discrete
        dλ, dgrad = accumulate_cost(dλ, y, p, t, S, dgrad)
    end
    return dλ
end

function split_states(du, u, t, S::ODEGaussAdjointSensitivityFunction; update = true)
    (; sol, y, checkpoint_sol, discrete, prob, f, GaussInt) = S
    if update
        if checkpoint_sol === nothing
            if t isa ForwardDiff.Dual && eltype(S.y) <: AbstractFloat
                y = sol(t, continuity = :right)
            else
                sol(y, t, continuity = :right)
            end
        else
            intervals = checkpoint_sol.intervals
            interval = intervals[checkpoint_sol.cursor]
            if !(interval[1] <= t <= interval[2])
                cursor′ = Gaussfindcursor(intervals, t)
                interval = intervals[cursor′]
                cpsol_t = current_time(checkpoint_sol.cpsol)
                if t isa ForwardDiff.Dual && eltype(S.y) <: AbstractFloat
                    y = sol(interval[1])
                else
                    sol(y, interval[1])
                end
                if checkpoint_sol.tstops === nothing
                    prob′ = remake(prob, tspan = intervals[cursor′], u0 = y)
                    cpsol′ = solve(prob′, sol.alg;
                        dt = abs(cpsol_t[end] - cpsol_t[end - 1]),
                        checkpoint_sol.tols...)
                else
                    if maximum(interval[1] .< checkpoint_sol.tstops .< interval[2])
                        # callback might have changed p
                        _p = reset_p(prob.kwargs[:callback], interval)
                        prob′ = remake(prob, tspan = intervals[cursor′], u0 = y, p = _p)
                        cpsol′ = solve(prob′, sol.alg;
                            dt = abs(cpsol_t[end] - cpsol_t[end - 1]),
                            tstops = checkpoint_sol.tstops,
                            checkpoint_sol.tols...)
                    else
                        prob′ = remake(prob, tspan = intervals[cursor′], u0 = y)
                        cpsol′ = solve(prob′, sol.alg;
                            dt = abs(cpsol_t[end] - cpsol_t[end - 1]),
                            tstops = checkpoint_sol.tstops,
                            checkpoint_sol.tols...)
                    end
                end
                checkpoint_sol.cpsol = cpsol′
                checkpoint_sol.cursor = cursor′
                GaussInt.sol = cpsol′
            end
            checkpoint_sol.cpsol(y, t, continuity = :right)
        end
    end
    λ = u
    dλ = du
    λ, nothing, y, dλ, nothing, nothing
end

function split_states(u, t, S::ODEGaussAdjointSensitivityFunction; update = true)
    (; y, sol) = S

    if update
        y = sol(t, continuity = :right)
    end

    λ = u

    λ, nothing, y, nothing, nothing
end

@noinline function ODEAdjointProblem(sol, sensealg::AbstractGAdjoint, alg,
        GaussInt::GaussIntegrand, integrating_cb,
        t = nothing,
        dgdu_discrete::DG1 = nothing,
        dgdp_discrete::DG2 = nothing,
        dgdu_continuous::DG3 = nothing,
        dgdp_continuous::DG4 = nothing,
        g::G = nothing,
        ::Val{RetCB} = Val(false);
        checkpoints = current_time(sol),
        callback = CallbackSet(), no_start = false,
        reltol = nothing, abstol = nothing, kwargs...) where {DG1, DG2, DG3, DG4, G,
        RetCB}
    dgdu_discrete === nothing && dgdu_continuous === nothing && g === nothing &&
        error("Either `dgdu_discrete`, `dgdu_continuous`, or `g` must be specified.")
    t !== nothing && dgdu_discrete === nothing && dgdp_discrete === nothing &&
        error("It looks like you're using the direct `adjoint_sensitivities` interface
               with a discrete cost function but no specified `dgdu_discrete` or `dgdp_discrete`.
               Please use the higher level `solve` interface or specify these two contributions.")

    (; p, u0, tspan) = sol.prob

    ## Force recompile mode until vjps are specialized to handle this!!!
    f = if sol.prob.f isa ODEFunction &&
           sol.prob.f.f isa FunctionWrappersWrappers.FunctionWrappersWrapper
        ODEFunction{isinplace(sol.prob), true}(unwrapped_f(sol.prob.f))
    else
        sol.prob.f
    end

    terminated = false
    if hasfield(typeof(sol), :retcode)
        if sol.retcode == ReturnCode.Terminated
            tspan = (tspan[1], last(current_time(sol)))
            terminated = true
        end
    end
    tspan = reverse(tspan)

    discrete = (t !== nothing &&
                (dgdu_continuous === nothing && dgdp_continuous === nothing ||
                 g !== nothing))

    # remove duplicates from checkpoints
    if ischeckpointing(sensealg, sol) &&
       (length(unique(checkpoints)) != length(checkpoints))
        _checkpoints, duplicate_iterator_times = separate_nonunique(checkpoints)
        tstops = duplicate_iterator_times[1]
        checkpoints = filter(x -> x ∉ tstops, _checkpoints)
        # check if start is in checkpoints. Otherwise first interval is missed.
        if checkpoints[1] != tspan[2]
            pushfirst!(checkpoints, tspan[2])
        end

        if haskey(kwargs, :tstops)
            (tstops !== kwargs[:tstops]) && unique!(push!(tstops, kwargs[:tstops]...))
        end

        # check if end is in checkpoints.
        if checkpoints[end] != tspan[1]
            push!(checkpoints, tspan[1])
        end
    else
        tstops = nothing
    end

    if ArrayInterface.ismutable(u0)
        len = length(u0)
        λ = similar(u0, len)
        λ .= false
    else
        λ = zero(u0)
    end
    sense = ODEGaussAdjointSensitivityFunction(g, sensealg, GaussInt, discrete, sol,
        dgdu_continuous, dgdp_continuous, f, alg, checkpoints, integrating_cb,
        (reltol = reltol, abstol = abstol), tstops, tspan = tspan)

    init_cb = (discrete || dgdu_discrete !== nothing) # && tspan[1] == t[end]
    z0 = vec(zero(λ))
    cb, rcb,
    _ = generate_callbacks(sense, dgdu_discrete, dgdp_discrete,
        λ, t, tspan[2],
        callback, init_cb, terminated, no_start)

    jac_prototype = sol.prob.f.jac_prototype
    adjoint_jac_prototype = !sense.discrete || jac_prototype === nothing ? nothing :
                            copy(jac_prototype')

    original_mm = sol.prob.f.mass_matrix
    if original_mm === I || original_mm === (I, I)
        odefun = ODEFunction{ArrayInterface.ismutable(z0), true}(sense,
            jac_prototype = adjoint_jac_prototype)
    else
        odefun = ODEFunction{ArrayInterface.ismutable(z0), true}(sense,
            mass_matrix = sol.prob.f.mass_matrix',
            jac_prototype = adjoint_jac_prototype)
    end
    if RetCB
        return ODEProblem(odefun, z0, tspan, p), cb, rcb
    else
        return ODEProblem(odefun, z0, tspan, p, callback = cb), cb, rcb
    end
end

function Gaussreset_p(CBS, interval)
    # check which events are close to tspan[1]
    if !isempty(CBS.discrete_callbacks)
        ts = map(CBS.discrete_callbacks) do cb
            indx = searchsortedfirst(cb.affect!.event_times, interval[1])
            (indx, cb.affect!.event_times[indx])
        end
        perm = minimum(sortperm([t for t in getindex.(ts, 2)]))
    end

    if !isempty(CBS.continuous_callbacks)
        ts2 = map(CBS.continuous_callbacks) do cb
            if !isempty(cb.affect!.event_times) && isempty(cb.affect_neg!.event_times)
                indx = searchsortedfirst(cb.affect!.event_times, interval[1])
                return (indx, cb.affect!.event_times[indx], 0) # zero for affect!
            elseif isempty(cb.affect!.event_times) && !isempty(cb.affect_neg!.event_times)
                indx = searchsortedfirst(cb.affect_neg!.event_times, interval[1])
                return (indx, cb.affect_neg!.event_times[indx], 1) # one for affect_neg!
            elseif !isempty(cb.affect!.event_times) && !isempty(cb.affect_neg!.event_times)
                indx1 = searchsortedfirst(cb.affect!.event_times, interval[1])
                indx2 = searchsortedfirst(cb.affect_neg!.event_times, interval[1])
                if cb.affect!.event_times[indx1] < cb.affect_neg!.event_times[indx2]
                    return (indx1, cb.affect!.event_times[indx1], 0)
                else
                    return (indx2, cb.affect_neg!.event_times[indx2], 1)
                end
            else
                error("Expected event but reset_p couldn't find event time. Please report this error.")
            end
        end
        perm2 = minimum(sortperm([t for t in getindex.(ts2, 2)]))
        # check if continuous or discrete callback was applied first if both occur in interval
        if isempty(CBS.discrete_callbacks)
            if ts2[perm2][3] == 0
                p = deepcopy(CBS.continuous_callbacks[perm2].affect!.pleft[getindex.(ts2, 1)[perm2]])
            else
                p = deepcopy(CBS.continuous_callbacks[perm2].affect_neg!.pleft[getindex.(
                    ts2,
                    1)[perm2]])
            end
        else
            if ts[perm][2] < ts2[perm2][2]
                p = deepcopy(CBS.discrete_callbacks[perm].affect!.pleft[getindex.(ts, 1)[perm]])
            else
                if ts2[perm2][3] == 0
                    p = deepcopy(CBS.continuous_callbacks[perm2].affect!.pleft[getindex.(
                        ts2,
                        1)[perm2]])
                else
                    p = deepcopy(CBS.continuous_callbacks[perm2].affect_neg!.pleft[getindex.(
                        ts2,
                        1)[perm2]])
                end
            end
        end
    else
        p = deepcopy(CBS.discrete_callbacks[perm].affect!.pleft[getindex.(ts, 1)[perm]])
    end

    return p
end

"""
    DDEHistoryWrapper

A wrapper that provides a history function for DDEs using the forward solution.
For times before t0, it uses the original history function from the problem.
For times >= t0, it interpolates from the forward solution.
"""
struct DDEHistoryWrapper{S, H, T}
    sol::S          # Forward DDE solution
    orig_h::H       # Original history function from problem
    t0::T           # Start time of the problem
end

function (h::DDEHistoryWrapper)(p, t)
    if t < h.t0
        return h.orig_h(p, t)
    else
        return h.sol(t)
    end
end

function GaussIntegrand(sol, sensealg, checkpoints, dgdp = nothing)
    prob = sol.prob
    (; f, tspan) = prob
    u0 = state_values(prob)
    p = parameter_values(prob)

    if p === nothing || p isa SciMLBase.NullParameters
        tunables, repack = p, identity
    elseif isscimlstructure(p)
        tunables, repack, _ = canonicalize(Tunable(), p)
    else
        tunables, repack = Functors.functor(p)
    end

    numparams = length(tunables)
    y = zero(state_values(prob))
    λ = zero(state_values(prob))
    # we need to alias `y`
    f_cache = zero(y)
    isautojacvec = get_jacvec(sensealg)

    unwrappedf = unwrapped_f(f)

    dgdp_cache = dgdp === nothing ? nothing : zero(p)

    # Check if this is a DDE problem and create history wrapper
    is_dde = prob isa SciMLBase.AbstractDDEProblem
    dde_history = if is_dde
        DDEHistoryWrapper(sol, prob.h, tspan[1])
    else
        nothing
    end

    if sensealg.autojacvec isa ReverseDiffVJP
        tape = if DiffEqBase.isinplace(prob)
            if is_dde
                # DDE in-place: f(du, u, h, p, t)
                # Create a tape that wraps the history function
                ReverseDiff.GradientTape((y, tunables, [tspan[2]])) do u, tunables, t
                    du1 = similar(tunables, size(u))
                    du1 .= false
                    # Create a history function that uses the current solution interpolant
                    h_t = (p_arg, t_arg) -> sol(t_arg)
                    unwrappedf(
                        du1, u, h_t, SciMLStructures.replace(Tunable(), p, tunables), first(t))
                    return vec(du1)
                end
            else
                ReverseDiff.GradientTape((y, tunables, [tspan[2]])) do u, tunables, t
                    du1 = similar(tunables, size(u))
                    du1 .= false
                    unwrappedf(
                        du1, u, SciMLStructures.replace(Tunable(), p, tunables), first(t))
                    return vec(du1)
                end
            end
        else
            if is_dde
                # DDE out-of-place: f(u, p, h, t)
                ReverseDiff.GradientTape((y, tunables, [tspan[2]])) do u, tunables, t
                    h_t = (p_arg, t_arg) -> sol(t_arg)
                    vec(unwrappedf(
                        u, SciMLStructures.replace(Tunable(), p, tunables), h_t, first(t)))
                end
            else
                ReverseDiff.GradientTape((y, tunables, [tspan[2]])) do u, tunables, t
                    vec(unwrappedf(
                        u, SciMLStructures.replace(Tunable(), p, tunables), first(t)))
                end
            end
        end
        if compile_tape(sensealg.autojacvec)
            paramjac_config = ReverseDiff.compile(tape)
        else
            paramjac_config = tape
        end
        pf = nothing
        pJ = nothing
    elseif sensealg.autojacvec isa EnzymeVJP
        pf = SciMLBase.isinplace(sol.prob.f) ? SciMLBase.Void(unwrappedf) : unwrappedf
        paramjac_config = zero(y), zero(y), Enzyme.make_zero(pf)
        pJ = nothing
    elseif sensealg.autojacvec isa MooncakeVJP
        pf = get_pf(sensealg.autojacvec, prob, f)
        paramjac_config = get_paramjac_config(
            MooncakeLoaded(), sensealg.autojacvec, pf, tunables, f, y, tspan[2])
        pJ = nothing
    elseif isautojacvec # Zygote
        paramjac_config = nothing
        pf = nothing
        pJ = nothing
    else
        if is_dde
            # For DDEs with numerical Jacobian, wrap the function
            pf = SciMLBase.ParamJacobianWrapper(
                (du, u, p_arg, t) -> begin
                    h_t = (p_h, t_h) -> sol(t_h)
                    unwrappedf(du, u, h_t, repack(p_arg), t)
                end, tspan[1], y)
        else
            pf = SciMLBase.ParamJacobianWrapper((du, u, p, t)->unwrappedf(du, u, repack(p), t), tspan[1], y)
        end
        pJ = similar(u0, length(u0), numparams)
        paramjac_config = build_param_jac_config(sensealg, pf, y, tunables)
    end

    cpsol = sol

    GaussIntegrand(cpsol, p, y, λ, pf, f_cache, pJ, paramjac_config,
        sensealg, dgdp_cache, dgdp, dde_history)
end

function g(f, du, u, p, t)
    Base.copyto!(du, f(u, p, t))
    nothing
end

# out = λ df(u, p, t)/dp at u=y, p=p, t=t
function vec_pjac!(out, λ, y, t, S::GaussIntegrand)
    (; pJ, pf, p, f_cache, dgdp_cache, paramjac_config, sensealg, sol, dde_history) = S
    f = sol.prob.f
    f = unwrapped_f(f)
    isautojacvec = get_jacvec(sensealg)
    is_dde = dde_history !== nothing
    # y is aliased
    if p === nothing || p isa SciMLBase.NullParameters
        tunables, repack = p, identity
    elseif isscimlstructure(p)
        tunables, repack, _ = canonicalize(Tunable(), p)
    else
        tunables, repack = Functors.functor(p)
    end

    if !isautojacvec
        if SciMLBase.has_paramjac(f)
            f.paramjac(pJ, y, p, t) # Calculate the parameter Jacobian into pJ
        else
            pf.t = t
            pf.u = y
            jacobian!(pJ, pf, tunables, f_cache, sensealg, paramjac_config)
        end
        mul!(out', λ', pJ)
    elseif sensealg.autojacvec isa ReverseDiffVJP
        tape = paramjac_config
        tu, tp, tt = ReverseDiff.input_hook(tape)
        output = ReverseDiff.output_hook(tape)
        ReverseDiff.unseed!(tu) # clear any "leftover" derivatives from previous calls
        ReverseDiff.unseed!(tp)
        ReverseDiff.unseed!(tt)
        ReverseDiff.value!(tu, y)
        ReverseDiff.value!(tp, tunables)
        ReverseDiff.value!(tt, [t])
        ReverseDiff.forward_pass!(tape)
        ReverseDiff.increment_deriv!(output, λ)
        ReverseDiff.reverse_pass!(tape)
        copyto!(vec(out), ReverseDiff.deriv(tp))
    elseif sensealg.autojacvec isa ZygoteVJP
        if is_dde
            # History is treated as known (not differentiated through)
            t0 = sol.prob.tspan[1]
            orig_h = sol.prob.h
            h_t = (p_arg, t_arg) -> begin
                Zygote.ignore() do
                    if t_arg < t0
                        orig_h(p_arg, t_arg)
                    else
                        sol(t_arg)
                    end
                end
            end
            if SciMLBase.isinplace(sol.prob)
                # DDE in-place: f(du, u, h, p, t)
                _dy, back = Zygote.pullback(tunables) do tunables
                    out_ = Zygote.Buffer(similar(y))
                    f(out_, y, h_t, repack(tunables), t)
                    vec(copy(out_))
                end
            else
                # DDE out-of-place: f(u, p, h, t)
                _dy, back = Zygote.pullback(tunables) do tunables
                    vec(f(y, repack(tunables), h_t, t))
                end
            end
        else
            _dy, back = Zygote.pullback(tunables) do tunables
                vec(f(y, repack(tunables), t))
            end
        end
        tmp = back(λ)
        if tmp[1] === nothing
            recursive_copyto!(out, 0)
        else
            recursive_copyto!(out, tmp[1])
        end
    elseif sensealg.autojacvec isa EnzymeVJP
        tmp3, tmp4, tmp6 = paramjac_config
        vtmp4 = vec(tmp4)
        vtmp4 .= λ
        Enzyme.remake_zero!(tmp3)
        Enzyme.remake_zero!(out)

        if SciMLBase.isinplace(sol.prob.f)
            Enzyme.remake_zero!(tmp6)

            if is_dde
                # For DDEs, we need to wrap the function with history
                h_t = (p_arg, t_arg) -> sol(t_arg)
                # Create a wrapped function for Enzyme that includes history
                wrapped_f = SciMLBase.Void((du, u, p_arg, t_arg) -> begin
                    f(du, u, h_t, p_arg, t_arg)
                    nothing
                end)
                tmp6_dde = Enzyme.make_zero(wrapped_f)
                Enzyme.autodiff(
                    sensealg.autojacvec.mode, Enzyme.Duplicated(wrapped_f, tmp6_dde), Enzyme.Const,
                    Enzyme.Duplicated(tmp3, tmp4),
                    Enzyme.Const(y), Enzyme.Duplicated(p, out), Enzyme.Const(t))
            else
                Enzyme.autodiff(
                    sensealg.autojacvec.mode, Enzyme.Duplicated(pf, tmp6), Enzyme.Const,
                    Enzyme.Duplicated(tmp3, tmp4),
                    Enzyme.Const(y), Enzyme.Duplicated(p, out), Enzyme.Const(t))
            end
        else
            tmp6 = Enzyme.make_zero(f)
            if is_dde
                # For out-of-place DDEs
                h_t = (p_arg, t_arg) -> sol(t_arg)
                wrapped_f = (u, p_arg, t_arg) -> f(u, p_arg, h_t, t_arg)
                tmp6_dde = Enzyme.make_zero(wrapped_f)
                Enzyme.autodiff(
                    sensealg.autojacvec.mode, Enzyme.Const(gclosure3), Enzyme.Duplicated(wrapped_f, tmp6_dde), Enzyme.Const,
                    Enzyme.Duplicated(tmp3, tmp4),
                    Enzyme.Const(y), Enzyme.Duplicated(p, out), Enzyme.Const(t))
            else
                Enzyme.autodiff(
                    sensealg.autojacvec.mode, Enzyme.Const(gclosure3), Enzyme.Duplicated(f, tmp6), Enzyme.Const,
                    Enzyme.Duplicated(tmp3, tmp4),
                    Enzyme.Const(y), Enzyme.Duplicated(p, out), Enzyme.Const(t))
            end
        end
    elseif sensealg.autojacvec isa MooncakeVJP
        _, _, p_grad = mooncake_run_ad(paramjac_config, y, p, t, λ)
        out .= p_grad
    else
        error("autojacvec choice $(sensealg.autojacvec) is not supported by GaussAdjoint")
    end
    # TODO: Add tracker?
    return out
end

function (S::GaussIntegrand)(out, t, λ)
    (; y, pJ, pf, p, f_cache, dgdp_cache, paramjac_config, sensealg, sol) = S
    if ArrayInterface.ismutable(y)
        sol(y, t)
    else
        y = sol(t)
    end
    vec_pjac!(out, λ, y, t, S)
    out = recursive_neg!(out)
    if S.dgdp !== nothing
        S.dgdp(dgdp_cache, y, p, t)
        out .+= dgdp_cache
    end
    out
end

function (S::GaussIntegrand)(t, λ)
    out = allocate_zeros(S.p)
    S(out, t, λ)
end

function _adjoint_sensitivities(sol, sensealg::AbstractGAdjoint, alg; t = nothing,
        dgdu_discrete = nothing,
        dgdp_discrete = nothing,
        dgdu_continuous = nothing,
        dgdp_continuous = nothing,
        g = nothing,
        abstol = 1e-6, reltol = 1e-3,
        checkpoints = current_time(sol),
        corfunc_analytical = false,
        callback = CallbackSet(), no_start = false,
        kwargs...)
    p = SymbolicIndexingInterface.parameter_values(sol)
    if !isscimlstructure(p) && !isfunctor(p)
        throw(SciMLStructuresCompatibilityError())
    end

    if p === nothing || p isa SciMLBase.NullParameters
        tunables, repack = p, identity
    elseif isscimlstructure(p)
        tunables, repack, _ = canonicalize(Tunable(), p)
    elseif isfunctor(p)
        tunables, repack = Functors.functor(p)
    else
        throw(SciMLStructuresCompatibilityError())
    end
    integrand = GaussIntegrand(sol, sensealg, checkpoints, dgdp_continuous)
    integrand_values = IntegrandValuesSum(allocate_zeros(tunables))
    if sensealg isa GaussAdjoint
        cb = IntegratingSumCallback((out, u, t, integrator) -> integrand(out, t, u),
            integrand_values, allocate_vjp(tunables))
    elseif sensealg isa GaussKronrodAdjoint
        cb = IntegratingGKSumCallback((out, u, t, integrator) -> integrand(out, t, u),
            integrand_values, allocate_vjp(tunables))
    end
    rcb = nothing
    cb2 = nothing
    adj_prob = nothing

    # For DDEs, the adjoint problem is still an ODE (no delays in adjoint equation)
    # We need to extract the underlying ODE algorithm from MethodOfSteps
    adj_alg = if sol.prob isa SciMLBase.AbstractDDEProblem && hasfield(typeof(alg), :alg)
        alg.alg  # Extract underlying ODE algorithm from MethodOfSteps
    else
        alg
    end

    if sol.prob isa ODEProblem || sol.prob isa SciMLBase.AbstractDDEProblem
        # For DDEs, the adjoint problem is still an ODE (no delays in adjoint equation)
        # The DDE-specific handling is done in GaussIntegrand via history wrapper
        adj_prob, cb2,
        rcb = ODEAdjointProblem(
            sol, sensealg, adj_alg, integrand, cb,
            t, dgdu_discrete,
            dgdp_discrete,
            dgdu_continuous, dgdp_continuous, g, Val(true);
            checkpoints = checkpoints,
            callback = callback, no_start = no_start,
            abstol = abstol, reltol = reltol, kwargs...)
    else
        error("Continuous adjoint sensitivities are only supported for ODE and DDE problems.")
    end

    tstops = ischeckpointing(sensealg, sol) ? checkpoints : similar(current_time(sol), 0)

    adj_sol = solve(
        adj_prob, adj_alg; abstol = abstol, reltol = reltol, save_everystep = false,
        save_start = false, save_end = true, saveat = eltype(state_values(sol, 1))[], tstops = tstops,
        callback = CallbackSet(cb, cb2), kwargs...)
    res = integrand_values.integrand

    if rcb !== nothing && !isempty(rcb.Δλas)
        iλ = zero(rcb.λ)
        out = zero(res)
        yy = similar(rcb.y)
        yy .= 0
        for (Δλa, tt) in rcb.Δλas
            (; algevar_idxs) = rcb.diffcache
            iλ[algevar_idxs] .= Δλa
            sol(yy, tt)
            vec_pjac!(out, iλ, yy, tt, integrand)
            res .+= out
            iλ .= zero(eltype(iλ))
        end
    end

    return state_values(adj_sol)[end], __maybe_adjoint(res)
end

__maybe_adjoint(x::AbstractArray) = x'
__maybe_adjoint(x) = x
