mutable struct GaussIntegrand{pType, uType, lType, rateType, S, PF, PJC, PJT, DGP,
    G}
    sol::S
    p::pType
    y::uType
    λ::lType
    pf::PF
    f_cache::rateType
    pJ::PJT
    paramjac_config::PJC
    sensealg::GaussAdjoint
    dgdp_cache::DGP
    dgdp::G
end

struct ODEGaussAdjointSensitivityFunction{C <: AdjointDiffCache,
    Alg <: GaussAdjoint,
    uType, SType, CPS, pType,
    fType <: AbstractDiffEqFunction} <: SensitivityFunction
    diffcache::C
    sensealg::Alg
    discrete::Bool
    y::uType
    sol::SType
    checkpoint_sol::CPS
    prob::pType
    f::fType
    GaussInt::GaussIntegrand
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
        checkpoints, tols, tstops = nothing;
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
    diffcache, y = adjointdiffcache(
        g, sensealg, discrete, sol, dgdu, dgdp, sol.prob.f, alg;
        quad = true)
    return ODEGaussAdjointSensitivityFunction(diffcache, sensealg, discrete,
        y, sol, checkpoint_sol, sol.prob, f, gaussint)
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

@noinline function ODEAdjointProblem(sol, sensealg::GaussAdjoint, alg,
        GaussInt::GaussIntegrand,
        t = nothing,
        dgdu_discrete::DG1 = nothing,
        dgdp_discrete::DG2 = nothing,
        dgdu_continuous::DG3 = nothing,
        dgdp_continuous::DG4 = nothing,
        g::G = nothing,
        ::Val{RetCB} = Val(false);
        checkpoints = current_time(sol),
        callback = CallbackSet(),
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
        dgdu_continuous, dgdp_continuous, f, alg, checkpoints,
        (reltol = reltol, abstol = abstol), tstops, tspan = tspan)

    init_cb = (discrete || dgdu_discrete !== nothing) # && tspan[1] == t[end]
    z0 = vec(zero(λ))
    cb, rcb, _ = generate_callbacks(sense, dgdu_discrete, dgdp_discrete,
        λ, t, tspan[2],
        callback, init_cb, terminated)

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

    if sensealg.autojacvec isa ReverseDiffVJP
        tape = if DiffEqBase.isinplace(prob)
            ReverseDiff.GradientTape((y, tunables, [tspan[2]])) do u, tunables, t
                du1 = similar(tunables, size(u))
                du1 .= false
                unwrappedf(
                    du1, u, SciMLStructures.replace(Tunable(), p, tunables), first(t))
                return vec(du1)
            end
        else
            ReverseDiff.GradientTape((y, tunables, [tspan[2]])) do u, tunables, t
                vec(unwrappedf(
                    u, SciMLStructures.replace(Tunable(), p, tunables), first(t)))
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
        pf = let f = unwrappedf
            if DiffEqBase.isinplace(prob)
                function (out, u, _p, t)
                    f(out, u, _p, t)
                    nothing
                end
            else
                !DiffEqBase.isinplace(prob)
                function (out, u, _p, t)
                    out .= f(u, _p, t)
                    nothing
                end
            end
        end
        paramjac_config = zero(y), zero(y), Enzyme.make_zero(pf)
        pJ = nothing
    elseif sensealg.autojacvec isa MooncakeVJP
        pf = get_pf(sensealg.autojacvec, prob, f)
        paramjac_config = get_paramjac_config(
            MooncakeLoaded(), sensealg.autojacvec, pf, p, f, y, tspan[2])
        pJ = nothing
    elseif isautojacvec # Zygote
        paramjac_config = nothing
        pf = nothing
        pJ = nothing
    else
        pf = SciMLBase.ParamJacobianWrapper(unwrappedf, tspan[1], y)
        pJ = similar(u0, length(u0), numparams)
        paramjac_config = build_param_jac_config(sensealg, pf, y, p)
    end

    cpsol = sol

    GaussIntegrand(cpsol, p, y, λ, pf, f_cache, pJ, paramjac_config,
        sensealg, dgdp_cache, dgdp)
end

# out = λ df(u, p, t)/dp at u=y, p=p, t=t
function vec_pjac!(out, λ, y, t, S::GaussIntegrand)
    (; pJ, pf, p, f_cache, dgdp_cache, paramjac_config, sensealg, sol) = S
    f = sol.prob.f
    isautojacvec = get_jacvec(sensealg)
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
            jacobian!(pJ, pf, p, f_cache, sensealg, paramjac_config)
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
        _dy, back = Zygote.pullback(tunables) do tunables
            vec(f(y, repack(tunables), t))
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
        out .= 0
        Enzyme.autodiff(
            Enzyme.Reverse, Enzyme.Duplicated(pf, tmp6), Enzyme.Const,
            Enzyme.Duplicated(tmp3, tmp4),
            Enzyme.Const(y), Enzyme.Duplicated(p, out), Enzyme.Const(t))
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

function _adjoint_sensitivities(sol, sensealg::GaussAdjoint, alg; t = nothing,
        dgdu_discrete = nothing,
        dgdp_discrete = nothing,
        dgdu_continuous = nothing,
        dgdp_continuous = nothing,
        g = nothing,
        abstol = 1e-6, reltol = 1e-3,
        checkpoints = current_time(sol),
        corfunc_analytical = false,
        callback = CallbackSet(),
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
    cb = IntegratingSumCallback((out, u, t, integrator) -> integrand(out, t, u),
        integrand_values, allocate_vjp(tunables))
    rcb = nothing
    cb2 = nothing
    adj_prob = nothing

    if sol.prob isa ODEProblem
        adj_prob, cb2, rcb = ODEAdjointProblem(
            sol, sensealg, alg, integrand, t, dgdu_discrete,
            dgdp_discrete,
            dgdu_continuous, dgdp_continuous, g, Val(true);
            checkpoints = checkpoints,
            callback = callback,
            abstol = abstol, reltol = reltol, kwargs...)
    else
        error("Continuous adjoint sensitivities are only supported for ODE problems.")
    end

    tstops = ischeckpointing(sensealg, sol) ? checkpoints : similar(current_time(sol), 0)

    adj_sol = solve(
        adj_prob, alg; abstol = abstol, reltol = reltol, save_everystep = false,
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

function update_p_integrand(integrand::GaussIntegrand, p)
    (; sol, y, λ, pf, f_cache, pJ, paramjac_config, sensealg, dgdp_cache, dgdp) = integrand
    GaussIntegrand(sol, p, y, λ, pf, f_cache, pJ, paramjac_config,
        sensealg, dgdp_cache, dgdp)
end

function update_integrand_and_dgrad(res, sensealg::GaussAdjoint, callbacks, integrand,
        adj_prob, sol, dgdu_discrete, dgdp_discrete, dλ, dgrad,
        ti, cur_time)
    for cb in callbacks.discrete_callbacks
        if ti ∈ cb.affect!.event_times
            integrand = _update_integrand_and_dgrad(res, sensealg, cb,
                integrand, adj_prob, sol,
                dgdu_discrete,
                dgdp_discrete, dλ, dgrad,
                ti, cur_time)
        end
    end
    for cb in callbacks.continuous_callbacks
        if ti ∈ cb.affect!.event_times ||
           ti ∈ cb.affect_neg!.event_times
            integrand = _update_integrand_and_dgrad(res, sensealg, cb,
                integrand, adj_prob, sol,
                dgdu_discrete,
                dgdp_discrete, dλ, dgrad,
                ti, cur_time)
        end
    end
    return integrand
end

function _update_integrand_and_dgrad(res, sensealg::GaussAdjoint, cb, integrand,
        adj_prob, sol, dgdu, dgdp, dλ, dgrad, t, cur_time)
    indx, pos_neg = get_indx(cb, t)
    tprev = get_tprev(cb, indx, pos_neg)

    wp = let tprev = tprev, pos_neg = pos_neg
        function (dp, p, u, t)
            _affect! = get_affect!(cb, pos_neg)
            fakeinteg = FakeIntegrator([x for x in u], [x for x in p], t, tprev)
            _affect!(fakeinteg)
            dp .= fakeinteg.p
        end
    end

    _p = similar(integrand.p, size(integrand.p))
    _p .= 0
    wp(_p, integrand.p, integrand.y, t)

    if _p != integrand.p
        fakeSp = CallbackSensitivityFunction(wp, sensealg, adj_prob.f.f.diffcache, sol.prob)
        #vjp with Jacobin given by dw/dp before event and vector given by grad
        vecjacobian!(res, integrand.p, res, integrand.y, t, fakeSp;
            dgrad = nothing, dy = nothing)
        integrand = update_p_integrand(integrand, _p)
    end

    w = let tprev = tprev, pos_neg = pos_neg
        function (du, u, p, t)
            _affect! = get_affect!(cb, pos_neg)
            fakeinteg = FakeIntegrator([x for x in u], [x for x in p], t, tprev)
            _affect!(fakeinteg)
            du .= vec(fakeinteg.u)
        end
    end

    # Create a fake sensitivity function to do the vjps needs to be done
    # to account for parameter dependence of affect function
    fakeS = CallbackSensitivityFunction(w, sensealg, adj_prob.f.f.diffcache, sol.prob)
    if dgdu !== nothing # discrete cost
        dgdu(dλ, integrand.y, integrand.p, t, cur_time)
    else
        error("Please provide `dgdu` to use adjoint_sensitivities with `GaussAdjoint()` and callbacks.")
    end

    @assert dgdp === nothing

    # account for implicit events

    @. dλ = -dλ - integrand.λ
    vecjacobian!(dλ, integrand.y, dλ, integrand.p, t, fakeS; dgrad = dgrad)
    res .-= dgrad
    return integrand
end
