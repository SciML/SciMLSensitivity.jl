struct AdjointDiffCache{UF, PF, G, TJ, PJT, uType, JC, GC, PJC, JNC, PJNC, rateType, DG1,
                        DG2, DI,
                        AI, FM}
    uf::UF
    pf::PF
    g::G
    J::TJ
    pJ::PJT
    dg_val::uType
    jac_config::JC
    g_grad_config::GC
    paramjac_config::PJC
    jac_noise_config::JNC
    paramjac_noise_config::PJNC
    f_cache::rateType
    dgdu::DG1
    dgdp::DG2
    diffvar_idxs::DI
    algevar_idxs::AI
    factorized_mass_matrix::FM
    issemiexplicitdae::Bool
end

"""
    adjointdiffcache(g,sensealg,discrete,sol,dg,alg;quad=false)

return (AdjointDiffCache, y)
"""
function adjointdiffcache(g::G, sensealg, discrete, sol, dgdu::DG1, dgdp::DG2, f, alg;
                          quad = false,
                          noiseterm = false, needs_jac = false) where {G, DG1, DG2}
    prob = sol.prob
    if prob isa Union{SteadyStateProblem, NonlinearProblem}
        @unpack u0, p = prob
        tspan = (nothing, nothing)
        #elseif prob isa SDEProblem
        #  @unpack tspan, u0, p = prob
    else
        @unpack u0, p, tspan = prob
    end

    # Remove any function wrappers: it breaks autodiff
    unwrappedf = unwrapped_f(f)

    numparams = p === nothing || p === DiffEqBase.NullParameters() ? 0 : length(p)
    numindvar = length(u0)
    isautojacvec = get_jacvec(sensealg)

    issemiexplicitdae = false
    mass_matrix = sol.prob.f.mass_matrix
    if mass_matrix isa UniformScaling
        factorized_mass_matrix = mass_matrix'
    elseif mass_matrix isa Tuple{UniformScaling, UniformScaling}
        factorized_mass_matrix = (I', I')
    else
        mass_matrix = mass_matrix'
        diffvar_idxs = findall(x -> any(!iszero, @view(mass_matrix[:, x])),
                               axes(mass_matrix, 2))
        algevar_idxs = setdiff(eachindex(u0), diffvar_idxs)
        # TODO: operator
        M̃ = @view mass_matrix[diffvar_idxs, diffvar_idxs]
        factorized_mass_matrix = lu(M̃, check = false)
        issuccess(factorized_mass_matrix) ||
            error("The submatrix corresponding to the differential variables of the mass matrix must be nonsingular!")
        isempty(algevar_idxs) || (issemiexplicitdae = true)
    end
    if !issemiexplicitdae
        diffvar_idxs = eachindex(u0)
        algevar_idxs = 1:0
    end

    if !needs_jac && !(issemiexplicitdae || !isautojacvec)
        J = nothing
    else
        if SciMLBase.forwarddiffs_model_time(alg)
            # 1 chunk is fine because it's only t
            J = dualcache(similar(u0, numindvar, numindvar),
                          ForwardDiff.pickchunksize(length(u0)))
        else
            J = similar(u0, numindvar, numindvar)
        end
    end

    if !discrete
        if dgdu !== nothing
            pg = nothing
            pg_config = nothing
            if dgdp !== nothing
                dg_val = (similar(u0, numindvar), similar(u0, numparams))
                dg_val[1] .= false
                dg_val[2] .= false
            else
                dg_val = similar(u0, numindvar) # number of funcs size
                dg_val .= false
            end
        else
            pgpu = UGradientWrapper(g, tspan[2], p)
            pgpu_config = build_grad_config(sensealg, pgpu, u0, p)
            pgpp = ParamGradientWrapper(g, tspan[2], u0)
            pgpp_config = build_grad_config(sensealg, pgpp, p, p)
            pg = (pgpu, pgpp)
            pg_config = (pgpu_config, pgpp_config)
            dg_val = (similar(u0, numindvar), similar(u0, numparams))
            dg_val[1] .= false
            dg_val[2] .= false
        end
    else
        dg_val = nothing
        pg = nothing
        pg_config = nothing
    end

    if DiffEqBase.has_jac(f) || (J === nothing)
        jac_config = nothing
        uf = nothing
    else
        if DiffEqBase.isinplace(prob)
            if !(prob isa RODEProblem)
                uf = DiffEqBase.UJacobianWrapper(unwrappedf, tspan[2], p)
            else
                uf = RODEUJacobianWrapper(unwrappedf, tspan[2], p, last(sol.W))
            end
            jac_config = build_jac_config(sensealg, uf, u0)
        else
            if !(prob isa RODEProblem)
                uf = DiffEqBase.UDerivativeWrapper(unwrappedf, tspan[2], p)
            else
                uf = RODEUDerivativeWrapper(unwrappedf, tspan[2], p, last(sol.W))
            end
            jac_config = nothing
        end
    end

    if prob isa Union{SteadyStateProblem, NonlinearProblem}
        y = copy(sol.u)
    else
        y = copy(sol.u[end])
    end

    if typeof(prob.p) <: DiffEqBase.NullParameters
        _p = similar(y, (0,))
    else
        _p = prob.p
    end

    @assert sensealg.autojacvec !== nothing

    if sensealg.autojacvec isa ReverseDiffVJP
        if prob isa Union{SteadyStateProblem, NonlinearProblem}
            if DiffEqBase.isinplace(prob)
                tape = ReverseDiff.GradientTape((y, _p)) do u, p
                    du1 = p !== nothing && p !== DiffEqBase.NullParameters() ?
                          similar(p, size(u)) : similar(u)
                    unwrappedf(du1, u, p, nothing)
                    return vec(du1)
                end
            else
                tape = ReverseDiff.GradientTape((y, _p)) do u, p
                    vec(unwrappedf(u, p, nothing))
                end
            end
        elseif noiseterm &&
               (!StochasticDiffEq.is_diagonal_noise(prob) || isnoisemixing(sensealg))
            tape = nothing
        else
            if DiffEqBase.isinplace(prob)
                if !(prob isa RODEProblem)
                    tape = ReverseDiff.GradientTape((y, _p, [tspan[2]])) do u, p, t
                        du1 = (p !== nothing && p !== DiffEqBase.NullParameters()) ?
                              similar(p, size(u)) : similar(u)
                        unwrappedf(du1, u, p, first(t))
                        return vec(du1)
                    end
                else
                    tape = ReverseDiff.GradientTape((y, _p, [tspan[2]], last(sol.W))) do u,
                                                                                         p,
                                                                                         t,
                                                                                         W
                        du1 = p !== nothing && p !== DiffEqBase.NullParameters() ?
                              similar(p, size(u)) : similar(u)
                        unwrappedf(du1, u, p, first(t), W)
                        return vec(du1)
                    end
                end
            else
                if !(prob isa RODEProblem)
                    tape = ReverseDiff.GradientTape((y, _p, [tspan[2]])) do u, p, t
                        vec(unwrappedf(u, p, first(t)))
                    end
                else
                    tape = ReverseDiff.GradientTape((y, _p, [tspan[2]], last(sol.W))) do u,
                                                                                         p,
                                                                                         t,
                                                                                         W
                        return unwrappedf(u, p, first(t), W)
                    end
                end
            end
        end

        if compile_tape(sensealg.autojacvec)
            paramjac_config = ReverseDiff.compile(tape)
        else
            paramjac_config = tape
        end

        pf = nothing
    elseif sensealg.autojacvec isa EnzymeVJP
        if typeof(prob.p) <: DiffEqBase.NullParameters
            if SciMLBase.forwarddiffs_model(alg)
                chunk = if sensealg.autojacvec.chunksize == 0
                    ForwardDiff.pickchunksize(length(u0))
                else
                    sensealg.autojacvec.chunksize
                end

                paramjac_config = dualcache(zero(y), chunk), prob.p,
                                  dualcache(zero(y), chunk),
                                  dualcache(zero(y), chunk)
            else
                paramjac_config = zero(y), prob.p, zero(y), zero(y)
            end
        else
            if SciMLBase.forwarddiffs_model(alg)
                chunk = if sensealg.autojacvec.chunksize == 0
                    ForwardDiff.pickchunksize(length(u0))
                else
                    sensealg.autojacvec.chunksize
                end

                paramjac_config = dualcache(zero(y), chunk),
                                  zero(_p),
                                  dualcache(zero(y), chunk),
                                  dualcache(zero(y), chunk)
            else
                paramjac_config = zero(y), zero(_p), zero(y), zero(y)
            end
        end
        pf = let f = unwrappedf
            if DiffEqBase.isinplace(prob) && prob isa RODEProblem
                function (out, u, _p, t, W)
                    f(out, u, _p, t, W)
                    nothing
                end
            elseif DiffEqBase.isinplace(prob)
                function (out, u, _p, t)
                    f(out, u, _p, t)
                    nothing
                end
            elseif !DiffEqBase.isinplace(prob) && prob isa RODEProblem
                function (out, u, _p, t, W)
                    out .= f(u, _p, t, W)
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
    elseif DiffEqBase.has_paramjac(f) || isautojacvec || quad ||
           sensealg.autojacvec isa EnzymeVJP
        paramjac_config = nothing
        pf = nothing
    else
        if DiffEqBase.isinplace(prob) &&
           !(p === nothing || p === DiffEqBase.NullParameters())
            if !(prob isa RODEProblem)
                pf = DiffEqBase.ParamJacobianWrapper(unwrappedf, tspan[1], y)
            else
                pf = RODEParamJacobianWrapper(unwrappedf, tspan[1], y, last(sol.W))
            end
            paramjac_config = build_param_jac_config(sensealg, pf, y, p)
        else
            if !(prob isa RODEProblem)
                pf = ParamGradientWrapper(unwrappedf, tspan[2], y)
            else
                pf = RODEParamGradientWrapper(unwrappedf, tspan[2], y, last(sol.W))
            end
            paramjac_config = nothing
        end
    end

    pJ = (quad || isautojacvec) ? nothing : similar(u0, numindvar, numparams)

    f_cache = DiffEqBase.isinplace(prob) ? deepcopy(u0) : nothing

    if noiseterm
        if sensealg.autojacvec isa ReverseDiffVJP
            jac_noise_config = nothing
            paramjac_noise_config = []

            if DiffEqBase.isinplace(prob)
                for i in 1:numindvar
                    function noisetape(indx)
                        if StochasticDiffEq.is_diagonal_noise(prob)
                            ReverseDiff.GradientTape((y, _p, [tspan[2]])) do u, p, t
                                du1 = p !== nothing && p !== DiffEqBase.NullParameters() ?
                                      similar(p, size(u)) : similar(u)
                                unwrappedf(du1, u, p, first(t))
                                return du1[indx]
                            end
                        else
                            ReverseDiff.GradientTape((y, _p, [tspan[2]])) do u, p, t
                                du1 = similar(p, size(prob.noise_rate_prototype))
                                du1 .= false
                                unwrappedf(du1, u, p, first(t))
                                return du1[:, indx]
                            end
                        end
                    end
                    tapei = noisetape(i)
                    if compile_tape(sensealg.autojacvec)
                        push!(paramjac_noise_config, ReverseDiff.compile(tapei))
                    else
                        push!(paramjac_noise_config, tapei)
                    end
                end
            else
                for i in 1:numindvar
                    function noisetapeoop(indx)
                        if StochasticDiffEq.is_diagonal_noise(prob)
                            ReverseDiff.GradientTape((y, _p, [tspan[2]])) do u, p, t
                                unwrappedf(u, p, first(t))[indx]
                            end
                        else
                            ReverseDiff.GradientTape((y, _p, [tspan[2]])) do u, p, t
                                unwrappedf(u, p, first(t))[:, indx]
                            end
                        end
                    end
                    tapei = noisetapeoop(i)
                    if compile_tape(sensealg.autojacvec)
                        push!(paramjac_noise_config, ReverseDiff.compile(tapei))
                    else
                        push!(paramjac_noise_config, tapei)
                    end
                end
            end
        elseif sensealg.autojacvec isa Bool
            if DiffEqBase.isinplace(prob)
                if StochasticDiffEq.is_diagonal_noise(prob)
                    pf = DiffEqBase.ParamJacobianWrapper(unwrappedf, tspan[1], y)
                    if isnoisemixing(sensealg)
                        uf = DiffEqBase.UJacobianWrapper(unwrappedf, tspan[2], p)
                        jac_noise_config = build_jac_config(sensealg, uf, u0)
                    else
                        jac_noise_config = nothing
                    end
                else
                    pf = ParamNonDiagNoiseJacobianWrapper(unwrappedf, tspan[1], y,
                                                          prob.noise_rate_prototype)
                    uf = UNonDiagNoiseJacobianWrapper(unwrappedf, tspan[2], p,
                                                      prob.noise_rate_prototype)
                    jac_noise_config = build_jac_config(sensealg, uf, u0)
                end
                paramjac_noise_config = build_param_jac_config(sensealg, pf, y, p)
            else
                if StochasticDiffEq.is_diagonal_noise(prob)
                    pf = ParamGradientWrapper(unwrappedf, tspan[2], y)
                    if isnoisemixing(sensealg)
                        uf = DiffEqBase.UDerivativeWrapper(unwrappedf, tspan[2], p)
                    end
                else
                    pf = ParamNonDiagNoiseGradientWrapper(unwrappedf, tspan[1], y)
                    uf = UNonDiagNoiseGradientWrapper(unwrappedf, tspan[2], p)
                end
                paramjac_noise_config = nothing
                jac_noise_config = nothing
            end
            if StochasticDiffEq.is_diagonal_noise(prob)
                pJ = similar(u0, numindvar, numparams)
                if isnoisemixing(sensealg)
                    J = similar(u0, numindvar, numindvar)
                end
            else
                pJ = similar(u0, numindvar * numindvar, numparams)
                J = similar(u0, numindvar * numindvar, numindvar)
            end

        else
            paramjac_noise_config = nothing
            jac_noise_config = nothing
        end
    else
        paramjac_noise_config = nothing
        jac_noise_config = nothing
    end

    adjoint_cache = AdjointDiffCache(uf, pf, pg, J, pJ, dg_val,
                                     jac_config, pg_config, paramjac_config,
                                     jac_noise_config, paramjac_noise_config,
                                     f_cache, dgdu, dgdp, diffvar_idxs, algevar_idxs,
                                     factorized_mass_matrix, issemiexplicitdae)

    return adjoint_cache, y
end

function getprob(S::SensitivityFunction)
    (S isa ODEBacksolveSensitivityFunction) ? S.prob : S.sol.prob
end
inplace_sensitivity(S::SensitivityFunction) = isinplace(getprob(S))

struct ReverseLossCallback{λType, timeType, yType, RefType, FMType, AlgType, dg1Type,
                           dg2Type,
                           cacheType, solType}
    isq::Bool
    λ::λType
    t::timeType
    y::yType
    cur_time::RefType
    idx::Int
    F::FMType
    sensealg::AlgType
    dgdu::dg1Type
    dgdp::dg2Type
    diffcache::cacheType
    sol::solType
end

function ReverseLossCallback(sensefun, λ, t, dgdu, dgdp, cur_time)
    @unpack sensealg, y = sensefun
    isq = (sensealg isa QuadratureAdjoint)

    @unpack factorized_mass_matrix = sensefun.diffcache
    prob = getprob(sensefun)
    idx = length(prob.u0)
    if ArrayInterfaceCore.ismutable(y)
        return ReverseLossCallback(isq, λ, t, y, cur_time, idx, factorized_mass_matrix,
                                   sensealg, dgdu, dgdp, sensefun.diffcache, nothing)
    else
        return ReverseLossCallback(isq, λ, t, y, cur_time, idx, factorized_mass_matrix,
                                   sensealg, dgdu, dgdp, sensefun.diffcache, sensefun.sol)
    end
end

function (f::ReverseLossCallback)(integrator)
    @unpack isq, λ, t, y, cur_time, idx, F, sensealg, dgdu, dgdp, sol = f
    @unpack diffvar_idxs, algevar_idxs, issemiexplicitdae, J, uf, f_cache, jac_config = f.diffcache

    p, u = integrator.p, integrator.u

    if sensealg isa BacksolveAdjoint
        copyto!(y, integrator.u[(end - idx + 1):end])
    end

    if ArrayInterfaceCore.ismutable(u)
        # Warning: alias here! Be careful with λ
        gᵤ = isq ? λ : @view(λ[1:idx])
        if dgdu !== nothing
            dgdu(gᵤ, y, p, t[cur_time[]], cur_time[])
            # add discrete dgdp contribution
            if dgdp !== nothing && !isq
                gp = @view(λ[(idx + 1):end])
                dgdp(gp, y, p, t[cur_time[]], cur_time[])
                u[(idx + 1):length(λ)] .+= gp
            end
        end
    else
        @assert sensealg isa QuadratureAdjoint
        outtype = DiffEqBase.parameterless_type(λ)
        y = sol(t[cur_time[]])
        gᵤ = dgdu(y, p, t[cur_time[]], cur_time[]; outtype = outtype)
    end

    if issemiexplicitdae
        if J isa DiffCache
            J = get_tmp(J, y)
        end
        if DiffEqBase.has_jac(f)
            f.jac(J, y, p, t)
        else 
            jacobian!(J, uf, y, f_cache, sensealg, jac_config)
        end
        dhdd = J[algevar_idxs, diffvar_idxs]
        dhda = J[algevar_idxs, algevar_idxs]
        # TODO: maybe need a `conj`
        Δλa = -dhda' \ gᵤ[algevar_idxs]
        Δλd = dhdd'Δλa + gᵤ[diffvar_idxs]
    else
        Δλd = gᵤ
    end

    if F !== nothing
        F !== I && F !== (I, I) && ldiv!(F, Δλd)
    end

    if ArrayInterfaceCore.ismutable(u)
        u[diffvar_idxs] .+= Δλd
    else
        @assert sensealg isa QuadratureAdjoint
        integrator.u += Δλd
    end
    u_modified!(integrator, true)
    cur_time[] -= 1
    return nothing
end

# handle discrete loss contributions
function generate_callbacks(sensefun, dgdu, dgdp, λ, t, t0, callback, init_cb,
                            terminated = false)
    if sensefun isa NILSASSensitivityFunction
        @unpack sensealg = sensefun.S
    else
        @unpack sensealg = sensefun
    end

    if !init_cb
        cur_time = Ref(1)
    else
        cur_time = Ref(length(t))
    end

    reverse_cbs = setup_reverse_callbacks(callback, sensealg, dgdu, dgdp, cur_time,
                                          terminated)

    init_cb || return reverse_cbs, nothing

    # callbacks can lead to non-unique time points
    _t, duplicate_iterator_times = separate_nonunique(t)

    rlcb = ReverseLossCallback(sensefun, λ, t, dgdu, dgdp, cur_time)

    if eltype(_t) !== typeof(t0)
        _t = convert.(typeof(t0), _t)
    end
    cb = PresetTimeCallback(_t, rlcb)

    # handle duplicates (currently only for double occurances)
    if duplicate_iterator_times !== nothing
        # use same ref for cur_time to cope with concrete_solve
        cbrev_dupl_affect = ReverseLossCallback(sensefun, λ, t, dgdu, dgdp, cur_time)
        cb_dupl = PresetTimeCallback(duplicate_iterator_times[1], cbrev_dupl_affect)
        return CallbackSet(cb, reverse_cbs, cb_dupl), duplicate_iterator_times
    else
        return CallbackSet(cb, reverse_cbs), duplicate_iterator_times
    end
end

function separate_nonunique(t)
    # t is already sorted
    _t = unique(t)
    ts_with_occurances = [(i, count(==(i), t)) for i in _t]

    # duplicates (only those values which occur > 1 times)
    dupl = filter(x -> last(x) > 1, ts_with_occurances)

    ts = first.(dupl)
    occurances = last.(dupl)

    if isempty(occurances)
        itrs = nothing
    else
        maxoc = maximum(occurances)
        maxoc > 2 &&
            error("More than two occurances of the same time point. Please report this.")
        # handle also more than two occurances
        itrs = [ts[occurances .>= i] for i in 2:maxoc]
    end

    return _t, itrs
end

function out_and_ts(_ts, duplicate_iterator_times, sol)
    if duplicate_iterator_times === nothing
        ts = _ts
        out = sol(ts)
    else
        # if callbacks are tracked, there is potentially an event_time that must be considered
        # in the loss function but doesn't occur in saveat/t. So we need to add it.
        # Note that if it doens't occur in saveat/t we even need to add it twice
        # However if the callbacks are not saving in the forward, we don't want to compute a loss
        # value for them. This information is given by sol.t/checkpoints.
        # Additionally we need to store the left and the right limit, respectively.
        duplicate_times = duplicate_iterator_times[1] # just treat two occurances at the moment (see separate_nonunique above)
        _ts = Array(_ts)
        for d in duplicate_times
            (d ∉ _ts) && push!(_ts, d)
        end

        u1 = sol(_ts).u
        u2 = sol(duplicate_times, continuity = :right).u
        saveat = vcat(_ts, duplicate_times...)
        perm = sortperm(saveat)
        ts = saveat[perm]
        u = vcat(u1, u2)[perm]
        out = DiffEqArray(u, ts)
    end
    return out, ts
end
