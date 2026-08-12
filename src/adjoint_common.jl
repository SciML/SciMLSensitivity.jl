# Wraps a pre-allocated dense primal buffer for use with Enzyme.Duplicated when the
# parameter is a view-backed array (e.g. ComponentArray with SubArray data). Enzyme
# requires primal and shadow to have the same concrete type; since the shadow is always
# a fresh dense allocation, the primal must also be dense. We pre-allocate once here
# and copyto! the current parameter values at each adjoint step.
struct EnzymeViewPrimalBuffer{T}
    buf::T
end

struct AdjointDiffCache{
        UF, PF, G, TJ, PJT, uType, JC, GC, PJC, JNC, PJNC, rateType, DG1,
        DG2, DI,
        AI, FM, tType, rType, AI2, DI2,
    }
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
    tunables::tType
    repack::rType
    # Fully implicit DAE (`DAEProblem`) structure: indices of the algebraic and
    # differential equations (zero/nonzero rows of ∂F/∂(du)), which for
    # mass-matrix DAEs coincide with the variable indices, and whether the
    # system is Hessenberg index-2.
    algeeq_idxs::AI2
    diffeq_idxs::DI2
    isdae::Bool
    index2::Bool
end

"""
    adjointdiffcache(g,sensealg,discrete,sol,dg,alg;quad=false)

return (AdjointDiffCache, y)
"""
function adjointdiffcache(
        g::G, sensealg, discrete, sol, dgdu::DG1, dgdp::DG2, f, alg;
        quad = false,
        noiseterm = false, needs_jac = false, use_full_p = false
    ) where {G, DG1, DG2}
    prob = sol.prob
    u0 = state_values(prob)
    p = parameter_values(prob)
    if use_full_p && p !== nothing && !(p isa SciMLBase.NullParameters)
        # Use full parameter object (including caches) for VJP computation.
        # Required for SCCNonlinearProblem where explicitfuns! write active
        # data into non-tunable parameter components.
        tunables, repack = p, identity
    elseif p === nothing || p isa SciMLBase.NullParameters
        tunables, repack = p, identity
    elseif isscimlstructure(p)
        tunables, repack, _ = canonicalize(Tunable(), p)
    elseif isfunctor(p)
        if !supports_structured_vjp(sensealg.autojacvec)
            error(
                "$(typeof(sensealg.autojacvec)) does not support Functors.jl parameter structs. " *
                    "Use ZygoteVJP() instead."
            )
        end
        tunables, repack = Functors.functor(p)
    else
        throw(SciMLStructuresCompatibilityError())
    end
    if prob isa AbstractNonlinearProblem
        tspan = (nothing, nothing)
        #elseif prob isa SDEProblem
        #  (; tspan, u0, p) = prob
    else
        tspan = prob.tspan
    end

    isinplace = DiffEqBase.isinplace(prob)
    isRODE = prob isa RODEProblem
    isDAE = prob isa SciMLBase.AbstractDAEProblem
    autojacvec = sensealg.autojacvec

    if isDAE && !(
            autojacvec isa Union{ReverseDiffVJP, ZygoteVJP, EnzymeVJP, TrackerVJP} ||
                autojacvec isa Bool
        )
        error(
            "$(nameof(typeof(autojacvec))) is not currently supported for DAEProblem adjoints. " *
                "Use `ReverseDiffVJP()`, `EnzymeVJP()`, `ZygoteVJP()`, `TrackerVJP()`, or `autojacvec = false`."
        )
    end

    if isRODE
        _W = last(sol.W.W)
    else
        _W = nothing
    end

    if prob isa AbstractNonlinearProblem
        y = copy(state_values(sol))
    else
        y = copy(state_values(sol)[end])
    end

    if prob.p isa SciMLBase.NullParameters
        _p = similar(y, (0,))
        _p .= false
    else
        _p = tunables
    end

    _t = tspan[2]

    # Remove any function wrappers: it breaks autodiff. For DAE residuals also
    # unwrap the DAEFunction itself (see `dae_unwrapped_f`).
    unwrappedf = isDAE ? dae_unwrapped_f(f) : unwrapped_f(f)

    numparams = p === nothing || p === SciMLBase.NullParameters() ? 0 : length(tunables)
    numindvar = isnothing(u0) ? nothing : length(u0)
    isautojacvec = get_jacvec(sensealg)

    issemiexplicitdae = false
    index2 = false
    if isDAE
        # Fully implicit DAE residual F(du, u, p, t): classify the algebraic
        # structure from the Jacobians at the terminal time and
        # `differential_vars`, since there is no mass matrix to read it from.
        _dy_terminal = similar(y)
        sol(_dy_terminal, _t, Val{1})
        diffvar_idxs, algevar_idxs, diffeq_idxs, algeeq_idxs,
            index2 = dae_adjoint_structure(
            unwrappedf, isinplace, _dy_terminal, y, p, _t,
            prob.differential_vars
        )
        factorized_mass_matrix = nothing
    else
        mass_matrix = sol.prob.f.mass_matrix
        if mass_matrix isa UniformScaling
            factorized_mass_matrix = mass_matrix'
        elseif mass_matrix isa Tuple{UniformScaling, UniformScaling}
            factorized_mass_matrix = (I', I')
        else
            mass_matrix = mass_matrix'
            diffvar_idxs = findall(
                x -> any(!iszero, @view(mass_matrix[:, x])),
                axes(mass_matrix, 2)
            )
            algevar_idxs = setdiff(eachindex(u0), diffvar_idxs)

            # TODO: operator
            M̃ = @view mass_matrix[diffvar_idxs, diffvar_idxs]

            factorized_mass_matrix = lu(M̃, check = false)
            issuccess(factorized_mass_matrix) ||
                error("The submatrix corresponding to the differential variables of the mass matrix must be nonsingular!")
            isempty(algevar_idxs) || (issemiexplicitdae = true)
        end
        if !issemiexplicitdae
            diffvar_idxs = isnothing(u0) ? nothing : eachindex(u0)
            algevar_idxs = 1:0
        end
        # For mass-matrix problems, equation i and variable i pair up.
        algeeq_idxs = algevar_idxs
        diffeq_idxs = diffvar_idxs
    end

    if !needs_jac && !issemiexplicitdae && (!(autojacvec isa Bool) || isDAE)
        # The implicit-DAE paths compute their Jacobians on the fly from the
        # residual, so no dense J buffer is needed even for `autojacvec = false`.
        J = nothing
    else
        if alg === nothing || SciMLBase.forwarddiffs_model_time(alg)
            if !isnothing(u0)
                # 1 chunk is fine because it's only t
                _J = similar(u0, numindvar, numindvar)
                _J .= 0
                J = DiffCache(_J, ForwardDiff.pickchunksize(length(u0)))
            else
                J = nothing
            end
        else
            J = similar(u0, numindvar, numindvar)
            J .= 0
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
                if !isnothing(u0)
                    dg_val = similar(u0, numindvar) # number of funcs size
                    dg_val .= false
                else
                    dg_val = nothing
                end
            end
        else
            pgpu = UGradientWrapper(g, _t, p)
            pgpu_config = build_grad_config(sensealg, pgpu, u0, tunables)
            pgpp = ParamGradientWrapper(g, _t, u0)
            pgpp_config = build_grad_config(sensealg, pgpp, tunables, tunables)
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

    if SciMLBase.has_jac(f) || J === nothing
        jac_config = nothing
        uf = nothing
    else
        if isinplace
            if !isRODE
                uf = SciMLBase.UJacobianWrapper(unwrappedf, _t, p)
            else
                uf = RODEUJacobianWrapper(unwrappedf, _t, p, _W)
            end
            jac_config = build_jac_config(sensealg, uf, u0)
        else
            if !isRODE
                uf = SciMLBase.UDerivativeWrapper(unwrappedf, _t, p)
            else
                uf = RODEUDerivativeWrapper(unwrappedf, _t, p, _W)
            end
            jac_config = nothing
        end
    end

    @assert autojacvec !== nothing

    if autojacvec isa ReverseDiffVJP
        if prob isa AbstractNonlinearProblem
            if isinplace
                tape = ReverseDiff.GradientTape((y, _p)) do u, p
                    du1 = p !== nothing && p !== SciMLBase.NullParameters() ?
                        similar(p, size(u)) : similar(u)
                    du1 .= false
                    unwrappedf(du1, u, p, nothing)
                    return vec(du1)
                end
            else
                tape = ReverseDiff.GradientTape((y, _p)) do u, p
                    vec(unwrappedf(u, p, nothing))
                end
            end
            if compile_tape(sensealg.autojacvec)
                paramjac_config = ReverseDiff.compile(tape)
            else
                paramjac_config = tape
            end
        elseif noiseterm &&
                (!SciMLBase.is_diagonal_noise(prob) || isnoisemixing(sensealg))
            tape = nothing
            paramjac_config = tape
        elseif isDAE
            paramjac_config = get_dae_paramjac_config(
                autojacvec, p, unwrappedf, y, _dy_terminal, _p, _t; isinplace
            )
        else
            paramjac_config = get_paramjac_config(
                autojacvec, p, unwrappedf, y, _p, _t;
                isinplace, isRODE, _W
            )
        end

        pf = nothing
    elseif autojacvec isa EnzymeVJP
        paramjac_config = get_paramjac_config(autojacvec, p, f, y, _p, _t; numindvar, alg)
        pf = get_pf(autojacvec; _f = unwrappedf, isinplace, isRODE)
        if isDAE
            # The residual takes `du` as an extra differentiated argument; append
            # its primal and shadow buffers to the standard Enzyme config.
            paramjac_config = if paramjac_config[1] isa LazyBufferCache
                (paramjac_config..., LazyBufferCache(), LazyBufferCache())
            else
                (paramjac_config..., zero(y), zero(y))
            end
        end
        _needs_shadow = !(p isa SciMLBase.NullParameters) &&
            isscimlstructure(p) && !(p isa AbstractArray)
        _shadow_p = if _needs_shadow
            # `repack` only writes the tunable fields and copies the rest of the
            # parameter object (e.g. `initials`, `discrete`) *by reference* from
            # `p`, so `repack(zero(tunables))` would alias the primal's non-tunable
            # arrays. `_vecjacobian!` then `Enzyme.remake_zero!`s this shadow, which
            # would zero the user's `p.initials` and corrupt the problem across
            # repeated differentiation (#1470). `make_zero(p)` allocates a fully
            # disjoint zeroed shadow; `canonicalize(Tunable(), _shadow_p)` still
            # reads back the accumulated tunable gradient.
            Enzyme.make_zero(p)
        elseif !(p isa SciMLBase.NullParameters) && p isa AbstractArray &&
                typeof(tunables) !== typeof(zero(tunables))
            # tunables is a view-backed array (e.g. ComponentArray with SubArray data).
            # zero(tunables) returns a dense array of a different concrete type, so
            # Enzyme.Duplicated(p, tmp2) would throw a type mismatch. Pre-allocate a
            # dense primal buffer here; at each adjoint step we copyto! the current p
            # values and pass this buffer as the Enzyme primal instead of p itself.
            EnzymeViewPrimalBuffer(copy(tunables))
        else
            nothing
        end
        paramjac_config = (paramjac_config..., Enzyme.make_zero(pf), _shadow_p)
    elseif autojacvec isa MooncakeVJP
        _pf = get_pf(autojacvec, prob, unwrappedf)
        # Wrap pf to accept flat tunables and repack to full parameter struct,
        # matching how ReverseDiff handles structured parameters.
        _needs_repack = isscimlstructure(p) && !(p isa AbstractArray)
        pf = if _needs_repack
            if isRODE
                let _pf = _pf, repack = repack
                    (out, u, _tunables, t, W) -> _pf(out, u, repack(_tunables), t, W)
                end
            else
                let _pf = _pf, repack = repack
                    (out, u, _tunables, t) -> _pf(out, u, repack(_tunables), t)
                end
            end
        else
            _pf
        end
        paramjac_config = get_paramjac_config(
            MooncakeLoaded(), autojacvec, pf, tunables, f, y, _t
        )
    elseif autojacvec isa ReactantVJP
        pf = get_pf(autojacvec, prob, unwrappedf)
        paramjac_config = get_paramjac_config(
            ReactantLoaded(), autojacvec, pf, p, f, y, _t;
            numindvar = numindvar, alg = alg
        )
    elseif SciMLBase.has_paramjac(f) || quad || !(autojacvec isa Bool) ||
            autojacvec isa EnzymeVJP || isDAE
        paramjac_config = nothing
        pf = nothing
    else
        _needs_repack = (isscimlstructure(p) && !(p isa AbstractArray)) || isfunctor(p)
        if isinplace &&
                !(p === nothing || p === SciMLBase.NullParameters())
            if !isRODE
                _pjac_f = _needs_repack ?
                    (du, u, p, t) -> unwrappedf(du, u, repack(p), t) :
                    unwrappedf
                pf = SciMLBase.ParamJacobianWrapper(_pjac_f, _t, y)
            else
                pf = RODEParamJacobianWrapper(unwrappedf, _t, y, _W)
            end
            paramjac_config = build_param_jac_config(sensealg, pf, y, tunables)
        else
            if !isRODE
                _pgrad_f = _needs_repack ?
                    (u, p, t) -> unwrappedf(u, repack(p), t) :
                    unwrappedf
                pf = ParamGradientWrapper(_pgrad_f, _t, y)
            else
                pf = RODEParamGradientWrapper(unwrappedf, _t, y, _W)
            end
            paramjac_config = nothing
        end
    end

    pJ = if (quad || !(autojacvec isa Bool) || isDAE)
        nothing
    else
        if !isnothing(u0)
            _pJ = similar(u0, numindvar, numparams)
            _pJ .= false
        else
            _pJ = nothing
        end
    end

    f_cache = isinplace ? deepcopy(u0) : nothing

    if noiseterm
        if autojacvec isa ReverseDiffVJP
            jac_noise_config = nothing
            paramjac_noise_config = []
            noise_rate_prototype = prob.noise_rate_prototype
            # number of Wiener processes
            m = noise_rate_prototype === nothing ? numindvar : size(noise_rate_prototype)[2]
            if isinplace
                for i in 1:m
                    function noisetape(idx)
                        return if SciMLBase.is_diagonal_noise(prob)
                            ReverseDiff.GradientTape((y, _p, [_t])) do u, p, t
                                du1 = p !== nothing && p !== SciMLBase.NullParameters() ?
                                    similar(p, size(u)) : similar(u)
                                copyto!(du1, false)
                                unwrappedf(du1, u, p, first(t))
                                return du1[idx]
                            end
                        else
                            ReverseDiff.GradientTape((y, _p, [_t])) do u, p, t
                                du1 = similar(p, size(noise_rate_prototype))
                                du1 .= false
                                unwrappedf(du1, u, p, first(t))
                                return du1[:, idx]
                            end
                        end
                    end
                    tapei = noisetape(i)
                    if compile_tape(autojacvec)
                        push!(paramjac_noise_config, ReverseDiff.compile(tapei))
                    else
                        push!(paramjac_noise_config, tapei)
                    end
                end
            else
                for i in 1:m
                    function noisetapeoop(idx)
                        return if SciMLBase.is_diagonal_noise(prob)
                            ReverseDiff.GradientTape((y, _p, [_t])) do u, p, t
                                unwrappedf(u, p, first(t))[idx]
                            end
                        else
                            ReverseDiff.GradientTape((y, _p, [_t])) do u, p, t
                                unwrappedf(u, p, first(t))[:, idx]
                            end
                        end
                    end
                    tapei = noisetapeoop(i)
                    if compile_tape(autojacvec)
                        push!(paramjac_noise_config, ReverseDiff.compile(tapei))
                    else
                        push!(paramjac_noise_config, tapei)
                    end
                end
            end
        elseif autojacvec isa EnzymeVJP
            # Cache the reusable buffers so `_jacNoise!(::EnzymeVJP)` allocates
            # nothing per reverse-solver step (mirrors the drift `_vecjacobian!`).
            jac_noise_config = nothing
            noise_rate_prototype = prob.noise_rate_prototype
            _diag = SciMLBase.is_diagonal_noise(prob)
            # Output-shaped buffers (state-shaped for diagonal noise, matrix-shaped
            # for non-diagonal), plus the state-shaped primal/shadow.
            _du_out = _diag ? zero(y) : zero(noise_rate_prototype)
            _du_seed = _diag ? zero(y) : zero(noise_rate_prototype)
            _u_buf = zero(y)
            _du_u = zero(y)
            _pf = get_pf(autojacvec; _f = unwrappedf, isinplace, isRODE)
            _func_shadow = Enzyme.make_zero(_pf)
            # Dense parameter-gradient shadow (used directly for plain-array and
            # view-backed `p`).
            _dp_dense = zero(_p)
            # #1470-safe parameter shadow: for a SciMLStructures (non-array) `p` the
            # shadow must be a fully disjoint `make_zero(p)` (NOT
            # `repack(zero(tunables))`, which aliases `p`'s non-tunable arrays);
            # re-zeroing a cached aliasing shadow would corrupt the user's `p`.
            _needs_shadow = !(p isa SciMLBase.NullParameters) &&
                isscimlstructure(p) && !(p isa AbstractArray)
            _shadow_p = if _needs_shadow
                Enzyme.make_zero(p)
            elseif !(p isa SciMLBase.NullParameters) && p isa AbstractArray &&
                    typeof(tunables) !== typeof(zero(tunables))
                # view-backed tunables (e.g. ComponentArray with SubArray data):
                # pre-allocate a dense primal buffer to match the dense shadow type.
                EnzymeViewPrimalBuffer(copy(tunables))
            else
                nothing
            end
            paramjac_noise_config = (
                du_out = _du_out, du_seed = _du_seed,
                u_buf = _u_buf, du_u = _du_u,
                func_shadow = _func_shadow, dp_dense = _dp_dense,
                shadow_p = _shadow_p,
            )
        elseif autojacvec isa Bool
            if isinplace
                if SciMLBase.is_diagonal_noise(prob)
                    pf = SciMLBase.ParamJacobianWrapper(unwrappedf, _t, y)
                    if isnoisemixing(sensealg)
                        uf = SciMLBase.UJacobianWrapper(unwrappedf, _t, p)
                        jac_noise_config = build_jac_config(sensealg, uf, u0)
                    else
                        jac_noise_config = nothing
                    end
                else
                    pf = ParamNonDiagNoiseJacobianWrapper(
                        unwrappedf, _t, y,
                        prob.noise_rate_prototype
                    )
                    uf = UNonDiagNoiseJacobianWrapper(
                        unwrappedf, _t, p,
                        prob.noise_rate_prototype
                    )
                    jac_noise_config = build_jac_config(sensealg, uf, u0)
                end
                paramjac_noise_config = build_param_jac_config(
                    sensealg, pf, y, SciMLStructures.replace(Tunable(), p, tunables)
                )
            else
                if SciMLBase.is_diagonal_noise(prob)
                    pf = ParamGradientWrapper(unwrappedf, _t, y)
                    if isnoisemixing(sensealg)
                        uf = SciMLBase.UDerivativeWrapper(unwrappedf, _t, p)
                    end
                else
                    pf = ParamNonDiagNoiseGradientWrapper(unwrappedf, _t, y)
                    uf = UNonDiagNoiseGradientWrapper(unwrappedf, _t, p)
                end
                paramjac_noise_config = nothing
                jac_noise_config = nothing
            end
            if SciMLBase.is_diagonal_noise(prob)
                pJ = similar(u0, numindvar, numparams)
                if isnoisemixing(sensealg)
                    J = similar(u0, numindvar, numindvar)
                end
                pJ .= false
                J .= false
            else
                pJ = similar(u0, numindvar * numindvar, numparams)
                J = similar(u0, numindvar * numindvar, numindvar)
                pJ .= false
                J .= false
            end

        else
            paramjac_noise_config = nothing
            jac_noise_config = nothing
        end
    else
        paramjac_noise_config = nothing
        jac_noise_config = nothing
    end

    adjoint_cache = AdjointDiffCache(
        uf, pf, pg, J, pJ, dg_val,
        jac_config, pg_config, paramjac_config,
        jac_noise_config, paramjac_noise_config,
        f_cache, dgdu, dgdp, diffvar_idxs, algevar_idxs,
        factorized_mass_matrix, issemiexplicitdae,
        tunables, repack,
        algeeq_idxs, diffeq_idxs, isDAE, index2
    )

    return adjoint_cache, y
end

# ReverseDiff tape for the fully implicit DAE residual F(du, u, p, t), taped over
# all four arguments so a single reverse pass yields the vjps with respect to
# `du`, `u`, and `p`.
function get_dae_paramjac_config(
        autojacvec::ReverseDiffVJP, p, f, y, dy, _p, _t;
        isinplace = true
    )
    if p === nothing || p isa SciMLBase.NullParameters
        tunables, repack = p, identity
    elseif isscimlstructure(p)
        tunables, repack, _ = canonicalize(Tunable(), p)
    else
        tunables, repack = p, identity
    end
    tape = if isinplace
        ReverseDiff.GradientTape((dy, y, _p, [_t])) do du, u, p, t
            res = (p !== nothing && p !== SciMLBase.NullParameters()) ?
                similar(p, size(u)) : similar(u)
            res .= false
            f(res, du, u, repack(p), first(t))
            return vec(res)
        end
    else
        ReverseDiff.GradientTape((dy, y, _p, [_t])) do du, u, p, t
            vec(f(du, u, repack(p), first(t)))
        end
    end
    if compile_tape(autojacvec)
        return ReverseDiff.compile(tape)
    else
        return tape
    end
end

function get_paramjac_config(
        autojacvec::ReverseDiffVJP, p, f, y, _p, _t;
        numindvar = nothing, alg = nothing, isinplace = true,
        isRODE = false, _W = nothing
    )
    # f = unwrappedf
    if p === nothing || p isa SciMLBase.NullParameters
        tunables, repack = p, identity
    elseif isscimlstructure(p)
        tunables, repack, _ = canonicalize(Tunable(), p)
    elseif isfunctor(p)
        error(
            "ReverseDiffVJP does not support Functors.jl parameter structs. " *
                "Use ZygoteVJP() instead or make `p` a SciMLStructure. See SciMLStructures.jl."
        )
    else
        tunables, repack = p, identity
    end
    if isinplace
        if !isRODE
            tape = ReverseDiff.GradientTape((y, _p, [_t])) do u, p, t
                du1 = (p !== nothing && p !== SciMLBase.NullParameters()) ?
                    similar(p, size(u)) : similar(u)
                du1 .= false
                f(du1, u, repack(p), first(t))
                return vec(du1)
            end
        else
            tape = ReverseDiff.GradientTape((y, _p, [_t], _W)) do u, p, t, W
                du1 = p !== nothing && p !== SciMLBase.NullParameters() ?
                    similar(p, size(u)) : similar(u)
                du1 .= false
                f(du1, u, p, first(t), W)
                return vec(du1)
            end
        end
    else
        if !isRODE
            tape = ReverseDiff.GradientTape((y, _p, [_t])) do u, p, t
                vec(f(u, repack(p), first(t)))
            end
        else
            tape = ReverseDiff.GradientTape((y, _p, [_t], _W)) do u, p, t, W
                return f(u, p, first(t), W)
            end
        end
    end

    if compile_tape(autojacvec)
        paramjac_config = ReverseDiff.compile(tape)
    else
        paramjac_config = tape
    end

    return paramjac_config
end

function get_paramjac_config(
        autojacvec::EnzymeVJP, p::SciMLBase.NullParameters, f, y, _p,
        _t;
        numindvar, alg, isinplace = nothing, isRODE = nothing,
        _W = nothing
    )
    if alg !== nothing && SciMLBase.forwarddiffs_model(alg)
        chunk = if autojacvec.chunksize == 0
            ForwardDiff.pickchunksize(numindvar)
        else
            autojacvec.chunksize
        end

        paramjac_config = LazyBufferCache(), p,
            LazyBufferCache(),
            LazyBufferCache(),
            LazyBufferCache()
    else
        paramjac_config = zero(y), p, zero(y), zero(y), zero(y)
    end
    return paramjac_config
end

function get_paramjac_config(
        autojacvec::EnzymeVJP, p, f, y, _p, _t; numindvar, alg,
        isinplace = nothing,
        isRODE = nothing, _W = nothing
    )
    if alg !== nothing && SciMLBase.forwarddiffs_model(alg)
        chunk = if autojacvec.chunksize == 0
            ForwardDiff.pickchunksize(numindvar)
        else
            autojacvec.chunksize
        end

        paramjac_config = LazyBufferCache(), zero(_p),
            LazyBufferCache(),
            LazyBufferCache(),
            LazyBufferCache()
    else
        paramjac_config = zero(y), zero(_p), zero(y), zero(y), zero(y)
    end
    return paramjac_config
end

# Dispatched on inside extension.
struct MooncakeLoaded end
struct ReactantLoaded end
"""
    ReactantDualTag

Tag type used by the Reactant extension to distinguish dual-number compilation
paths from floating-point compilation paths.

This is developer API for `SciMLSensitivityReactantExt` and related extension
code.
"""
struct ReactantDualTag end

"""
    ReactantVJPConfig

Compiled-kernel cache used by `ReactantVJP`.

## Fields

  - `float_kernel`, `dual_kernel`: Reactant-compiled kernels for ordinary and
    dual-number execution.
  - `float_caches`, `dual_caches`: buffers reused by the corresponding kernels.
  - `is_nullparams`: whether the differentiated problem has null parameters.
  - `chunk_size`: ForwardDiff chunk size used when dual execution is required.

This is developer API for the Reactant extension; users normally configure it
through `ReactantVJP`.
"""
struct ReactantVJPConfig{FK, DK, FC, DC}
    float_kernel::FK
    dual_kernel::DK
    float_caches::FC
    dual_caches::DC
    is_nullparams::Bool
    chunk_size::Int
end

function get_paramjac_config(::Any, ::MooncakeVJP, pf, p, f, y, _t)
    msg = "MooncakeVJP requires Mooncake.jl is loaded. Install the package and do " * "`using Mooncake` to use this functionality"
    error(msg)
end

function get_paramjac_config(
        ::Any, ::ReactantVJP, pf, p, f, y, _t;
        numindvar = nothing, alg = nothing
    )
    msg = "ReactantVJP requires Reactant.jl is loaded. Install the package and do " *
        "`using Reactant` to use this functionality"
    error(msg)
end

function get_pf(
        autojacvec::ReverseDiffVJP; _f = nothing, isinplace = nothing,
        isRODE = nothing
    )
    return nothing
end

function get_pf(autojacvec::EnzymeVJP; _f, isinplace, isRODE)
    return isinplace ? SciMLBase.Void(_f) : _f
end

function get_pf(::MooncakeVJP, prob, _f)
    isinplace = DiffEqBase.isinplace(prob)
    isRODE = isa(prob, RODEProblem)
    return pf = let f = _f
        if isinplace && isRODE
            function (out, u, _p, t, W)
                f(out, u, _p, t, W)
                return out
            end
        elseif isinplace
            function (out, u, _p, t)
                f(out, u, _p, t)
                return out
            end
        elseif !isinplace && isRODE
            function (out, u, _p, t, W)
                out .= f(u, _p, t, W)
                return out
            end
        else
            # !isinplace
            function (out, u, _p, t)
                out .= f(u, _p, t)
                return out
            end
        end
    end
end

function mooncake_run_ad(paramjac_config, y, p, t, λ)
    msg = "MooncakeVJP requires Mooncake.jl is loaded. Install the package and do " * "`using Mooncake` to use this functionality"
    error(msg)
end

function get_pf(::ReactantVJP, prob, _f)
    isinplace = DiffEqBase.isinplace(prob)
    isRODE = isa(prob, RODEProblem)
    return pf = let f = _f
        if isinplace && isRODE
            function (out, u, _p, t, W)
                f(out, u, _p, t, W)
                return out
            end
        elseif isinplace
            function (out, u, _p, t)
                f(out, u, _p, t)
                return out
            end
        elseif !isinplace && isRODE
            function (out, u, _p, t, W)
                out .= f(u, _p, t, W)
                return out
            end
        else
            function (out, u, _p, t)
                out .= f(u, _p, t)
                return out
            end
        end
    end
end

function reactant_run_ad!(dλ, dgrad, dy, paramjac_config, y, p, t, λ)
    msg = "ReactantVJP requires Reactant.jl is loaded. Install the package and do " *
        "`using Reactant` to use this functionality"
    error(msg)
end

function reactant_run_dual_ad!(dλ, dgrad, dy, config, y, p, t, λ)
    msg = "ReactantVJP requires Reactant.jl is loaded. Install the package and do " *
        "`using Reactant` to use this functionality"
    error(msg)
end

function reactant_run_cb_ad!(dλ, dgrad, dy, paramjac_config, y, p, t, tprev, λ)
    msg = "ReactantVJP requires Reactant.jl is loaded. Install the package and do " *
        "`using Reactant` to use this functionality"
    error(msg)
end

function get_cb_paramjac_config(::Any, ::ReactantVJP, raw_affect, event_idx, y, p, _t, mode)
    msg = "ReactantVJP requires Reactant.jl is loaded. Install the package and do " *
        "`using Reactant` to use this functionality"
    error(msg)
end

function get_cb_paramjac_config(::Any, ::MooncakeVJP, raw_affect, event_idx, y, p, _t, mode)
    msg = "MooncakeVJP requires Mooncake.jl is loaded. Install the package and do " *
        "`using Mooncake` to use this functionality"
    error(msg)
end

function getprob(S::SensitivityFunction)
    return (S isa ODEBacksolveSensitivityFunction) ? S.prob : S.sol.prob
end
inplace_sensitivity(S::SensitivityFunction) = isinplace(getprob(S))

"""
    build_adjoint_jac(sol, sensealg, alg, f, tspan)

Construct a `jac(J_adj, λ, p, t)` for the adjoint ODE of the state-only adjoint
methods (`GaussAdjoint`, `QuadratureAdjoint`, whose adjoint state is just `λ`).
The adjoint rhs `-(∂f/∂u)ᵀλ` is linear in `λ`, so its Jacobian is exactly
`-(∂f/∂u)ᵀ` evaluated at the interpolated forward solution — independent of `λ`.

When the user supplies an analytical `jac` (with a `jac_prototype`), evaluate it
at `y(t)` and negate-transpose. Otherwise, for dense in-place ODEs solved by a
ForwardDiff-based implicit method, fall back to differentiating the *forward*
rhs (ForwardDiff/FiniteDiff per `sensealg`'s `autodiff` setting) and
negate-transposing. Without this, the implicit adjoint solver differentiates the
vjp-based adjoint rhs itself, which costs a full vjp (plus a forward-solution
interpolation) per Jacobian column — and with `ReverseDiffVJP` re-records the
tape with `Dual` numbers on every Jacobian evaluation.

Returns `nothing` when no cheap exact Jacobian can be built; the adjoint solver
then falls back to its own AD/FD of the adjoint rhs.
"""
function build_adjoint_jac(sol, sensealg, alg, f, tspan)
    jac_prototype = sol.prob.f.jac_prototype
    u0 = state_values(sol.prob)
    if SciMLBase.has_jac(sol.prob.f) && jac_prototype !== nothing
        _fwd_jac_cache = copy(jac_prototype)
        _fwd_jac_fn = sol.prob.f.jac
        _fwd_sol = sol
        if jac_prototype isa SparseArrays.AbstractSparseMatrixCSC
            return function (J_adj, _λ, _p, _t)
                _y = _fwd_sol(_t, continuity = :right)
                _fwd_jac_fn(_fwd_jac_cache, _y, _p, _t)
                SparseArrays.ftranspose!(J_adj, _fwd_jac_cache, -)
                return nothing
            end
        else
            return function (J_adj, _λ, _p, _t)
                _y = _fwd_sol(_t, continuity = :right)
                _fwd_jac_fn(_fwd_jac_cache, _y, _p, _t)
                transpose!(J_adj, _fwd_jac_cache)
                lmul!(-1, J_adj)
                return nothing
            end
        end
    elseif jac_prototype === nothing && alg !== nothing &&
            SciMLBase.forwarddiffs_model(alg) &&
            SciMLBase.isinplace(sol.prob) && u0 isa AbstractArray &&
            ArrayInterface.ismutable(u0) && eltype(u0) <: AbstractFloat
        # `forwarddiffs_model(alg)` implies the adjoint solver would push `Dual`
        # numbers through the vjp-based adjoint rhs anyway, so requiring the
        # forward rhs to be ForwardDiff-compatible here does not narrow support.
        _fwd_sol = sol
        _f = unwrapped_f(f)
        numindvar = length(u0)
        _J_cache = similar(u0, numindvar, numindvar)
        _J_cache .= false
        _uf = SciMLBase.UJacobianWrapper(_f, tspan[2], parameter_values(sol.prob))
        _jac_config = build_jac_config(sensealg, _uf, u0)
        _y_cache = similar(u0)
        _fx_cache = similar(u0)
        return function (J_adj, _λ, _p, _t)
            _fwd_sol(_y_cache, _t, continuity = :right)
            _uf.t = _t
            _uf.p = _p
            jacobian!(_J_cache, _uf, _y_cache, _fx_cache, sensealg, _jac_config)
            transpose!(J_adj, _J_cache)
            lmul!(-1, J_adj)
            return nothing
        end
    else
        return nothing
    end
end

struct ReverseLossCallback{
        λType, timeType, yType, RefType, FMType, AlgType, dg1Type,
        dg2Type,
        cacheType, fType, solType, ΔλasType,
    }
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
    f::fType
    sol::solType
    Δλas::ΔλasType
    no_start::Bool
end

function ReverseLossCallback(sensefun, λ, t, dgdu, dgdp, cur_time, no_start)
    (; sensealg, y) = sensefun
    isq = (sensealg isa QuadratureAdjoint) || (sensealg isa AbstractGAdjoint)

    (; factorized_mass_matrix, isdae) = sensefun.diffcache
    prob = getprob(sensefun)
    idx = length(state_values(prob))
    Δλas = Tuple{typeof(λ), eltype(t)}[]
    if ArrayInterface.ismutable(y) && !isdae
        return ReverseLossCallback(
            isq, λ, t, y, cur_time, idx, factorized_mass_matrix,
            sensealg, dgdu, dgdp, sensefun.diffcache, sensefun.f,
            nothing, Δλas, no_start
        )
    else
        # The DAE jump corrections interpolate the forward solution (and its
        # derivative) at the jump time, so keep `sol` around.
        return ReverseLossCallback(
            isq, λ, t, y, cur_time, idx, factorized_mass_matrix,
            sensealg, dgdu, dgdp, sensefun.diffcache, sensefun.f,
            sensefun.sol, Δλas, no_start
        )
    end
end

function (f::ReverseLossCallback)(integrator)
    (; isq, λ, t, y, cur_time, idx, F, sensealg, dgdu, dgdp, sol, no_start) = f
    (;
        diffvar_idxs, algevar_idxs, issemiexplicitdae,
        J, uf, f_cache, jac_config,
        algeeq_idxs, diffeq_idxs, isdae, index2,
    ) = f.diffcache

    no_start && !(sensealg isa BacksolveAdjoint) && cur_time[] == 1 && return nothing

    p, u = integrator.p, integrator.u
    ti = t[cur_time[]]

    if sensealg isa BacksolveAdjoint
        copyto!(y, integrator.u[(end - idx + 1):end])
    end

    # For implicit DAEs the shared `y` buffer holds the last rhs evaluation
    # point, so interpolate the forward solution at the jump time instead.
    isdae && sol(y, ti)

    if ArrayInterface.ismutable(u)
        # Warning: alias here! Be careful with λ
        gᵤ = isq ? λ : @view(λ[1:idx])
        if dgdu !== nothing
            dgdu(gᵤ, y, p, ti, cur_time[])
            # add discrete dgdp contribution; for the DAE adjoint the parameter
            # gradient block sits behind both the w and λ blocks
            if dgdp !== nothing && !isq
                poff = isdae ? 2 * idx : idx
                gp = @view(λ[(idx + 1):end])
                dgdp(gp, y, p, ti, cur_time[])
                u[(poff + 1):(poff + length(gp))] .+= gp
            end
        end
    else
        # Immutable (e.g. SVector) state: only valid for adjoints whose state is
        # exactly λ with no parameter-gradient augmentation.
        @assert isq
        outtype = ArrayInterface.parameterless_type(λ)
        y = sol(ti)
        gᵤ = dgdu(y, p, ti, cur_time[]; outtype)
    end

    if issemiexplicitdae || (isdae && !isempty(algevar_idxs))
        if isdae
            # ∂F/∂u of the residual at the interpolated (du, u); the shared
            # Jacobian machinery is bypassed since the residual signature differs.
            dy = sol(ti, Val{1})
            residual = unwrapped_f(f.f)
            J_u = dae_u_jacobian(residual, DiffEqBase.isinplace(sol.prob), dy, y, p, ti)
        else
            if J isa DiffCache
                J = get_tmp(J, y)
            end
            if SciMLBase.has_jac(f.f)
                f.f.jac(J, y, p, ti)
            else
                jacobian!(J, uf, y, f_cache, sensealg, jac_config)
            end
            J_u = J
        end
        dhdd = J_u[algeeq_idxs, diffvar_idxs]
        if isdae && index2
            all(iszero, @view(gᵤ[algevar_idxs])) ||
                error("Discrete cost contributions on the algebraic variables of an index-2 DAE are not supported: they correspond to derivatives of delta distributions in the adjoint. Reformulate the cost in terms of the differential variables.")
            dy = sol(ti, Val{1})
            residual = unwrapped_f(f.f)
            J_du = dae_du_jacobian(residual, DiffEqBase.isinplace(sol.prob), dy, y, p, ti)
            N = (lu(J_du[diffeq_idxs, diffvar_idxs]) \ J_u[diffeq_idxs, algevar_idxs])'
            Δλa = -((N * dhdd') \ (N * gᵤ[diffvar_idxs]))
        else
            dhda = J_u[algeeq_idxs, algevar_idxs]
            Δλa = -(dhda' \ gᵤ[algevar_idxs])
        end
        Δλd = dhdd'Δλa + gᵤ[diffvar_idxs]
        push!(f.Δλas, (Δλa, ti))
    else
        Δλd = isdae ? gᵤ[diffvar_idxs] : gᵤ
    end

    if F !== nothing
        F !== I && F !== (I, I) && ldiv!(F, Δλd)
    end

    if ArrayInterface.ismutable(u)
        u[diffvar_idxs] .+= Δλd
    else
        @assert isq
        integrator.u += Δλd
    end
    derivative_discontinuity!(integrator, true)
    cur_time[] -= 1
    return nothing
end

# handle discrete loss contributions
function generate_callbacks(
        sensefun, dgdu, dgdp, λ, t, t0, callback, init_cb,
        terminated = false, no_start = false
    )
    if sensefun isa NILSASSensitivityFunction
        (; sensealg) = sensefun.S
    else
        (; sensealg) = sensefun
    end

    if !init_cb
        cur_time = Ref(1)
    else
        cur_time = Ref(length(t))
    end

    reverse_cbs = setup_reverse_callbacks(
        callback, sensealg, dgdu, dgdp, cur_time,
        terminated
    )

    init_cb || return reverse_cbs, nothing, nothing

    # callbacks can lead to non-unique time points
    _t, duplicate_iterator_times = separate_nonunique(t)

    rlcb = ReverseLossCallback(sensefun, λ, t, dgdu, dgdp, cur_time, no_start)

    if eltype(_t) !== typeof(t0)
        _t = convert.(typeof(t0), _t)
    end
    cb = PresetTimeCallback(_t, rlcb)

    # handle duplicates (currently only for double occurrences)
    if duplicate_iterator_times !== nothing
        # use same ref for cur_time to cope with concrete_solve
        cbrev_dupl_affect = ReverseLossCallback(
            sensefun, λ, t, dgdu, dgdp, cur_time, no_start
        )
        cb_dupl = PresetTimeCallback(duplicate_iterator_times[1], cbrev_dupl_affect)
        return CallbackSet(cb, reverse_cbs, cb_dupl), rlcb, duplicate_iterator_times
    else
        return CallbackSet(cb, reverse_cbs), rlcb, duplicate_iterator_times
    end
end

function separate_nonunique(t)
    # t is already sorted
    _t = unique(t)
    ts_with_occurrences = [(i, count(==(i), t)) for i in _t]

    # duplicates (only those values which occur > 1 times)
    dupl = filter(x -> last(x) > 1, ts_with_occurrences)

    ts = first.(dupl)
    occurrences = last.(dupl)

    if isempty(occurrences)
        itrs = nothing
    else
        maxoc = maximum(occurrences)
        maxoc > 2 &&
            error("More than two occurrences of the same time point. Please report this.")
        # handle also more than two occurrences
        itrs = [ts[occurrences .>= i] for i in 2:maxoc]
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
        # Note that if it doesn't occur in saveat/t we even need to add it twice
        # However if the callbacks are not saving in the forward, we don't want to compute a loss
        # value for them. This information is given by sol.t/checkpoints.
        # Additionally we need to store the left and the right limit, respectively.
        duplicate_times = duplicate_iterator_times[1] # just treat two occurrences at the moment (see separate_nonunique above)
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
