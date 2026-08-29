struct ODEQuadratureAdjointSensitivityFunction{
        C <: AdjointDiffCache,
        Alg <: QuadratureAdjoint,
        uType, SType,
        fType <: AbstractDiffEqFunction,
    } <: SensitivityFunction
    diffcache::C
    sensealg::Alg
    discrete::Bool
    y::uType
    sol::SType
    f::fType
end

function ODEQuadratureAdjointSensitivityFunction(
        g, sensealg, discrete, sol, dgdu, dgdp,
        alg
    )
    p = sol.prob.p
    _use_full_p = hasproperty(sensealg, :diff_tunables) &&
        sensealg.diff_tunables isa Val{false} &&
        isscimlstructure(p) && !(p isa AbstractArray)
    diffcache,
        y = adjointdiffcache(
        g, sensealg, discrete, sol, dgdu, dgdp, sol.prob.f, alg;
        quad = true, use_full_p = _use_full_p
    )
    return ODEQuadratureAdjointSensitivityFunction(
        diffcache, sensealg, discrete,
        y, sol, sol.prob.f
    )
end

# u = λ'
function (S::ODEQuadratureAdjointSensitivityFunction)(du, u, p, t)
    (; sol, discrete) = S
    f = sol.prob.f

    λ, grad, y, dλ, dgrad, dy = split_states(du, u, t, S)

    vecjacobian!(dλ, y, λ, p, t, S)
    dλ .*= -one(eltype(λ))

    discrete || accumulate_cost!(dλ, y, p, t, S)
    return nothing
end

function (S::ODEQuadratureAdjointSensitivityFunction)(u, p, t)
    (; sol, discrete) = S
    f = sol.prob.f

    λ, grad, y, dgrad, dy = split_states(u, t, S)

    dy, dλ, dgrad = vecjacobian(y, λ, p, t, S; dgrad, dy)
    dλ *= (-one(eltype(λ)))

    if !discrete
        dλ, dgrad = accumulate_cost(dλ, y, p, t, S, dgrad)
    end
    return dλ
end

function split_states(du, u, t, S::ODEQuadratureAdjointSensitivityFunction; update = true)
    (; y, sol) = S

    if update
        if t isa ForwardDiff.Dual && eltype(y) <: AbstractFloat
            y = sol(t, continuity = :right)
        else
            sol(y, t, continuity = :right)
        end
    end

    λ = u
    dλ = du

    return λ, nothing, y, dλ, nothing, nothing
end

function split_states(u, t, S::ODEQuadratureAdjointSensitivityFunction; update = true)
    (; y, sol) = S

    if update
        y = sol(t, continuity = :right)
    end

    λ = u

    return λ, nothing, y, nothing, nothing
end

# g is either g(t,u,p) or discrete g(t,u,i)
@noinline function ODEAdjointProblem(
        sol, sensealg::QuadratureAdjoint, alg,
        t = nothing,
        dgdu_discrete::DG1 = nothing,
        dgdp_discrete::DG2 = nothing,
        dgdu_continuous::DG3 = nothing,
        dgdp_continuous::DG4 = nothing,
        g::G = nothing,
        ::Val{RetCB} = Val(false); no_start = false,
        callback = CallbackSet()
    ) where {
        DG1, DG2, DG3, DG4, G,
        RetCB,
    }
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

    discrete = (
        t !== nothing &&
            (
            dgdu_continuous === nothing && dgdp_continuous === nothing ||
                g !== nothing
        )
    )

    if ArrayInterface.ismutable(u0)
        len = length(u0)
        λ = similar(u0, len)
        λ .= false
    else
        λ = zero(u0)
    end
    sense = ODEQuadratureAdjointSensitivityFunction(
        g, sensealg, discrete, sol,
        dgdu_continuous, dgdp_continuous, alg
    )

    init_cb = (discrete || dgdu_discrete !== nothing) # && tspan[1] == t[end]
    z0 = vec(zero(λ))
    cb, rcb,
        _ = generate_callbacks(
        sense, dgdu_discrete, dgdp_discrete,
        λ, t, tspan[2],
        callback, init_cb, terminated, no_start
    )

    jac_prototype = sol.prob.f.jac_prototype
    # The adjoint state is just λ, so the transposed forward prototype is the
    # adjoint rhs sparsity for continuous costs too (the dgdu forcing term is
    # λ-independent).
    adjoint_jac_prototype = jac_prototype === nothing ? nothing :
        copy(jac_prototype')

    # The adjoint rhs is linear in λ, so its Jacobian is exactly -(df/du)^T at
    # the interpolated forward solution: evaluate it from the user's analytical
    # Jacobian when given, or by ForwardDiff/FiniteDiff of the forward rhs.
    # This avoids differentiating the vjp-based adjoint rhs in the implicit
    # adjoint solver.
    adjoint_jac = build_adjoint_jac(sol, sensealg, alg, f, tspan)

    original_mm = sol.prob.f.mass_matrix
    if original_mm === I || original_mm === (I, I)
        odefun = ODEFunction{ArrayInterface.ismutable(z0), true}(
            sense,
            jac_prototype = adjoint_jac_prototype,
            jac = adjoint_jac
        )
    else
        odefun = ODEFunction{ArrayInterface.ismutable(z0), true}(
            sense,
            mass_matrix = sol.prob.f.mass_matrix',
            jac_prototype = adjoint_jac_prototype,
            jac = adjoint_jac
        )
    end
    if RetCB
        return ODEProblem(odefun, z0, tspan, p, callback = cb), rcb
    else
        return ODEProblem(odefun, z0, tspan, p, callback = cb)
    end
end

"""
    AdjointSensitivityIntegrand(sol, adj_sol, sensealg, dgdp=nothing)

Callable quadrature integrand for parameter-gradient accumulation in
`QuadratureAdjoint` and related continuous adjoint paths.

## Arguments

  - `sol`: forward solution.
  - `adj_sol`: reverse adjoint solution.
  - `sensealg`: quadrature-style adjoint algorithm.
  - `dgdp`: optional continuous cost derivative with respect to parameters.

## Returns

A callable object. Calling `integrand(out, t)` writes the row vector
``lambda(t)' * df/ dp + dg / dp`` into `out`, and calling `integrand(t)`
allocates and returns that value.
"""
struct AdjointSensitivityIntegrand{
        pType, uType, lType, rateType, S, AS, PF, PJC, PJT, DGP,
        G, tType, rType, UF, DUT,
    }
    sol::S
    adj_sol::AS
    p::pType
    y::uType
    λ::lType
    pf::PF
    f_cache::rateType
    pJ::PJT
    paramjac_config::PJC
    sensealg::QuadratureAdjoint
    dgdp_cache::DGP
    dgdp::G
    tunables::tType
    repack::rType
    # `unwrapped_f(sol.prob.f)` hoisted to construction: digging the function
    # out of a FunctionWrappersWrapper is dynamic and would otherwise dominate
    # every `vec_pjac!` call.
    unwrappedf::UF
    # Derivative interpolation buffer for fully implicit DAE residuals; `nothing`
    # for ODE problems.
    dy::DUT
end

function AdjointSensitivityIntegrand(sol, adj_sol, sensealg, dgdp = nothing)
    prob = sol.prob
    adj_prob = adj_sol.prob
    (; f, tspan) = prob
    p = parameter_values(prob)
    u0 = state_values(prob)

    _use_full_p = hasproperty(sensealg, :diff_tunables) &&
        sensealg.diff_tunables isa Val{false} &&
        isscimlstructure(p) && !(p isa AbstractArray)
    if _use_full_p
        tunables, repack = p, identity
    elseif isscimlstructure(p) && !(p isa AbstractArray)
        tunables, repack, _ = canonicalize(Tunable(), p)
    else
        tunables, repack = p, identity
    end

    numparams = length(tunables)
    y = zero(state_values(prob))
    λ = zero(state_values(adj_prob))
    # we need to alias `y`
    f_cache = zero(y)
    isautojacvec = get_jacvec(sensealg)

    unwrappedf = prob isa SciMLBase.AbstractDAEProblem ? dae_unwrapped_f(f) :
        unwrapped_f(f)

    dgdp_cache = dgdp === nothing ? nothing : zero(tunables)

    if prob isa SciMLBase.AbstractDAEProblem
        dy = zero(y)
        pf, pJ, paramjac_config = dae_integrand_paramjac_config(
            sensealg, prob, unwrappedf, y, dy, tunables, repack, tspan[2]
        )
        return AdjointSensitivityIntegrand(
            sol, adj_sol, p, y, λ, pf, f_cache, pJ, paramjac_config,
            sensealg, dgdp_cache, dgdp, tunables, repack, unwrappedf, dy
        )
    end

    if sensealg.autojacvec isa ReverseDiffVJP
        tape = if DiffEqBase.isinplace(prob)
            ReverseDiff.GradientTape((y, tunables, [tspan[2]])) do u, tunables, t
                du1 = similar(tunables, size(u))
                du1 .= false
                unwrappedf(du1, u, repack(tunables), first(t))
                return vec(du1)
            end
        else
            ReverseDiff.GradientTape((y, tunables, [tspan[2]])) do u, tunables, t
                vec(unwrappedf(u, repack(tunables), first(t)))
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
        _pf = get_pf(sensealg.autojacvec, prob, f)
        # Wrap pf to accept flat tunables and repack to full parameter struct
        _needs_repack = isscimlstructure(p) && !(p isa AbstractArray)
        pf = if _needs_repack
            let _pf = _pf, repack = repack
                (out, u, _tunables, t) -> _pf(out, u, repack(_tunables), t)
            end
        else
            _pf
        end
        paramjac_config = get_paramjac_config(
            MooncakeLoaded(), sensealg.autojacvec, pf, tunables, f, y, tspan[2]
        )
        pJ = nothing
    elseif sensealg.autojacvec isa ReactantVJP
        pf = get_pf(sensealg.autojacvec, prob, f)
        paramjac_config = get_paramjac_config(
            ReactantLoaded(), sensealg.autojacvec, pf, tunables, f, y, tspan[2]
        )
        pJ = nothing
    elseif isautojacvec # Zygote
        paramjac_config = nothing
        pf = nothing
        pJ = nothing
    else
        _needs_repack = isscimlstructure(p) && !(p isa AbstractArray)
        _pjac_f = _needs_repack ?
            (du, u, p, t) -> unwrappedf(du, u, repack(p), t) :
            unwrappedf
        pf = SciMLBase.ParamJacobianWrapper(_pjac_f, tspan[1], y)
        pJ = similar(u0, length(u0), numparams)
        pJ .= false
        paramjac_config = build_param_jac_config(sensealg, pf, y, tunables)
    end
    return AdjointSensitivityIntegrand(
        sol, adj_sol, p, y, λ, pf, f_cache, pJ, paramjac_config,
        sensealg, dgdp_cache, dgdp, tunables, repack, unwrappedf, nothing
    )
end

function gclosure4(f, du, u, p, t)
    Base.copyto!(du, f(u, p, t))
    return nothing
end

# `λ ⋅ f(y, repack(tunables), t)`, differentiated by Enzyme with respect to
# `tunables` only (everything else held constant). Top-level so the out-of-place
# immutable-state parameter vjp can call `Enzyme.gradient` without a closure.
_enzyme_vecpjac_dot(f, repack, y, tunables, t, λ) = dot(vec(f(y, repack(tunables), t)), vec(λ))

# out = λ df(u, p, t)/dp at u=y, p=p, t=t
function vec_pjac!(out, λ, y, t, S::AdjointSensitivityIntegrand)
    # Fully implicit DAE residuals take the p-vjp of F(du, u, p, t) instead.
    S.dy === nothing || return dae_vec_pjac!(out, λ, y, t, S)
    (; pJ, pf, p, f_cache, dgdp_cache, paramjac_config, sensealg, sol, adj_sol, tunables, repack) = S
    _odef = sol.prob.f
    f = S.unwrappedf

    # Priority: vjp_p > paramjac > autojacvec dispatch.
    if SciMLBase.has_vjp_p(_odef)
        _odef.vjp_p(out, λ, y, p, t)
        return out
    elseif SciMLBase.has_paramjac(_odef)
        _pJ = pJ
        if _pJ === nothing
            _pJ = similar(y, length(y), length(out))
        end
        _odef.paramjac(_pJ, y, p, t)
        # Use _pJ' * λ instead of out' = λ' * _pJ to avoid GPU scalar indexing
        mul!(vec(out), _pJ', λ)
        return out
    end

    isautojacvec = get_jacvec(sensealg)
    # y is aliased
    if !isautojacvec
        if SciMLBase.has_paramjac(f)
            f.paramjac(pJ, y, p, t) # Calculate the parameter Jacobian into pJ
        else
            pf.t = t
            pf.u = y
            jacobian!(pJ, pf, tunables, f_cache, sensealg, paramjac_config)
        end
        # Use pJ' * λ instead of out' = λ' * pJ to avoid GPU scalar indexing
        mul!(vec(out), pJ', λ)
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
        if SciMLBase.isinplace(sol.prob.f)
            _dy, back = Zygote.pullback(tunables) do tunables
                du_buf = Zygote.Buffer(y)
                f(du_buf, y, repack(tunables), t)
                vec(copy(du_buf))
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
    elseif sensealg.autojacvec isa MooncakeVJP
        _, _, p_grad = mooncake_run_ad(paramjac_config, y, p, t, λ)
        out .= p_grad
    elseif sensealg.autojacvec isa ReactantVJP
        reactant_run_ad!(nothing, out, nothing, paramjac_config, y, p, t, λ)
    elseif sensealg.autojacvec isa EnzymeVJP
        if !ArrayInterface.ismutable(y)
            # Out-of-place immutable state (e.g. `SVector`): the buffered
            # `Duplicated` path below needs mutable temporaries built from
            # `zero(y)`, so instead form the parameter vjp `(∂f/∂p)^T λ` as the
            # gradient of `λ ⋅ f(y, repack(tunables), t)` with respect to the
            # tunables, holding `f`, `repack`, `y`, `t`, `λ` constant.
            if out isa typeof(tunables) && ArrayInterface.ismutable(out)
                # Write the gradient directly into `out` instead of allocating a
                # fresh gradient (and result tuple) on every quadrature node.
                Enzyme.remake_zero!(out)
                Enzyme.autodiff(
                    sensealg.autojacvec.mode, Enzyme.Const(_enzyme_vecpjac_dot),
                    Enzyme.Active, Enzyme.Const(f), Enzyme.Const(repack),
                    Enzyme.Const(y), Enzyme.Duplicated(tunables, out),
                    Enzyme.Const(t), Enzyme.Const(λ)
                )
            else
                res = Enzyme.gradient(
                    sensealg.autojacvec.mode, _enzyme_vecpjac_dot, Enzyme.Const(f),
                    Enzyme.Const(repack), Enzyme.Const(y), tunables,
                    Enzyme.Const(t), Enzyme.Const(λ)
                )
                recursive_copyto!(out, res[4])
            end
        else
            tmp3, tmp4, tmp6 = paramjac_config
            vtmp4 = vec(tmp4)

            Enzyme.remake_zero!(out)
            Enzyme.remake_zero!(tmp3)
            vtmp4 .= λ

            _shadow_enzyme = nothing
            if !(p isa AbstractArray)
                _shadow_enzyme = repack(out)
                dup = Enzyme.Duplicated(p, _shadow_enzyme)
            else
                dup = Enzyme.Duplicated(p, out)
            end

            if SciMLBase.isinplace(sol.prob.f)
                Enzyme.remake_zero!(tmp6)
                Enzyme.autodiff(
                    sensealg.autojacvec.mode,
                    Enzyme.Duplicated(SciMLBase.Void(f), tmp6), Enzyme.Const,
                    Enzyme.Duplicated(tmp3, tmp4),
                    Enzyme.Const(y), dup, Enzyme.Const(t)
                )
            else
                tmp6 = Enzyme.make_zero(f)
                Enzyme.autodiff(
                    sensealg.autojacvec.mode, Enzyme.Const(gclosure4), Enzyme.Const,
                    Enzyme.Duplicated(f, tmp6),
                    Enzyme.Duplicated(tmp3, tmp4),
                    Enzyme.Const(y), dup, Enzyme.Const(t)
                )
            end

            if _shadow_enzyme !== nothing && _shadow_enzyme !== out
                _use_full_p_enzyme = hasproperty(sensealg, :diff_tunables) &&
                    sensealg.diff_tunables isa Val{false}
                if !_use_full_p_enzyme && isscimlstructure(_shadow_enzyme)
                    grad_tunables, _, _ = canonicalize(Tunable(), _shadow_enzyme)
                else
                    grad_tunables = _shadow_enzyme
                end
                copyto!(out, grad_tunables)
            end
        end
    end

    # TODO: Add tracker?
    return out
end

function (S::AdjointSensitivityIntegrand)(out, t)
    (; y, λ, pJ, pf, p, f_cache, dgdp_cache, paramjac_config, sensealg, sol, adj_sol) = S
    if ArrayInterface.ismutable(y)
        sol(y, t)
        adj_sol(λ, t)
    else
        y = sol(t)
        λ = adj_sol(t)
    end
    # The DAE adjoint state is z = [w; λ]: extract the λ block.
    _λ = S.dy === nothing ? λ : @view(λ[(length(y) + 1):(2 * length(y))])
    vec_pjac!(out, _λ, y, t, S)
    # For the residual convention F(du, u, p, t) = 0 the parameter-gradient
    # integrand is g_p - λ'F_p, while the raw p-vjp is +F_p'λ (which is what the
    # boundary-jump corrections need), so negate it here.
    S.dy === nothing || recursive_neg!(out)

    if S.dgdp !== nothing
        S.dgdp(dgdp_cache, y, p, t)
        out .+= dgdp_cache
    end
    return out'
end

function (S::AdjointSensitivityIntegrand)(t)
    out = similar(S.tunables)
    out .= false
    return S(out, t)
end

function _adjoint_sensitivities(
        sol, sensealg::QuadratureAdjoint, alg; t = nothing,
        dgdu_discrete = nothing,
        dgdp_discrete = nothing,
        dgdu_continuous = nothing,
        dgdp_continuous = nothing,
        g = nothing, no_start = false,
        abstol = sensealg.abstol, reltol = sensealg.reltol,
        callback = CallbackSet(),
        kwargs...
    )
    callback = adjoint_callbacks(sol, callback)
    adj_prob,
        rcb = if sol.prob isa SciMLBase.AbstractDAEProblem
        DAEAdjointProblem(
            sol, sensealg, alg, t, dgdu_discrete, dgdp_discrete,
            dgdu_continuous, dgdp_continuous, g, Val(true);
            callback, no_start
        )
    else
        ODEAdjointProblem(
            sol, sensealg, alg, t, dgdu_discrete, dgdp_discrete,
            dgdu_continuous, dgdp_continuous, g, Val(true);
            callback, no_start
        )
    end
    adj_sol = solve(
        adj_prob, alg; abstol, reltol,
        save_everystep = true, save_start = true, kwargs...
    )

    # The DAE adjoint state is z = [w; λ] with dG/du0 = w(t0).
    du0 = if sol.prob isa SciMLBase.AbstractDAEProblem
        state_values(adj_sol)[end][1:length(state_values(sol.prob))]
    else
        state_values(adj_sol)[end]
    end

    p = sol.prob.p
    if p === nothing || p === SciMLBase.NullParameters()
        return du0, nothing
    else
        integrand = AdjointSensitivityIntegrand(sol, adj_sol, sensealg, dgdp_continuous)
        if t === nothing
            res,
                err = quadgk(
                integrand, sol.prob.tspan[1], sol.prob.tspan[2],
                atol = abstol, rtol = reltol
            )
        else
            res = zero(integrand.tunables)'

            # handle discrete dgdp contributions
            if dgdp_discrete !== nothing
                (; y) = integrand
                cur_time = length(t)
                dgdp_cache = copy(res)
                dgdp_discrete(dgdp_cache, y, p, t[cur_time], cur_time)
                res .+= dgdp_cache
            end

            if callback !== nothing
                cur_time = length(t)
                dλ = similar(integrand.λ)
                dλ .*= false
                dgrad = similar(res)
                dgrad .*= false
            end

            # correction for end interval.
            if t[end] != sol.prob.tspan[2] && sol.retcode !== ReturnCode.Terminated
                res .+= quadgk(
                    integrand, t[end], sol.prob.tspan[end],
                    atol = abstol, rtol = reltol
                )[1]
            end

            if sol.retcode === ReturnCode.Terminated
                integrand = update_integrand_and_dgrad(
                    res, sensealg, callback, integrand,
                    adj_prob, sol, dgdu_discrete,
                    dgdp_discrete, dλ, dgrad, t[end],
                    cur_time
                )
            end

            for i in (length(t) - 1):-1:1
                if ArrayInterface.ismutable(res)
                    res .+= quadgk(
                        integrand, t[i], t[i + 1],
                        atol = abstol, rtol = reltol
                    )[1]
                else
                    res += quadgk(
                        integrand, t[i], t[i + 1],
                        atol = abstol, rtol = reltol
                    )[1]
                end
                if t[i] == t[i + 1]
                    integrand = update_integrand_and_dgrad(
                        res, sensealg, callback,
                        integrand,
                        adj_prob, sol, dgdu_discrete,
                        dgdp_discrete, dλ, dgrad, t[i],
                        cur_time
                    )
                end
                if dgdp_discrete !== nothing
                    (; y) = integrand
                    dgdp_discrete(dgdp_cache, y, p, t[cur_time], cur_time)
                    res .+= dgdp_cache
                end
                (callback !== nothing || dgdp_discrete !== nothing) &&
                    (cur_time -= one(cur_time))
            end
            # correction for start interval
            if t[1] != sol.prob.tspan[1]
                res .+= quadgk(
                    integrand, sol.prob.tspan[1], t[1],
                    atol = abstol, rtol = reltol
                )[1]
            end
        end
        if rcb !== nothing && !isempty(rcb.Δλas)
            iλ = zero(rcb.λ)
            out = zero(res')
            yy = similar(rcb.y)
            yy .= false
            for (Δλa, tt) in rcb.Δλas
                (; algeeq_idxs) = rcb.diffcache
                iλ[algeeq_idxs] .= Δλa
                sol(yy, tt)
                vec_pjac!(out, iλ, yy, tt, integrand)
                res .+= out'
                iλ .= zero(eltype(iλ))
            end
        end
        return du0, res
    end
end

function update_p_integrand(integrand::AdjointSensitivityIntegrand, p)
    (;
        sol, adj_sol, y, λ, pf, f_cache, pJ, paramjac_config,
        sensealg, dgdp_cache, dgdp, unwrappedf, dy,
    ) = integrand
    _use_full_p = hasproperty(sensealg, :diff_tunables) &&
        sensealg.diff_tunables isa Val{false} &&
        isscimlstructure(p) && !(p isa AbstractArray)
    if _use_full_p
        tunables, repack = p, identity
    elseif isscimlstructure(p) && !(p isa AbstractArray)
        tunables, repack, _ = canonicalize(Tunable(), p)
    else
        tunables, repack = p, identity
    end
    return AdjointSensitivityIntegrand(
        sol, adj_sol, p, y, λ, pf, f_cache, pJ, paramjac_config,
        sensealg, dgdp_cache, dgdp, tunables, repack, unwrappedf, dy
    )
end

function update_integrand_and_dgrad(
        res, sensealg::QuadratureAdjoint, callbacks, integrand,
        adj_prob, sol, dgdu_discrete, dgdp_discrete, dλ, dgrad,
        ti, cur_time
    )
    callbacks = CallbackSet(callbacks)
    for cb in callbacks.discrete_callbacks
        is_tracked_callback(cb) || continue
        if ti ∈ cb.affect!.event_times
            integrand = _update_integrand_and_dgrad(
                res, sensealg, cb,
                integrand, adj_prob, sol,
                dgdu_discrete,
                dgdp_discrete, dλ, dgrad,
                ti, cur_time
            )
        end
    end
    for cb in callbacks.continuous_callbacks
        is_tracked_callback(cb) || continue
        if ti ∈ cb.affect!.event_times ||
                ti ∈ tracked_event_times(cb.affect_neg!)
            integrand = _update_integrand_and_dgrad(
                res, sensealg, cb,
                integrand, adj_prob, sol,
                dgdu_discrete,
                dgdp_discrete, dλ, dgrad,
                ti, cur_time
            )
        end
    end
    return integrand
end

function _update_integrand_and_dgrad(
        res, sensealg::QuadratureAdjoint, cb, integrand,
        adj_prob, sol, dgdu, dgdp, dλ, dgrad, t, cur_time
    )
    indx, pos_neg = get_indx(cb, t)
    tprev = get_tprev(cb, indx, pos_neg)

    # Callbacks always use ReverseDiffVJP for their VJP computations,
    # independent of the ODE adjoint's autojacvec choice.
    cb_autojacvec = ReverseDiffVJP()
    cb_sensealg = setvjp(sensealg, cb_autojacvec)

    wp = CallbackAffectPWrapper(cb, cb_autojacvec, pos_neg, nothing, tprev)

    # `wp` writes the tunable portion of the post-event parameters; the integrand
    # is rebuilt from the full parameter object, so repack before comparing.
    _pt = similar(integrand.tunables, size(integrand.tunables))
    _pt .= false
    wp(_pt, integrand.y, integrand.p, t)

    if _pt != integrand.tunables
        paramjac_config = _get_wp_paramjac_config(
            cb_autojacvec, integrand.p, wp, integrand.y, integrand.p, t
        )
        pf = get_pf(cb_autojacvec; _f = wp, isinplace = true, isRODE = false)
        if cb_autojacvec isa EnzymeVJP
            paramjac_config = (paramjac_config..., Enzyme.make_zero(pf), nothing)
        end

        diffcache_wp = AdjointDiffCache(
            nothing, pf, nothing, nothing, nothing,
            nothing, nothing, nothing, paramjac_config,
            nothing, nothing, nothing, nothing, nothing,
            nothing, nothing, nothing, false,
            nothing, identity,
            nothing, nothing, false, false
        )

        fakeSp = CallbackSensitivityFunctionPSwap(wp, cb_sensealg, diffcache_wp, sol.prob)
        #vjp with Jacobin given by dw/dp before event and vector given by grad
        vecjacobian!(
            nothing, integrand.y, res, integrand.p, t, fakeSp;
            dgrad = res, dy = nothing
        )
        integrand = update_p_integrand(integrand, integrand.repack(_pt))
    end

    w = CallbackAffectWrapper(cb, cb_autojacvec, pos_neg, nothing, tprev)

    # Create a fake sensitivity function to do the vjps needs to be done
    # to account for parameter dependence of affect function
    fakeS = CallbackSensitivityFunction(w, cb_sensealg, adj_prob.f.f.diffcache, sol.prob)
    if dgdu !== nothing # discrete cost
        dgdu(dλ, integrand.y, integrand.p, t, cur_time)
    else
        error("Please provide `dgdu` to use adjoint_sensitivities with `QuadratureAdjoint()` and callbacks.")
    end

    @assert dgdp === nothing

    # account for implicit events

    @. dλ = -dλ - integrand.λ
    vecjacobian!(dλ, integrand.y, dλ, integrand.p, t, fakeS; dgrad)
    res .-= dgrad
    return integrand
end
