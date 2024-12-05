struct ODEQuadratureAdjointSensitivityFunction{C <: AdjointDiffCache,
    Alg <: QuadratureAdjoint,
    uType, SType,
    fType <: AbstractDiffEqFunction
} <: SensitivityFunction
    diffcache::C
    sensealg::Alg
    discrete::Bool
    y::uType
    sol::SType
    f::fType
end

function ODEQuadratureAdjointSensitivityFunction(g, sensealg, discrete, sol, dgdu, dgdp,
        alg)
    diffcache, y = adjointdiffcache(
        g, sensealg, discrete, sol, dgdu, dgdp, sol.prob.f, alg;
        quad = true)
    return ODEQuadratureAdjointSensitivityFunction(diffcache, sensealg, discrete,
        y, sol, sol.prob.f)
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

    dy, dλ, dgrad = vecjacobian(y, λ, p, t, S; dgrad = dgrad, dy = dy)
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

    λ, nothing, y, dλ, nothing, nothing
end

function split_states(u, t, S::ODEQuadratureAdjointSensitivityFunction; update = true)
    (; y, sol) = S

    if update
        y = sol(t, continuity = :right)
    end

    λ = u

    λ, nothing, y, nothing, nothing
end

# g is either g(t,u,p) or discrete g(t,u,i)
@noinline function ODEAdjointProblem(sol, sensealg::QuadratureAdjoint, alg,
        t = nothing,
        dgdu_discrete::DG1 = nothing,
        dgdp_discrete::DG2 = nothing,
        dgdu_continuous::DG3 = nothing,
        dgdp_continuous::DG4 = nothing,
        g::G = nothing,
        ::Val{RetCB} = Val(false);
        callback = CallbackSet()) where {DG1, DG2, DG3, DG4, G,
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

    if ArrayInterface.ismutable(u0)
        len = length(u0)
        λ = similar(u0, len)
        λ .= false
    else
        λ = zero(u0)
    end
    sense = ODEQuadratureAdjointSensitivityFunction(g, sensealg, discrete, sol,
        dgdu_continuous, dgdp_continuous, alg)

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
        return ODEProblem(odefun, z0, tspan, p, callback = cb), rcb
    else
        return ODEProblem(odefun, z0, tspan, p, callback = cb)
    end
end

struct AdjointSensitivityIntegrand{pType, uType, lType, rateType, S, AS, PF, PJC, PJT, DGP,
    G}
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
end

function AdjointSensitivityIntegrand(sol, adj_sol, sensealg, dgdp = nothing)
    prob = sol.prob
    adj_prob = adj_sol.prob
    (; f, tspan) = prob
    p = parameter_values(prob)
    u0 = state_values(prob)
    numparams = length(p)
    y = zero(state_values(prob))
    λ = zero(state_values(adj_prob))
    # we need to alias `y`
    f_cache = zero(y)
    isautojacvec = get_jacvec(sensealg)

    unwrappedf = unwrapped_f(f)

    dgdp_cache = dgdp === nothing ? nothing : zero(p)

    if sensealg.autojacvec isa ReverseDiffVJP
        tape = if DiffEqBase.isinplace(prob)
            ReverseDiff.GradientTape((y, prob.p, [tspan[2]])) do u, p, t
                du1 = similar(p, size(u))
                du1 .= false
                unwrappedf(du1, u, p, first(t))
                return vec(du1)
            end
        else
            ReverseDiff.GradientTape((y, prob.p, [tspan[2]])) do u, p, t
                vec(unwrappedf(u, p, first(t)))
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
        pJ .= false
        paramjac_config = build_param_jac_config(sensealg, pf, y, p)
    end
    AdjointSensitivityIntegrand(sol, adj_sol, p, y, λ, pf, f_cache, pJ, paramjac_config,
        sensealg, dgdp_cache, dgdp)
end

# out = λ df(u, p, t)/dp at u=y, p=p, t=t
function vec_pjac!(out, λ, y, t, S::AdjointSensitivityIntegrand)
    (; pJ, pf, p, f_cache, dgdp_cache, paramjac_config, sensealg, sol, adj_sol) = S
    f = sol.prob.f
    isautojacvec = get_jacvec(sensealg)
    # y is aliased
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
        ReverseDiff.value!(tp, p)
        ReverseDiff.value!(tt, [t])
        ReverseDiff.forward_pass!(tape)
        ReverseDiff.increment_deriv!(output, λ)
        ReverseDiff.reverse_pass!(tape)
        copyto!(vec(out), ReverseDiff.deriv(tp))
    elseif sensealg.autojacvec isa ZygoteVJP
        _dy, back = Zygote.pullback(p) do p
            vec(f(y, p, t))
        end
        tmp = back(λ)
        if tmp[1] === nothing
            out[:] .= 0
        else
            out[:] .= vec(tmp[1])
        end
    elseif sensealg.autojacvec isa MooncakeVJP
        _, _, p_grad = mooncake_run_ad(paramjac_config, y, p, t, λ)
        out .= p_grad
    elseif sensealg.autojacvec isa EnzymeVJP
        tmp3, tmp4, tmp6 = paramjac_config
        tmp4 .= λ
        out .= 0
        Enzyme.autodiff(
            Enzyme.Reverse, Enzyme.Duplicated(pf, tmp6), Enzyme.Const,
            Enzyme.Duplicated(tmp3, tmp4),
            Enzyme.Const(y), Enzyme.Duplicated(p, out), Enzyme.Const(t))
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
    vec_pjac!(out, λ, y, t, S)

    if S.dgdp !== nothing
        S.dgdp(dgdp_cache, y, p, t)
        out .+= dgdp_cache
    end
    out'
end

function (S::AdjointSensitivityIntegrand)(t)
    out = similar(S.p)
    out .= false
    S(out, t)
end

function _adjoint_sensitivities(sol, sensealg::QuadratureAdjoint, alg; t = nothing,
        dgdu_discrete = nothing,
        dgdp_discrete = nothing,
        dgdu_continuous = nothing,
        dgdp_continuous = nothing,
        g = nothing,
        abstol = sensealg.abstol, reltol = sensealg.reltol,
        callback = CallbackSet(),
        kwargs...)
    adj_prob, rcb = ODEAdjointProblem(sol, sensealg, alg, t, dgdu_discrete, dgdp_discrete,
        dgdu_continuous, dgdp_continuous, g, Val(true);
        callback)
    adj_sol = solve(adj_prob, alg; abstol = abstol, reltol = reltol,
        save_everystep = true, save_start = true, kwargs...)

    p = sol.prob.p
    if p === nothing || p === SciMLBase.NullParameters()
        return state_values(adj_sol)[end], nothing
    else
        integrand = AdjointSensitivityIntegrand(sol, adj_sol, sensealg, dgdp_continuous)
        if t === nothing
            res, err = quadgk(integrand, sol.prob.tspan[1], sol.prob.tspan[2],
                atol = abstol, rtol = reltol)
        else
            res = zero(integrand.p)'

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
                res .+= quadgk(integrand, t[end], sol.prob.tspan[end],
                    atol = abstol, rtol = reltol)[1]
            end

            if sol.retcode === ReturnCode.Terminated
                integrand = update_integrand_and_dgrad(res, sensealg, callback, integrand,
                    adj_prob, sol, dgdu_discrete,
                    dgdp_discrete, dλ, dgrad, t[end],
                    cur_time)
            end

            for i in (length(t) - 1):-1:1
                if ArrayInterface.ismutable(res)
                    res .+= quadgk(integrand, t[i], t[i + 1],
                        atol = abstol, rtol = reltol)[1]
                else
                    res += quadgk(integrand, t[i], t[i + 1],
                        atol = abstol, rtol = reltol)[1]
                end
                if t[i] == t[i + 1]
                    integrand = update_integrand_and_dgrad(res, sensealg, callback,
                        integrand,
                        adj_prob, sol, dgdu_discrete,
                        dgdp_discrete, dλ, dgrad, t[i],
                        cur_time)
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
                res .+= quadgk(integrand, sol.prob.tspan[1], t[1],
                    atol = abstol, rtol = reltol)[1]
            end
        end
        if rcb !== nothing && !isempty(rcb.Δλas)
            iλ = zero(rcb.λ)
            out = zero(res')
            yy = similar(rcb.y)
            yy .= false
            for (Δλa, tt) in rcb.Δλas
                (; algevar_idxs) = rcb.diffcache
                iλ[algevar_idxs] .= Δλa
                sol(yy, tt)
                vec_pjac!(out, iλ, yy, tt, integrand)
                res .+= out'
                iλ .= zero(eltype(iλ))
            end
        end
        return state_values(adj_sol)[end], res
    end
end

function update_p_integrand(integrand::AdjointSensitivityIntegrand, p)
    (; sol, adj_sol, y, λ, pf, f_cache, pJ, paramjac_config, sensealg, dgdp_cache, dgdp) = integrand
    AdjointSensitivityIntegrand(sol, adj_sol, p, y, λ, pf, f_cache, pJ, paramjac_config,
        sensealg, dgdp_cache, dgdp)
end

function update_integrand_and_dgrad(res, sensealg::QuadratureAdjoint, callbacks, integrand,
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

function _update_integrand_and_dgrad(res, sensealg::QuadratureAdjoint, cb, integrand,
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
    _p .= false
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
        error("Please provide `dgdu` to use adjoint_sensitivities with `QuadratureAdjoint()` and callbacks.")
    end

    @assert dgdp === nothing

    # account for implicit events

    @. dλ = -dλ - integrand.λ
    vecjacobian!(dλ, integrand.y, dλ, integrand.p, t, fakeS; dgrad = dgrad)
    res .-= dgrad
    return integrand
end
