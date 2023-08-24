struct ODEGaussAdjointSensitivityFunction{C <: AdjointDiffCache,
    Alg <: GaussAdjoint,
    uType, SType,
    fType <: DiffEqBase.AbstractDiffEqFunction,
} <: SensitivityFunction
    diffcache::C
    sensealg::Alg
    discrete::Bool
    y::uType
    sol::SType
    f::fType
end

TruncatedStacktraces.@truncate_stacktrace ODEGaussAdjointSensitivityFunction

function ODEGaussAdjointSensitivityFunction(g, sensealg, discrete, sol, dgdu, dgdp,
    alg)
    diffcache, y = adjointdiffcache(g, sensealg, discrete, sol, dgdu, dgdp, sol.prob.f, alg;
        quad = true)
    return ODEGaussAdjointSensitivityFunction(diffcache, sensealg, discrete,
        y, sol, sol.prob.f)
end

# u = λ'
function (S::ODEGaussAdjointSensitivityFunction)(du, u, p, t)
    @unpack sol, discrete = S
    f = sol.prob.f

    λ, grad, y, dλ, dgrad, dy = split_states(du, u, t, S)

    vecjacobian!(dλ, y, λ, p, t, S)
    dλ .*= -one(eltype(λ))

    discrete || accumulate_cost!(dλ, y, p, t, S)
    return nothing
end

function (S::ODEGaussAdjointSensitivityFunction)(u, p, t)
    @unpack sol, discrete = S
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
    @unpack y, sol = S

    if update
        if typeof(t) <: ForwardDiff.Dual && eltype(y) <: AbstractFloat
            y = sol(t, continuity = :right)
        else
            sol(y, t, continuity = :right)
        end
    end

    λ = u
    dλ = du

    λ, nothing, y, dλ, nothing, nothing
end

function split_states(u, t, S::ODEGaussAdjointSensitivityFunction; update = true)
    @unpack y, sol = S

    if update
        y = sol(t, continuity = :right)
    end

    λ = u

    λ, nothing, y, nothing, nothing
end

# g is either g(t,u,p) or discrete g(t,u,i)
@noinline function ODEAdjointProblem(sol, sensealg::GaussAdjoint, alg,
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

    @unpack p, u0, tspan = sol.prob

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
            tspan = (tspan[1], sol.t[end])
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
    sense = ODEGaussAdjointSensitivityFunction(g, sensealg, discrete, sol,
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
        return ODEProblem(odefun, z0, tspan, p), cb, rcb
    else
        return ODEProblem(odefun, z0, tspan, p, callback = cb), cb, rcb
    end
end

struct GaussIntegrand{pType, uType, lType, rateType, S, PF, PJC, PJT, DGP,
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

function GaussIntegrand(sol, sensealg, dgdp = nothing)
    prob = sol.prob
    @unpack f, p, tspan, u0 = prob
    numparams = length(p)
    y = zero(sol.prob.u0)
    λ = zero(sol.prob.u0)
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
        paramjac_config = zero(y), zero(y)
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
        pJ = nothing
    elseif isautojacvec # Zygote
        paramjac_config = nothing
        pf = nothing
        pJ = nothing
    else
        pf = DiffEqBase.ParamJacobianWrapper(unwrappedf, tspan[1], y)
        pJ = similar(u0, length(u0), numparams)
        paramjac_config = build_param_jac_config(sensealg, pf, y, p)
    end
    GaussIntegrand(sol, p, y, λ, pf, f_cache, pJ, paramjac_config,
        sensealg, dgdp_cache, dgdp)
end

# out = λ df(u, p, t)/dp at u=y, p=p, t=t
function vec_pjac!(out, λ, y, t, S::GaussIntegrand)
    @unpack pJ, pf, p, f_cache, dgdp_cache, paramjac_config, sensealg, sol = S
    f = sol.prob.f
    isautojacvec = get_jacvec(sensealg)
    # y is aliased

    if !isautojacvec
        if DiffEqBase.has_paramjac(f)
            f.paramjac(pJ, y, p, t) # Calculate the parameter Jacobian into pJ
        else
            pf.t = t
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
        out[:] .= vec(tmp[1])
    elseif sensealg.autojacvec isa EnzymeVJP
        tmp3, tmp4 = paramjac_config
        tmp4 .= λ
        out .= 0
        Enzyme.autodiff(Enzyme.Reverse, pf, Enzyme.Duplicated(tmp3, tmp4),
            y, Enzyme.Duplicated(p, out), t)
    else
        error("autojacvec choice $(sensealg.autojacvec) is not supported by GaussAdjoint")
    end

    # TODO: Add tracker?
    return out
end

function (S::GaussIntegrand)(out, t, λ)
    @unpack y, pJ, pf, p, f_cache, dgdp_cache, paramjac_config, sensealg, sol = S
    if ArrayInterface.ismutable(y)
        sol(y, t)
    else
        y = sol(t)
    end
    vec_pjac!(out, λ, y, t, S)

    if S.dgdp !== nothing
        S.dgdp(dgdp_cache, y, p, t)
        out .+= dgdp_cache
    end
    out'
end

function (S::GaussIntegrand)(t, λ)
    out = similar(S.p)
    S(out, t, λ)
end

function _adjoint_sensitivities(sol, sensealg::GaussAdjoint, alg; t = nothing,
    dgdu_discrete = nothing,
    dgdp_discrete = nothing,
    dgdu_continuous = nothing,
    dgdp_continuous = nothing,
    g = nothing,
    abstol = sensealg.abstol, reltol = sensealg.reltol,
    callback = CallbackSet(),
    kwargs...)
    integrand = GaussIntegrand(sol, sensealg, dgdp_continuous)
    integrand_values = IntegrandValues(Vector{Float64})
    cb = IntegratingCallback((out, u, t, integrator) -> vec(integrand(out, t, u)), integrand_values, similar(sol.prob.p))
    adj_prob, cb2, rcb = ODEAdjointProblem(sol, sensealg, alg, t, dgdu_discrete, dgdp_discrete,
        dgdu_continuous, dgdp_continuous, g, Val(true);
        callback)
    adj_sol = solve(adj_prob, alg; abstol = abstol, reltol = reltol, save_everystep = false, 
            save_start = false, save_end = true, callback = CallbackSet(cb,cb2), kwargs...)
    res = compute_dGdp(integrand_values)'

    if rcb !== nothing && !isempty(rcb.Δλas)
        iλ = zero(rcb.λ)
        out = zero(res')
        yy = similar(rcb.y)
        for (Δλa, tt) in rcb.Δλas
            @unpack algevar_idxs = rcb.diffcache
            iλ[algevar_idxs] .= Δλa
            sol(yy, tt)
            vec_pjac!(out, iλ, yy, tt, integrand)
            res .+= out'
            iλ .= zero(eltype(iλ))
        end
    end

    return adj_sol[end], res
end

function compute_dGdp(integrand::IntegrandValues)
    res = zeros(length(integrand.integrand[1]))
    for (i, j) in enumerate(integrand.integrand)
        res .+= j
    end
    return res
end

function update_p_integrand(integrand::GaussIntegrand, p)
    @unpack sol, y, λ, pf, f_cache, pJ, paramjac_config, sensealg, dgdp_cache, dgdp = integrand
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
