struct ODEQuadratureAdjointSensitivityFunction{C <: AdjointDiffCache,
                                               Alg <: QuadratureAdjoint,
                                               uType, SType,
                                               fType <: DiffEqBase.AbstractDiffEqFunction
                                               } <: SensitivityFunction
    diffcache::C
    sensealg::Alg
    discrete::Bool
    y::uType
    sol::SType
    f::fType
end

function ODEQuadratureAdjointSensitivityFunction(g, sensealg, discrete, sol, dg)
    diffcache, y = adjointdiffcache(g, sensealg, discrete, sol, dg, sol.prob.f; quad = true)
    return ODEQuadratureAdjointSensitivityFunction(diffcache, sensealg, discrete,
                                                   y, sol, sol.prob.f)
end

# u = λ'
function (S::ODEQuadratureAdjointSensitivityFunction)(du, u, p, t)
    @unpack sol, discrete = S
    f = sol.prob.f

    λ, grad, y, dλ, dgrad, dy = split_states(du, u, t, S)

    vecjacobian!(dλ, y, λ, p, t, S)
    dλ .*= -one(eltype(λ))

    discrete || accumulate_cost!(dλ, y, p, t, S)
    return nothing
end

function split_states(du, u, t, S::ODEQuadratureAdjointSensitivityFunction; update = true)
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

# g is either g(t,u,p) or discrete g(t,u,i)
@noinline function ODEAdjointProblem(sol, sensealg::QuadratureAdjoint,
                                     t = nothing,
                                     dg_discrete::DG1 = nothing,
                                     dg_continuous::DG2 = nothing,
                                     g::G = nothing;
                                     callback = CallbackSet()) where {DG1, DG2, G}
    dg_discrete === nothing && dg_continuous === nothing && g === nothing &&
        error("Either `dg_discrete`, `dg_continuous`, or `g` must be specified.")

    @unpack f, p, u0, tspan = sol.prob
    terminated = false
    if hasfield(typeof(sol), :retcode)
        if sol.retcode == :Terminated
            tspan = (tspan[1], sol.t[end])
            terminated = true
        end
    end
    tspan = reverse(tspan)

    discrete = (t !== nothing && dg_continuous === nothing)

    len = length(u0)
    λ = similar(u0, len)
    λ .= false
    sense = ODEQuadratureAdjointSensitivityFunction(g, sensealg, discrete, sol,
                                                    dg_continuous)

    init_cb = (discrete || dg_discrete !== nothing) # && tspan[1] == t[end]
    z0 = vec(zero(λ))
    cb, duplicate_iterator_times = generate_callbacks(sense, dg_discrete, λ, t, tspan[2],
                                                      callback, init_cb, terminated)

    jac_prototype = sol.prob.f.jac_prototype
    adjoint_jac_prototype = !sense.discrete || jac_prototype === nothing ? nothing :
                            copy(jac_prototype')

    original_mm = sol.prob.f.mass_matrix
    if original_mm === I || original_mm === (I, I)
        odefun = ODEFunction(sense, jac_prototype = adjoint_jac_prototype)
    else
        odefun = ODEFunction(sense, mass_matrix = sol.prob.f.mass_matrix',
                             jac_prototype = adjoint_jac_prototype)
    end
    return ODEProblem(odefun, z0, tspan, p, callback = cb)
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
    @unpack f, p, tspan, u0 = prob
    numparams = length(p)
    y = zero(sol.prob.u0)
    λ = zero(adj_sol.prob.u0)
    # we need to alias `y`
    f_cache = zero(y)
    f_cache .= false
    isautojacvec = get_jacvec(sensealg)

    dgdp_cache = dgdp === nothing ? nothing : zero(p)

    if sensealg.autojacvec isa ReverseDiffVJP
        tape = if DiffEqBase.isinplace(prob)
            ReverseDiff.GradientTape((y, prob.p, [tspan[2]])) do u, p, t
                du1 = similar(p, size(u))
                du1 .= false
                f(du1, u, p, first(t))
                return vec(du1)
            end
        else
            ReverseDiff.GradientTape((y, prob.p, [tspan[2]])) do u, p, t
                vec(f(u, p, first(t)))
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
        pf = let f = f.f
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
        pf = DiffEqBase.ParamJacobianWrapper(f, tspan[1], y)
        pJ = similar(u0, length(u0), numparams)
        paramjac_config = build_param_jac_config(sensealg, pf, y, p)
    end
    AdjointSensitivityIntegrand(sol, adj_sol, p, y, λ, pf, f_cache, pJ, paramjac_config,
                                sensealg, dgdp_cache, dgdp)
end

function (S::AdjointSensitivityIntegrand)(out, t)
    @unpack y, λ, pJ, pf, p, f_cache, dgdp_cache, paramjac_config, sensealg, sol, adj_sol = S
    f = sol.prob.f
    sol(y, t)
    adj_sol(λ, t)
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
        Enzyme.autodiff(pf, Enzyme.Duplicated(tmp3, tmp4),
                        y, Enzyme.Duplicated(p, out), t)
    end

    # TODO: Add tracker?

    if S.dgdp !== nothing
        S.dgdp(dgdp_cache, y, p, t)
        out .+= dgdp_cache
    end
    out'
end

function (S::AdjointSensitivityIntegrand)(t)
    out = similar(S.p)
    S(out, t)
end

function _adjoint_sensitivities(sol, sensealg::QuadratureAdjoint, alg; t = nothing,
                                dg_discrete = nothing, dg_continuous = nothing,
                                g = nothing,
                                abstol = sensealg.abstol, reltol = sensealg.reltol,
                                callback = CallbackSet(),
                                kwargs...)
    dgdu, dgdp = dg_continuous isa Tuple ? dg_continuous : (dg_continuous, nothing)
    adj_prob = ODEAdjointProblem(sol, sensealg, t, dg_discrete, dgdu, g; callback)
    adj_sol = solve(adj_prob, alg; abstol = abstol, reltol = reltol,
                    save_everystep = true, save_start = true, kwargs...)

    p = sol.prob.p
    if p === nothing || p === DiffEqBase.NullParameters()
        return -adj_sol[end], nothing
    else
        integrand = AdjointSensitivityIntegrand(sol, adj_sol, sensealg, dgdp)

        if t === nothing
            res, err = quadgk(integrand, sol.prob.tspan[1], sol.prob.tspan[2],
                              atol = abstol, rtol = reltol)
        else
            res = zero(integrand.p)'

            if callback !== nothing
                cur_time = length(t)
                dλ = similar(integrand.λ)
                dλ .*= false
                dgrad = similar(res)
                dgrad .*= false
            end

            # correction for end interval.
            if t[end] != sol.prob.tspan[2]
                res .+= quadgk(integrand, t[end], sol.prob.tspan[end],
                               atol = abstol, rtol = reltol)[1]
            end

            for i in (length(t) - 1):-1:1
                res .+= quadgk(integrand, t[i], t[i + 1],
                               atol = abstol, rtol = reltol)[1]
                if t[i] == t[i + 1]
                    for cb in callback.discrete_callbacks
                        if t[i] ∈ cb.affect!.event_times
                            integrand = update_integrand_and_dgrad(res, sensealg, cb,
                                                                   integrand, adj_prob, sol,
                                                                   dg_discrete, dλ, dgrad,
                                                                   t[i], cur_time)
                        end
                    end
                    for cb in callback.continuous_callbacks
                        if t[i] ∈ cb.affect!.event_times ||
                           t[i] ∈ cb.affect_neg!.event_times
                            integrand = update_integrand_and_dgrad(res, sensealg, cb,
                                                                   integrand, adj_prob, sol,
                                                                   dg_discrete, dλ, dgrad,
                                                                   t[i], cur_time)
                        end
                    end
                end
                callback !== nothing && (cur_time -= one(cur_time))
            end
            # correction for start interval
            if t[1] != sol.prob.tspan[1]
                res .+= quadgk(integrand, sol.prob.tspan[1], t[1],
                               atol = abstol, rtol = reltol)[1]
            end
        end
        return adj_sol[end], res
    end
end

function update_p_integrand(integrand::AdjointSensitivityIntegrand, p)
    @unpack sol, adj_sol, y, λ, pf, f_cache, pJ, paramjac_config, sensealg, dgdp_cache, dgdp = integrand
    AdjointSensitivityIntegrand(sol, adj_sol, p, y, λ, pf, f_cache, pJ, paramjac_config,
                                sensealg, dgdp_cache, dgdp)
end

function update_integrand_and_dgrad(res, sensealg::QuadratureAdjoint, cb, integrand,
                                    adj_prob, sol, dg, dλ, dgrad, t, cur_time)
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
    if dg !== nothing # discrete cost
        dg(dλ, integrand.y, integrand.p, t, cur_time)
    else
        error("Please provide `dg_discrete` to use adjoint_sensitivities with `QuadratureAdjoint()` and callbacks.")
    end

    # account for implicit events

    @. dλ = -dλ - integrand.λ
    vecjacobian!(dλ, integrand.y, dλ, integrand.p, t, fakeS; dgrad = dgrad)
    res .-= dgrad
    return integrand
end
