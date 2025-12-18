module SciMLSensitivityMooncakeExt

using SciMLSensitivity: SciMLSensitivity
using Mooncake: Mooncake
import SciMLSensitivity: get_paramjac_config, mooncake_run_ad, MooncakeVJP, MooncakeLoaded,
                         DiffEqBase, MooncakeAdjoint
using SciMLSensitivity: SciMLBase, SciMLStructures, canonicalize, Tunable, isscimlstructure,
                        SciMLStructuresCompatibilityError, convert_tspan,
                        has_continuous_callback,
                        unwrapped_f, state_values, current_time
using SciMLSensitivity: FunctionWrappersWrappers, ODEFunction
using ChainRulesCore: NoTangent, ZeroTangent, Tangent, unthunk
using Accessors: @reset

function get_paramjac_config(::MooncakeLoaded, ::MooncakeVJP, pf, p, f, y, _t)
    dy_mem = zero(y)
    λ_mem = zero(y)
    cache = Mooncake.prepare_pullback_cache(pf, dy_mem, y, p, _t)
    return cache, pf, λ_mem, dy_mem
end

function mooncake_run_ad(paramjac_config::Tuple, y, p, t, λ)
    cache, pf, λ_mem, dy_mem = paramjac_config
    λ_mem .= λ
    dy, _ = Mooncake.value_and_pullback!!(cache, λ_mem, pf, dy_mem, y, p, t)
    y_grad = cache.tangents[3]
    p_grad = cache.tangents[4]
    return dy, y_grad, p_grad
end

function SciMLBase._concrete_solve_adjoint(
        prob::Union{SciMLBase.AbstractDiscreteProblem,
            SciMLBase.AbstractODEProblem,
            SciMLBase.AbstractDAEProblem,
            SciMLBase.AbstractDDEProblem,
            SciMLBase.AbstractSDEProblem,
            SciMLBase.AbstractSDDEProblem,
            SciMLBase.AbstractRODEProblem},
        alg, sensealg::MooncakeAdjoint,
        u0, p, originator::SciMLBase.ADOriginator,
        args...;
        kwargs...)
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
        if (convert_tspan(sensealg) === nothing &&
            ((haskey(kwargs, :callback) && has_continuous_callback(kwargs[:callback])))) ||
           (convert_tspan(sensealg) !== nothing && convert_tspan(sensealg))
            _tspan = convert.(eltype(_p), prob.tspan)
        else
            _tspan = prob.tspan
        end

        if DiffEqBase.isinplace(prob)
            if prob.f isa ODEFunction &&
               (prob.f.f isa FunctionWrappersWrappers.FunctionWrappersWrapper ||
                SciMLBase.specialization(prob.f) === SciMLBase.AutoSpecialize)
                f = ODEFunction{isinplace(prob), SciMLBase.FullSpecialize}(unwrapped_f(prob.f))
                _prob = remake(
                    prob, f = f, u0 = _u0, p = _p, tspan = _tspan, callback = nothing)
            else
                _prob = remake(prob, u0 = _u0, p = _p, tspan = _tspan, callback = nothing)
            end
        else
            _prob = remake(prob, u0 = _u0, p = SciMLStructures.replace(Tunable(), p, _p),
                tspan = _tspan, callback = nothing)
        end

        kwargs_filtered = NamedTuple(filter(x -> x[1] != :sensealg, kwargs))
        sol = solve(_prob, alg, args...; sensealg = DiffEqBase.SensitivityADPassThrough(),
            kwargs_filtered...)
        sol = SciMLBase.sensitivity_solution(sol, state_values(sol), current_time(sol))
        @reset sol.prob = prob
        sol
    end

    out,
    pullback = Mooncake.value_and_pullback!!(
        Mooncake.CoDual(mooncake_adjoint_forwardpass, Mooncake.NoFData()),
        Mooncake.CoDual(u0, Mooncake.zero_rdata(u0)),
        Mooncake.CoDual(tunables, Mooncake.zero_rdata(tunables))
    )

    function mooncake_adjoint_backpass(ybar)
        tmp = if eltype(ybar) <: Number && u0 isa Array
            Array(ybar)
        elseif eltype(ybar) <: Number
            ybar
        elseif ybar isa Tangent
            ut = unthunk.(ybar.u)
            ut_ = map(ut) do u
                (u isa ZeroTangent || u isa NoTangent) ? zero(u0) : u
            end
            reduce(hcat, ut_)
        elseif ybar[1] isa Array
            return Array(ybar)
        else
            tmp = vec(ybar.u[1])
            for i in 2:length(ybar.u)
                tmp = hcat(tmp, vec(ybar.u[i]))
            end
            return reshape(tmp, size(ybar.u[1])..., length(ybar.u))
        end

        _, u0bar, pbar = pullback(tmp)
        _u0bar = u0bar

        if originator isa SciMLBase.TrackerOriginator ||
           originator isa SciMLBase.ReverseDiffOriginator
            (NoTangent(), NoTangent(), _u0bar, pbar, NoTangent(),
                ntuple(_ -> NoTangent(), length(args))...)
        else
            (NoTangent(), NoTangent(), NoTangent(),
                _u0bar, pbar, NoTangent(),
                ntuple(_ -> NoTangent(), length(args))...)
        end
    end

    u = state_values(out)
    SciMLBase.sensitivity_solution(out, u, current_time(out)),
    mooncake_adjoint_backpass
end

end
