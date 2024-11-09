## High level

# Here is where we can add a default algorithm for computing sensitivities
# Based on problem information!

const have_not_warned_vjp = Ref(true)
const STACKTRACE_WITH_VJPWARN = Ref(false)

function inplace_vjp(prob, u0, p, verbose, repack)
    du = zero(u0)

    ez = try
        f = unwrapped_f(prob.f)

        function adfunc(out, u, _p, t)
            f(out, u, repack(_p), t)
            nothing
        end
        Enzyme.autodiff(Enzyme.Reverse, adfunc, Enzyme.Duplicated(du, copy(u0)),
            Enzyme.Duplicated(copy(u0), zero(u0)), Enzyme.Duplicated(copy(p), zero(p)), Enzyme.Const(prob.tspan[1]))
        true
    catch e
        if verbose && have_not_warned_vjp[]
            @warn "Potential performance improvement omitted. EnzymeVJP tried and failed in the automated AD choice algorithm. To show the stack trace, set SciMLSensitivity.STACKTRACE_WITH_VJPWARN[] = true. To turn off this printing, add `verbose = false` to the `solve` call.\n"
            STACKTRACE_WITH_VJPWARN[] && showerror(stderr, e)
            println()
            have_not_warned_vjp[] = false
        end
        false
    end
    if ez
        return EnzymeVJP()
    end

    # Determine if we can compile ReverseDiff
    compile = try
        f = unwrapped_f(prob.f)
        if DiffEqBase.isinplace(prob)
            !hasbranching(f, copy(u0), u0, repack(p), prob.tspan[1])
        else
            !hasbranching(f, u0, repack(p), prob.tspan[1])
        end
    catch
        false
    end

    vjp = try
        f = unwrapped_f(prob.f)
        if p === nothing || p isa SciMLBase.NullParameters
            ReverseDiff.GradientTape((copy(u0), [prob.tspan[1]])) do u, t
                du1 = similar(u, size(u))
                du1 .= 0
                f(du1, u, p, first(t))
                return vec(du1)
            end
        else
            ReverseDiff.GradientTape((copy(u0), p, [prob.tspan[1]])) do u, p, t
                du1 = similar(u, size(u))
                du1 .= 0
                f(du1, u, repack(p), first(t))
                return vec(du1)
            end
        end
        ReverseDiffVJP(compile)
    catch e
        if verbose
            @warn "Potential performance improvement omitted. ReverseDiffVJP tried and failed in the automated AD choice algorithm. To show the stack trace, set SciMLSensitivity.STACKTRACE_WITH_VJPWARN[] = true. To turn off this printing, add `verbose = false` to the `solve` call.\n"
            STACKTRACE_WITH_VJPWARN[] && showerror(stderr, e)
            println()
            have_not_warned_vjp[] = false
        end
        false
    end

    return vjp
end

function automatic_sensealg_choice(
        prob::Union{SciMLBase.AbstractODEProblem,
            SciMLBase.AbstractSDEProblem},
        u0, p, verbose, repack)
    if p === nothing || p isa SciMLBase.NullParameters
        tunables, repack = p, identity
    elseif isscimlstructure(p)
        tunables, repack, _ = canonicalize(Tunable(), p)
    else
        throw(SciMLStructuresCompatibilityError())
    end

    default_sensealg = if p !== SciMLBase.NullParameters() &&
                          !(eltype(u0) <: ForwardDiff.Dual) &&
                          !(eltype(p) <: ForwardDiff.Dual) &&
                          !(eltype(u0) <: Complex) &&
                          !(eltype(p) <: Complex) &&
                          length(u0) + length(tunables) <= 100
        ForwardDiffSensitivity()
    elseif u0 isa GPUArraysCore.AbstractGPUArray || !DiffEqBase.isinplace(prob)
        # only Zygote is GPU compatible and fast
        # so if out-of-place, try Zygote

        vjp = try
            p = prob.p
            y = prob.u0
            f = prob.f
            t = prob.tspan[1]
            λ = zero(prob.u0)

            if p === nothing || p isa SciMLBase.NullParameters
                _dy, back = Zygote.pullback(y) do u
                    vec(f(u, p, t))
                end
                tmp1 = back(λ)
            else
                _dy, back = Zygote.pullback(y, p) do u, p
                    vec(f(u, p, t))
                end
                tmp1, tmp2 = back(λ)
            end
            ZygoteVJP()
        catch e
            if verbose
                @warn "Potential performance improvement omitted. ZygoteVJP tried and failed in the automated AD choice algorithm. To show the stack trace, set SciMLSensitivity.STACKTRACE_WITH_VJPWARN[] = true. To turn off this printing, add `verbose = false` to the `solve` call.\n"
                STACKTRACE_WITH_VJPWARN[] && showerror(stderr, e)
                println()
            end
            false
        end

        if vjp == false
            vjp = try
                if p === nothing || p isa SciMLBase.NullParameters
                    ReverseDiff.gradient((u) -> sum(prob.f(u, p, prob.tspan[1])), u0)
                else
                    ReverseDiff.gradient(
                        (u, _p) -> sum(prob.f(u, repack(_p), prob.tspan[1])), u0, tunables)
                end
                ReverseDiffVJP()
            catch e
                if verbose
                    @warn "Potential performance improvement omitted. ReverseDiffVJP tried and failed in the automated AD choice algorithm. To show the stack trace, set SciMLSensitivity.STACKTRACE_WITH_VJPWARN[] = true. To turn off this printing, add `verbose = false` to the `solve` call.\n"
                    STACKTRACE_WITH_VJPWARN[] && showerror(stderr, e)
                    println()
                end
                false
            end
        end

        if vjp == false
            vjp = try
                p = prob.p
                y = prob.u0
                f = prob.f
                t = prob.tspan[1]
                λ = zero(prob.u0)

                if p === nothing || p isa SciMLBase.NullParameters
                    _dy, back = Tracker.forward(y) do u
                        vec(f(u, p, t))
                    end
                    tmp1 = back(λ)
                else
                    _dy, back = Tracker.forward(y, tunables) do u, tunables
                        vec(f(u, repack(tunables), t))
                    end
                    tmp1, tmp2 = back(λ)
                end
                TrackerVJP()
            catch e
                if verbose
                    @warn "Potential performance improvement omitted. TrackerVJP tried and failed in the automated AD choice algorithm. To show the stack trace, set SciMLSensitivity.STACKTRACE_WITH_VJPWARN[] = true. To turn off this printing, add `verbose = false` to the `solve` call.\n"
                    STACKTRACE_WITH_VJPWARN[] && showerror(stderr, e)
                    println()
                end
                false
            end
        end

        if vjp isa Bool
            if verbose
                @warn "Reverse-Mode AD VJP choices all failed. Falling back to numerical VJPs"
            end

            if p === nothing || p === SciMLBase.NullParameters()
                # QuadratureAdjoint skips all p calculations until the end
                # So it's the fastest when there are no parameters
                QuadratureAdjoint(autodiff = false, autojacvec = vjp)
            elseif prob isa ODEProblem && !(vjp isa TrackerVJP)
                GaussAdjoint(autodiff = false, autojacvec = vjp)
            else
                InterpolatingAdjoint(autodiff = false, autojacvec = vjp)
            end
        else
            if p === nothing || p === SciMLBase.NullParameters()
                # QuadratureAdjoint skips all p calculations until the end
                # So it's the fastest when there are no parameters
                QuadratureAdjoint(autojacvec = vjp)
            elseif prob isa ODEProblem && !(vjp isa TrackerVJP)
                GaussAdjoint(autojacvec = vjp)
            else
                InterpolatingAdjoint(autojacvec = vjp)
            end
        end
    else
        vjp = inplace_vjp(prob, u0, p, verbose, repack)
        if vjp isa Bool
            if verbose
                @warn "Reverse-Mode AD VJP choices all failed. Falling back to numerical VJPs"
            end
            # If reverse-mode isn't working, just fallback to numerical vjps
            if p === nothing || p === SciMLBase.NullParameters()
                QuadratureAdjoint(autodiff = false, autojacvec = vjp)
            elseif prob isa ODEProblem && !(vjp isa TrackerVJP)
                GaussAdjoint(autodiff = false, autojacvec = vjp)
            else
                InterpolatingAdjoint(autodiff = false, autojacvec = vjp)
            end
        else
            if p === nothing || p === SciMLBase.NullParameters()
                QuadratureAdjoint(autojacvec = vjp)
            elseif prob isa ODEProblem && !(vjp isa TrackerVJP)
                GaussAdjoint(autojacvec = vjp)
            else
                InterpolatingAdjoint(autojacvec = vjp)
            end
        end
    end
    return default_sensealg
end

function automatic_sensealg_choice(
        prob::AbstractNonlinearProblem, u0, p,
        verbose, repack)
    default_sensealg = if u0 isa GPUArraysCore.AbstractGPUArray ||
                          !DiffEqBase.isinplace(prob)
        # autodiff = false because forwarddiff fails on many GPU kernels
        # this only effects the Jacobian calculation and is same computation order
        SteadyStateAdjoint(autodiff = false, autojacvec = ZygoteVJP())
    else
        vjp = inplace_vjp(prob, u0, p, verbose, repack)
        SteadyStateAdjoint(autojacvec = vjp)
    end
    return default_sensealg
end

function DiffEqBase._concrete_solve_adjoint(
        prob::Union{SciMLBase.AbstractODEProblem,
            SciMLBase.AbstractSDEProblem,
            SciMLBase.AbstractRODEProblem},
        alg, sensealg::Nothing, u0, p,
        originator::SciMLBase.ADOriginator, args...;
        verbose = true, kwargs...)
    if haskey(kwargs, :callback)
        has_cb = kwargs[:callback] !== nothing
    else
        has_cb = false
    end

    if !(p === nothing || p isa SciMLBase.NullParameters)
        if !isscimlstructure(p) && !isfunctor(p)
            throw(SciMLStructuresCompatibilityError())
        end
    end

    if p === nothing || p isa SciMLBase.NullParameters
        tunables, repack = p, identity
    elseif isscimlstructure(p)
        tunables, repack, aliases = canonicalize(Tunable(), p)
    elseif isfunctor(p)
        tunables, repack = Functors.functor(p)
    else
        throw(SciMLStructuresCompatibilityError())
    end

    default_sensealg = automatic_sensealg_choice(prob, u0, tunables, verbose, repack)
    if has_cb && default_sensealg isa AbstractAdjointSensitivityAlgorithm
        default_sensealg = setvjp(default_sensealg, ReverseDiffVJP())
    end
    DiffEqBase._concrete_solve_adjoint(prob, alg, default_sensealg, u0, p,
        originator, args...; verbose,
        kwargs...)
end

function DiffEqBase._concrete_solve_adjoint(
        prob::AbstractNonlinearProblem, alg,
        sensealg::Nothing, u0, p,
        originator::SciMLBase.ADOriginator, args...;
        verbose = true, kwargs...)
    if !(p === nothing || p isa SciMLBase.NullParameters)
        if !isscimlstructure(p) && !isfunctor(p)
            throw(SciMLStructuresCompatibilityError())
        end
    end

    if p === nothing || p isa SciMLBase.NullParameters
        tunables, repack = p, identity
    elseif isscimlstructure(p)
        tunables, repack, aliases = canonicalize(Tunable(), p)
    else
        tunables, repack = Functors.functor(p)
    end

    default_sensealg = automatic_sensealg_choice(prob, u0, tunables, verbose, repack)
    DiffEqBase._concrete_solve_adjoint(prob, alg, default_sensealg, u0, p,
        originator::SciMLBase.ADOriginator, args...; verbose,
        kwargs...)
end

function DiffEqBase._concrete_solve_adjoint(
        prob::Union{SciMLBase.AbstractDiscreteProblem,
            SciMLBase.AbstractDDEProblem,
            SciMLBase.AbstractSDDEProblem,
            SciMLBase.AbstractDAEProblem},
        alg, sensealg::Nothing,
        u0, p, originator::SciMLBase.ADOriginator,
        args...; kwargs...)
    if length(u0) + length(p) > 100
        default_sensealg = ReverseDiffAdjoint()
    else
        default_sensealg = ForwardDiffSensitivity()
    end
    DiffEqBase._concrete_solve_adjoint(prob, alg, default_sensealg, u0, p,
        originator::SciMLBase.ADOriginator, args...;
        kwargs...)
end

const ADJOINT_STEADY_PROBLEM_ERROR_MESSAGE = """
                                             Chosen adjoint method is not compatible with the chosen problem. NonlinearProblem
                                             and SteadyStateProblem require specific adjoint choices (like SteadyStateAdjoint)
                                             and will not work with adjoints designed for time series models. For more details,
                                             see https://docs.sciml.ai/SciMLSensitivity/stable/.
                                             """

struct AdjointSteadyProblemPairingError <: Exception
    prob::Any
    sensealg::Any
end

function Base.showerror(io::IO, e::AdjointSteadyProblemPairingError)
    println(io, ADJOINT_STEADY_PROBLEM_ERROR_MESSAGE)
    print(io, "Problem type: ")
    println(io, e.prob)
    print(io, "Sensitivity algorithm type: ")
    println(io, e.sensealg)
end

# Also include AbstractForwardSensitivityAlgorithm until a dispatch is made!
function DiffEqBase._concrete_solve_adjoint(
        prob::AbstractNonlinearProblem, alg,
        sensealg::Union{
            AbstractAdjointSensitivityAlgorithm,
            AbstractForwardSensitivityAlgorithm,
            TrackerAdjoint,
            ReverseDiffAdjoint,
            AbstractShadowingSensitivityAlgorithm
        },
        u0, p, originator::SciMLBase.ADOriginator,
        args...; kwargs...)
    throw(AdjointSteadyProblemPairingError(prob, sensealg))
end

function DiffEqBase._concrete_solve_adjoint(
        prob::Union{SciMLBase.AbstractODEProblem,
            SciMLBase.AbstractSDEProblem,
            SciMLBase.AbstractRODEProblem},
        alg,
        sensealg::Union{BacksolveAdjoint,
            QuadratureAdjoint,
            InterpolatingAdjoint,
            GaussAdjoint},
        u0, p, originator::SciMLBase.ADOriginator,
        args...; save_start = true, save_end = true,
        saveat = eltype(prob.tspan)[],
        save_idxs = nothing,
        kwargs...)
    if !(sensealg isa GaussAdjoint) &&
       !(p isa Union{Nothing, SciMLBase.NullParameters, AbstractArray}) ||
       (p isa AbstractArray && !Base.isconcretetype(eltype(p)))
        throw(AdjointSensitivityParameterCompatibilityError())
    end

    if p === nothing || p isa SciMLBase.NullParameters
        tunables, repack = p, identity
    elseif isscimlstructure(p)
        tunables, repack, aliases = canonicalize(Tunable(), p)
    elseif sensealg isa Union{QuadratureAdjoint, GaussAdjoint}
        tunables, repack = Functors.functor(p)
    else
        throw(SciMLStructuresCompatibilityError())
    end
    # Remove saveat, etc. from kwargs since it's handled separately
    # and letting it jump back in there can break the adjoint
    kwargs_prob = NamedTuple(filter(
        x -> x[1] != :saveat && x[1] != :save_start &&
                 x[1] != :save_end && x[1] != :save_idxs,
        prob.kwargs))

    if haskey(kwargs, :callback)
        cb = track_callbacks(CallbackSet(kwargs[:callback]), current_time(prob),
            state_values(prob), parameter_values(prob),
            sensealg)
        _prob = remake(prob; u0 = u0, p = p, kwargs = merge(kwargs_prob, (; callback = cb)))
    else
        cb = nothing
        _prob = remake(prob; u0 = u0, p = p, kwargs = kwargs_prob)
    end

    # Remove callbacks, saveat, etc. from kwargs since it's handled separately
    kwargs_fwd = NamedTuple{Base.diff_names(Base._nt_names(values(kwargs)), (:callback,))}(values(kwargs))

    # Capture the callback_adj for the reverse pass and remove both callbacks
    kwargs_adj = NamedTuple{
        Base.diff_names(Base._nt_names(values(kwargs)),
        (:callback_adj, :callback))}(values(kwargs))
    isq = sensealg isa QuadratureAdjoint
    if sensealg isa BacksolveAdjoint
        sol = solve(_prob, alg, args...; save_noise = true,
            save_start = save_start, save_end = save_end,
            saveat = saveat, kwargs_fwd...)
    elseif ischeckpointing(sensealg)
        sol = solve(_prob, alg, args...; save_noise = true,
            save_start = true, save_end = true,
            saveat = saveat, kwargs_fwd...)
    else
        sol = solve(_prob, alg, args...; save_noise = true, save_start = true,
            save_end = true, kwargs_fwd...)
    end

    # Force `save_start` and `save_end` in the forward pass This forces the
    # solver to do the backsolve all the way back to `u0` Since the start aliases
    # `_prob.u0`, this doesn't actually use more memory But it cleans up the
    # implementation and makes `save_start` and `save_end` arg safe.
    if sensealg isa BacksolveAdjoint
        # Saving behavior unchanged
        ts = current_time(sol)
        only_end = length(ts) == 1 && ts[1] == _prob.tspan[2]
        out = SciMLBase.sensitivity_solution(sol, state_values(sol), ts)
    elseif saveat isa Number
        if _prob.tspan[2] > _prob.tspan[1]
            ts = _prob.tspan[1]:convert(typeof(_prob.tspan[2]), abs(saveat)):_prob.tspan[2]
        else
            ts = _prob.tspan[2]:convert(typeof(_prob.tspan[2]), abs(saveat)):_prob.tspan[1]
        end
        # if _prob.tspan[2]-_prob.tspan[1] is not a multiple of saveat, one looses the last ts value
        last(current_time(sol)) !== ts[end] && (ts = fix_endpoints(sensealg, sol, ts))
        if cb === nothing
            _out = sol(ts)
        else
            _, duplicate_iterator_times = separate_nonunique(current_time(sol))
            _out, ts = out_and_ts(ts, duplicate_iterator_times, sol)
        end

        out = if save_idxs === nothing
            out = SciMLBase.sensitivity_solution(sol, state_values(_out), ts)
        else
            _outf = getu(_out, save_idxs)
            out = SciMLBase.sensitivity_solution(sol, _outf(_out), ts)
        end
        only_end = length(ts) == 1 && ts[1] == _prob.tspan[2]
    elseif isempty(saveat)
        no_start = !save_start
        no_end = !save_end
        sol_idxs = 1:length(sol)
        no_start && (sol_idxs = sol_idxs[2:end])
        no_end && (sol_idxs = sol_idxs[1:(end - 1)])
        only_end = length(sol_idxs) <= 1
        _u = sol.u[sol_idxs]
        u = save_idxs === nothing ? _u : [x[save_idxs] for x in _u]
        ts = current_time(sol, sol_idxs)
        out = SciMLBase.sensitivity_solution(sol, u, ts)
    else
        _saveat = saveat isa Array ? sort(saveat) : saveat # for minibatching
        if cb === nothing
            _saveat = eltype(_saveat) <: typeof(prob.tspan[2]) ?
                      convert.(typeof(_prob.tspan[2]), _saveat) : _saveat
            ts = _saveat
            _out = sol(ts)
        else
            _ts, duplicate_iterator_times = separate_nonunique(current_time(sol))
            _out, ts = out_and_ts(_saveat, duplicate_iterator_times, sol)
        end

        out = if save_idxs === nothing
            out = SciMLBase.sensitivity_solution(sol, state_values(_out), ts)
        else
            _outf = getu(_out, save_idxs)
            out = SciMLBase.sensitivity_solution(sol, _outf(_out), ts)
        end
        only_end = length(ts) == 1 && ts[1] == _prob.tspan[2]
    end

    _save_idxs = save_idxs === nothing ? Colon() : save_idxs

    function adjoint_sensitivity_backpass(Δ)
        function df_iip(_out, u, p, t, i)
            outtype = _out isa SubArray ?
                      ArrayInterface.parameterless_type(_out.parent) :
                      ArrayInterface.parameterless_type(_out)
            if only_end
                eltype(Δ) <: NoTangent && return
                if (Δ isa AbstractArray{<:AbstractArray} || Δ isa AbstractVectorOfArray) &&
                   length(Δ) == 1 && i == 1
                    # user did sol[end] on only_end
                    x = Δ isa AbstractVectorOfArray ? Δ.u[1] : Δ[1]
                    if _save_idxs isa Number
                        vx = vec(x)
                        _out[_save_idxs] .= vx[_save_idxs]
                    elseif _save_idxs isa Colon
                        vec(_out) .= vec(adapt(outtype, x))
                    else
                        vec(@view(_out[_save_idxs])) .= adapt(outtype,
                            vec(x)[_save_idxs])
                    end
                else
                    Δ isa NoTangent && return
                    if _save_idxs isa Number
                        x = vec(Δ)
                        _out[_save_idxs] .= adapt(outtype, @view(x[_save_idxs]))
                    elseif _save_idxs isa Colon
                        vec(_out) .= vec(adapt(outtype, Δ))
                    else
                        x = vec(Δ)
                        vec(@view(_out[_save_idxs])) .= adapt(outtype, @view(x[_save_idxs]))
                    end
                end
            else
                Δu = Δ isa Tangent ? Δ.u : Δ
                !Base.isconcretetype(eltype(Δ)) &&
                    (Δu[i] isa NoTangent || eltype(Δu) <: NoTangent) && return
                if Δ isa AbstractArray{<:AbstractArray} || Δ isa AbstractVectorOfArray ||
                   Δ isa Tangent
                    x = (Δ isa AbstractVectorOfArray || Δ isa Tangent) ? Δ.u[i] : Δ[i]
                    if _save_idxs isa Number
                        _out[_save_idxs] = x[_save_idxs]
                    elseif _save_idxs isa Colon
                        vec(_out) .= (x isa NoTangent || x isa ZeroTangent) ? vec(zero(u)) :
                                     vec(x)
                    else
                        vec(@view(_out[_save_idxs])) .= vec(@view(x[_save_idxs]))
                    end
                else
                    if _save_idxs isa Number
                        _out[_save_idxs] = adapt(outtype,
                            reshape(Δ, prod(size(Δ)[1:(end - 1)]),
                                size(Δ)[end])[_save_idxs,
                                i])
                    elseif _save_idxs isa Colon
                        vec(_out) .= vec(adapt(outtype,
                            reshape(Δ, prod(size(Δ)[1:(end - 1)]),
                                size(Δ)[end])[:, i]))
                    else
                        vec(@view(_out[_save_idxs])) .= vec(adapt(outtype,
                            reshape(Δ,
                                prod(size(Δ)[1:(end - 1)]),
                                size(Δ)[end])[:,
                                i]))
                    end
                end
            end
        end

        function df_oop(u, p, t, i; outtype = nothing)
            if only_end
                eltype(Δ) <: NoTangent && return
                if (Δ isa AbstractArray{<:AbstractArray} || Δ isa AbstractVectorOfArray) &&
                   length(Δ) == 1 && i == 1
                    # user did sol[end] on only_end
                    x = Δ isa AbstractVectorOfArray ? Δ.u[1] : Δ[1]
                    if _save_idxs isa Number
                        vx = vec(x)
                        _out = adapt(outtype, @view(vx[_save_idxs]))
                    elseif _save_idxs isa Colon
                        _out = adapt(outtype, x)
                    else
                        _out = adapt(outtype, vec(x)[_save_idxs])
                    end
                else
                    Δ isa NoTangent && return
                    if _save_idxs isa Number
                        x = vec(Δ)
                        _out = adapt(outtype, @view(x[_save_idxs]))
                    elseif _save_idxs isa Colon
                        _out = adapt(outtype, vec(Δ))
                    else
                        x = vec(Δ)
                        _out = adapt(outtype, @view(x[_save_idxs]))
                    end
                end
            else
                !Base.isconcretetype(eltype(Δ)) &&
                    (Δ[i] isa NoTangent || eltype(Δ) <: NoTangent) && return
                if Δ isa AbstractArray{<:AbstractArray} || Δ isa AbstractVectorOfArray
                    x = Δ isa AbstractVectorOfArray ? Δ.u[i] : Δ[i]
                    if _save_idxs isa Number
                        _out = @view(x[_save_idxs])
                    elseif _save_idxs isa Colon
                        _out = vec(x)
                    else
                        _out = vec(@view(x[_save_idxs]))
                    end
                else
                    if _save_idxs isa Number
                        _out = adapt(outtype,
                            reshape(Δ, prod(size(Δ)[1:(end - 1)]),
                                size(Δ)[end])[_save_idxs,
                                i])
                    elseif _save_idxs isa Colon
                        _out = vec(adapt(outtype,
                            reshape(Δ, prod(size(Δ)[1:(end - 1)]),
                                size(Δ)[end])[:, i]))
                    else
                        _out = vec(adapt(outtype,
                            reshape(Δ,
                                prod(size(Δ)[1:(end - 1)]),
                                size(Δ)[end])[:, i]))
                    end
                end
            end
            return _out
        end

        if haskey(kwargs_adj, :callback_adj)
            cb2 = CallbackSet(cb, kwargs[:callback_adj])
        else
            cb2 = cb
        end
        if ArrayInterface.ismutable(eltype(state_values(sol)))
            du0, dp = adjoint_sensitivities(sol, alg, args...; t = ts,
                dgdu_discrete = df_iip,
                sensealg = sensealg,
                callback = cb2,
                kwargs_adj...)
        else
            du0, dp = adjoint_sensitivities(sol, alg, args...; t = ts,
                dgdu_discrete = df_oop,
                sensealg = sensealg,
                callback = cb2,
                kwargs_adj...)
        end

        du0 = reshape(du0, size(u0))

        dp = p === nothing || p === DiffEqBase.NullParameters() ? nothing :
             dp isa AbstractArray ? reshape(dp', size(tunables)) : dp

        _, repack_adjoint = if p === nothing || p === DiffEqBase.NullParameters() ||
                               !isscimlstructure(p)
            nothing, x -> (x,)
        else
            Zygote.pullback(p) do p
                t, _, _ = canonicalize(Tunable(), p)
                t
            end
        end

        if originator isa SciMLBase.TrackerOriginator ||
           originator isa SciMLBase.ReverseDiffOriginator
            (NoTangent(), NoTangent(), du0, repack_adjoint(dp)[1], NoTangent(),
                ntuple(_ -> NoTangent(), length(args))...)
        else
            (NoTangent(), NoTangent(), NoTangent(),
                du0, repack_adjoint(dp)[1], NoTangent(),
                ntuple(_ -> NoTangent(), length(args))...)
        end
    end
    out, adjoint_sensitivity_backpass
end

# Prefer this route since it works better with callback AD
function DiffEqBase._concrete_solve_adjoint(prob::SciMLBase.AbstractODEProblem, alg,
        sensealg::ForwardSensitivity,
        u0, p, originator::SciMLBase.ADOriginator,
        args...;
        save_idxs = nothing,
        kwargs...)
    if p === nothing || p isa SciMLBase.NullParameters
        tunables, repack = p, identity
    elseif isscimlstructure(p)
        tunables, repack, _ = canonicalize(Tunable(), p)
    else
        throw(SciMLStructuresCompatibilityError())
    end

    if tunables isa AbstractArray && eltype(tunables) <: ForwardDiff.Dual &&
       !(eltype(u0) <: ForwardDiff.Dual)
        # Handle double differentiation case
        u0 = eltype(p).(u0)
    end

    ## Force recompile mode until jvps are specialized to handle this!!!
    _f = if prob.f isa ODEFunction &&
            (prob.f.f isa FunctionWrappersWrappers.FunctionWrappersWrapper ||
             SciMLBase.specialization(prob.f) === SciMLBase.AutoSpecialize)
        ODEFunction{isinplace(prob), SciMLBase.FullSpecialize}(unwrapped_f(prob.f))
    else
        prob.f
    end

    # callback = nothing ensures only the callback in kwargs is used
    _prob = ODEForwardSensitivityProblem(
        _f, u0, prob.tspan, p; sensealg = sensealg, callback = nothing)
    sol = solve(_prob, alg, args...; kwargs...)
    _, du = extract_local_sensitivities(sol, sensealg, Val(true))
    ts = current_time(sol)

    u = if save_idxs === nothing
        uf = getu(sol, 1:length(u0))
        reshape.(uf(sol), Ref(size(u0)))
    else
        uf = getu(sol, _save_idxs)
        uf(sol)
    end
    out = SciMLBase.sensitivity_solution(sol, u, ts)

    if originator isa SciMLBase.EnzymeOriginator
        @reset out.prob = prob
    end

    function forward_sensitivity_backpass(Δ)
        adj = sum(eachindex(du)) do i
            J = du[i]
            if Δ isa AbstractVector
                v = Δ[i]
            elseif Δ isa DESolution || Δ isa AbstractVectorOfArray
                v = Δ.u[i]
            elseif Δ isa AbstractMatrix
                v = @view Δ[:, i]
            else
                v = @view Δ[.., i]
            end
            J'vec(v)
        end

        du0 = @not_implemented("ForwardSensitivity does not differentiate with respect to u0. Change your sensealg.")

        if originator isa SciMLBase.TrackerOriginator ||
           originator isa SciMLBase.ReverseDiffOriginator
            (NoTangent(), NoTangent(), du0, adj, NoTangent(),
                ntuple(_ -> NoTangent(), length(args))...)
        else
            (NoTangent(), NoTangent(), NoTangent(), du0, adj, NoTangent(),
                ntuple(_ -> NoTangent(), length(args))...)
        end
    end
    out, forward_sensitivity_backpass
end

function DiffEqBase._concrete_solve_forward(prob::SciMLBase.AbstractODEProblem, alg,
        sensealg::AbstractForwardSensitivityAlgorithm,
        u0, p, originator::SciMLBase.ADOriginator,
        args...; save_idxs = nothing,
        kwargs...)
    _prob = ODEForwardSensitivityProblem(
        prob.f, u0, prob.tspan, p; sensealg = sensealg, callback = nothing)
    sol = solve(_prob, args...; kwargs...)

    if originator isa SciMLBase.EnzymeOriginator
        @reset sol.prob = prob
    end

    u, du = extract_local_sensitivities(sol, Val(true))
    _save_idxs = save_idxs === nothing ? (1:length(u0)) : save_idxs
    ts = current_time(sol)
    uf = getu(sol, _save_idxs)
    out = SciMLBase.sensitivity_solution(sol,
        ForwardDiff.value.(uf(sol, 1:length(sol))), ts)
    function _concrete_solve_pushforward(Δself, ::Nothing, ::Nothing, x3, Δp, args...)
        x3 !== nothing && error("Pushforward currently requires no u0 derivatives")
        du * Δp
    end
    out, _concrete_solve_pushforward
end

const FORWARDDIFF_SENSITIVITY_PARAMETER_COMPATIBILITY_MESSAGE = """
                                                                ForwardDiffSensitivity assumes the `AbstractArray` interface for `p`. Thus while
                                                                DifferentialEquations.jl can support any parameter struct type, usage
                                                                with ForwardDiffSensitivity requires that `p` could be a valid
                                                                type for being the initial condition `u0` of an array. This means that
                                                                many simple types, such as `Tuple`s and `NamedTuple`s, will work as
                                                                parameters in normal contexts but will fail during ForwardDiffSensitivity
                                                                construction. To work around this issue for complicated cases like nested structs,
                                                                look into defining `p` using `AbstractArray` libraries such as RecursiveArrayTools.jl
                                                                or ComponentArrays.jl.
                                                                """

struct ForwardDiffSensitivityParameterCompatibilityError <: Exception end

function Base.showerror(io::IO, e::ForwardDiffSensitivityParameterCompatibilityError)
    print(io, FORWARDDIFF_SENSITIVITY_PARAMETER_COMPATIBILITY_MESSAGE)
end

# Generic Fallback for ForwardDiff
function DiffEqBase._concrete_solve_adjoint(
        prob::Union{SciMLBase.AbstractODEProblem,
            SciMLBase.AbstractDAEProblem,
            SciMLBase.AbstractDDEProblem,
            SciMLBase.AbstractSDEProblem,
            SciMLBase.AbstractSDDEProblem,
            SciMLBase.AbstractRODEProblem},
        alg,
        sensealg::ForwardDiffSensitivity{CS, CTS},
        u0, p, originator::SciMLBase.ADOriginator,
        args...; saveat = eltype(prob.tspan)[],
        kwargs...) where {CS, CTS}
    if p === nothing || p isa SciMLBase.NullParameters
        tunables, repack = p, identity
    elseif isscimlstructure(p)
        tunables, repack, _ = canonicalize(Tunable(), p)
    else
        throw(SciMLStructuresCompatibilityError())
    end

    if saveat isa Number
        _saveat = prob.tspan[1]:saveat:prob.tspan[2]
    else
        _saveat = saveat
    end

    # use the callback in kwargs, not prob
    sol = solve(remake(prob, p = p, u0 = u0, callback = nothing),
        alg, args...; saveat = _saveat, kwargs...)

    if originator isa SciMLBase.EnzymeOriginator
        @reset sol.prob = prob
    end

    # saveat values
    # need all values here. Not only unique ones.
    # if a callback is saving two times in primal solution, we also need to get it at least
    # two times in the solution using dual numbers.
    ts = current_time(sol)

    function forward_sensitivity_backpass(Δ)
        if !(p === nothing || p === SciMLBase.NullParameters())
            dp = @thunk begin
                chunk_size = if CS === 0 && length(tunables) < 12
                    length(tunables)
                elseif CS !== 0
                    CS
                else
                    12
                end

                num_chunks = length(tunables) ÷ chunk_size
                num_chunks * chunk_size != length(tunables) && (num_chunks += 1)

                pparts = typeof(tunables[1:1])[]
                for j in 0:(num_chunks - 1)
                    local chunk
                    if ((j + 1) * chunk_size) <= length(tunables)
                        chunk = ((j * chunk_size + 1):((j + 1) * chunk_size))
                        pchunk = vec(tunables)[chunk]
                        pdualpart = seed_duals(pchunk, prob.f,
                            ForwardDiff.Chunk{chunk_size}())
                    else
                        chunk = ((j * chunk_size + 1):length(tunables))
                        pchunk = vec(tunables)[chunk]
                        pdualpart = seed_duals(pchunk, prob.f,
                            ForwardDiff.Chunk{length(chunk)}())
                    end

                    pdualvec = if j == 0
                        vcat(pdualpart, tunables[((j + 1) * chunk_size + 1):end])
                    elseif j == num_chunks - 1
                        vcat(tunables[1:(j * chunk_size)], pdualpart)
                    else
                        vcat(tunables[1:(j * chunk_size)], pdualpart,
                            tunables[(((j + 1) * chunk_size) + 1):end])
                    end

                    pdual = SciMLStructures.replace(Tunable(), p, pdualvec)
                    u0dual = convert.(eltype(pdualvec), u0)

                    if (convert_tspan(sensealg) === nothing &&
                        ((haskey(kwargs, :callback) &&
                          has_continuous_callback(kwargs[:callback])))) ||
                       (convert_tspan(sensealg) !== nothing && convert_tspan(sensealg))
                        tspandual = convert.(eltype(pdual), prob.tspan)
                    else
                        tspandual = prob.tspan
                    end

                    ## Force recompile mode because it won't handle the duals
                    ## Would require a manual tag to be applied
                    if prob.f isa ODEFunction
                        if prob.f.jac_prototype !== nothing
                            _f = ODEFunction{SciMLBase.isinplace(prob.f),
                                SciMLBase.FullSpecialize}(unwrapped_f(prob.f),
                                jac_prototype = convert.(eltype(u0dual),
                                    prob.f.jac_prototype))
                        else
                            _f = ODEFunction{SciMLBase.isinplace(prob.f),
                                SciMLBase.FullSpecialize}(unwrapped_f(prob.f))
                        end
                    elseif prob.f isa SDEFunction && prob.f.jac_prototype !== nothing
                        _f = SDEFunction{SciMLBase.isinplace(prob.f),
                            SciMLBase.FullSpecialize}(unwrapped_f(prob.f),
                            jac_prototype = convert.(eltype(u0dual),
                                prob.f.jac_prototype))
                    else
                        _f = prob.f
                    end
                    # use the callback from kwargs, not prob
                    _prob = remake(prob, f = _f, u0 = u0dual, p = pdual,
                        tspan = tspandual, callback = nothing)

                    if _prob isa SDEProblem
                        _prob.noise_rate_prototype !== nothing && (_prob = remake(_prob,
                            noise_rate_prototype = convert.(eltype(pdual),
                                _prob.noise_rate_prototype)))
                    end

                    if saveat isa Number
                        _saveat = prob.tspan[1]:saveat:prob.tspan[2]
                    else
                        _saveat = saveat
                    end

                    _sol = solve(_prob, alg, args...; saveat = ts, kwargs...)
                    _, du = extract_local_sensitivities(_sol, sensealg, Val(true))

                    if haskey(kwargs, :callback)
                        # handle bounds errors: ForwardDiffSensitivity uses dual numbers, so there
                        # can be more or less time points in the primal solution
                        # than in the solution using dual numbers when adaptive solvers are used.
                        # First step: filter all values, so that only time steps that actually occur
                        # in the primal are left. This is for example necessary when `terminate!`
                        # is used.
                        indxs = findall(t -> t ∈ ts, current_time(_sol))
                        _ts = current_time(_sol, indxs)
                        # after this filtering step, we might end up with a too large amount of indices.
                        # For example, if a callback saved values in the primal, then we now potentially
                        # save it by `saveat` and by `save_positions` of the callback.
                        # Second step. Drop these duplicates values.
                        if length(indxs) != length(ts)
                            for i in (length(_ts) - 1):-1:2
                                if _ts[i] == _ts[i + 1] && _ts[i] == _ts[i - 1]
                                    deleteat!(indxs, i)
                                end
                            end
                        end
                        _du = @view du[indxs]
                    else
                        _du = du
                    end

                    _dp = sum(eachindex(_du)) do i
                        J = _du[i]
                        if Δ isa AbstractVector
                            v = Δ[i]
                        elseif Δ isa AbstractVectorOfArray || Δ isa Tangent
                            v = Δ.u[i]
                        elseif Δ isa AbstractMatrix
                            v = @view Δ[:, i]
                        else
                            v = @view Δ[.., i]
                        end
                        if !(Δ isa NoTangent || v isa ZeroTangent)
                            if u0 isa Number
                                ForwardDiff.value.(J'v)
                            elseif v isa Tangent
                                ForwardDiff.value.(J'vec(v.x))
                            else
                                ForwardDiff.value.(J'vec(v))
                            end
                        else
                            zero(p)
                        end
                    end
                    push!(pparts, vec(_dp))
                end
                reduce(vcat, pparts)
            end
        else
            dp = nothing
        end

        du0 = @thunk begin
            chunk_size = if CS === 0 && length(u0) < 12
                length(u0)
            elseif CS !== 0
                CS
            else
                12
            end

            num_chunks = length(u0) ÷ chunk_size
            num_chunks * chunk_size != length(u0) && (num_chunks += 1)

            du0parts = u0 isa Number ? typeof(u0)[] : typeof(u0[1:1])[]

            local _du0

            for j in 0:(num_chunks - 1)
                local chunk
                if u0 isa Number
                    u0dualpart = seed_duals(u0, prob.f,
                        ForwardDiff.Chunk{chunk_size}())
                elseif ((j + 1) * chunk_size) <= length(u0)
                    chunk = ((j * chunk_size + 1):((j + 1) * chunk_size))
                    u0chunk = vec(u0)[chunk]
                    u0dualpart = seed_duals(u0chunk, prob.f,
                        ForwardDiff.Chunk{chunk_size}())
                else
                    chunk = ((j * chunk_size + 1):length(u0))
                    u0chunk = vec(u0)[chunk]
                    u0dualpart = seed_duals(u0chunk, prob.f,
                        ForwardDiff.Chunk{length(chunk)}())
                end

                if u0 isa Number
                    u0dual = u0dualpart
                else
                    u0dualvec = if j == 0
                        vcat(u0dualpart, u0[((j + 1) * chunk_size + 1):end])
                    elseif j == num_chunks - 1
                        vcat(u0[1:(j * chunk_size)], u0dualpart)
                    else
                        vcat(u0[1:(j * chunk_size)], u0dualpart,
                            u0[(((j + 1) * chunk_size) + 1):end])
                    end

                    u0dual = ArrayInterface.restructure(u0, u0dualvec)
                end

                if p === nothing || p === SciMLBase.NullParameters()
                    pdual = tunables
                else
                    pdual = convert.(eltype(u0dual), tunables)
                end

                if (convert_tspan(sensealg) === nothing &&
                    ((haskey(kwargs, :callback) &&
                      has_continuous_callback(kwargs[:callback])))) ||
                   (convert_tspan(sensealg) !== nothing && convert_tspan(sensealg))
                    tspandual = convert.(eltype(pdual), prob.tspan)
                else
                    tspandual = prob.tspan
                end

                ## Force recompile mode because it won't handle the duals
                ## Would require a manual tag to be applied
                if prob.f isa ODEFunction
                    if prob.f.jac_prototype !== nothing
                        _f = ODEFunction{SciMLBase.isinplace(prob.f),
                            SciMLBase.FullSpecialize}(unwrapped_f(prob.f),
                            jac_prototype = convert.(eltype(pdual),
                                prob.f.jac_prototype))
                    else
                        _f = ODEFunction{SciMLBase.isinplace(prob.f),
                            SciMLBase.FullSpecialize}(unwrapped_f(prob.f))
                    end
                elseif prob.f isa SDEFunction && prob.f.jac_prototype !== nothing
                    _f = SDEFunction{SciMLBase.isinplace(prob.f), SciMLBase.FullSpecialize}(
                        unwrapped_f(prob.f),
                        jac_prototype = convert.(eltype(pdual),
                            prob.f.jac_prototype))
                else
                    _f = prob.f
                end

                _p = if p isa SciMLBase.NullParameters
                    p
                else
                    SciMLStructures.replace(Tunable(), p, pdual)
                end

                # use the callback from kwargs, not prob
                _prob = remake(prob, f = _f, u0 = u0dual,
                    p = _p,
                    tspan = tspandual, callback = nothing)

                if _prob isa SDEProblem
                    _prob.noise_rate_prototype !== nothing && (_prob = remake(_prob,
                        noise_rate_prototype = convert.(eltype(pdual),
                            _prob.noise_rate_prototype)))
                end

                if saveat isa Number
                    _saveat = prob.tspan[1]:saveat:prob.tspan[2]
                else
                    _saveat = saveat
                end

                _sol = solve(_prob, alg, args...; saveat = ts, kwargs...)
                _, du = extract_local_sensitivities(_sol, sensealg, Val(true))

                if haskey(kwargs, :callback)
                    # handle bounds errors: ForwardDiffSensitivity uses dual numbers, so there
                    # can be more or less time points in the primal solution
                    # than in the solution using dual numbers when adaptive solvers are used.
                    # First step: filter all values, so that only time steps that actually occur
                    # in the primal are left. This is for example necessary when `terminate!`
                    # is used.
                    indxs = findall(t -> t ∈ ts, current_time(_sol))
                    _ts = current_time(_sol, indxs)
                    # after this filtering step, we might end up with a too large amount of indices.
                    # For example, if a callback saved values in the primal, then we now potentially
                    # save it by `saveat` and by `save_positions` of the callback.
                    # Second step. Drop these duplicates values.
                    if length(indxs) != length(ts)
                        for i in (length(_ts) - 1):-1:2
                            if _ts[i] == _ts[i + 1] && _ts[i] == _ts[i - 1]
                                deleteat!(indxs, i)
                            end
                        end
                    end
                    _du = @view du[indxs]
                else
                    _du = du
                end

                _du0 = sum(eachindex(_du)) do i
                    J = _du[i]
                    if Δ isa AbstractVector
                        v = Δ[i]
                    elseif Δ isa AbstractVectorOfArray || Δ isa Tangent
                        v = Δ.u[i]
                    elseif Δ isa AbstractMatrix
                        v = @view Δ[:, i]
                    else
                        v = @view Δ[.., i]
                    end
                    if !(Δ isa NoTangent || v isa ZeroTangent)
                        if u0 isa Number
                            ForwardDiff.value.(J'v)
                        elseif v isa Tangent
                            ForwardDiff.value.(J'vec(v.x))
                        else
                            ForwardDiff.value.(J'vec(v))
                        end
                    else
                        zero(u0)
                    end
                end

                if !(u0 isa Number)
                    push!(du0parts, vec(_du0))
                end
            end

            if u0 isa Number
                first(_du0)
            else
                ArrayInterface.restructure(u0, reduce(vcat, du0parts))
            end
        end

        _, repack_adjoint = if p === nothing || p === DiffEqBase.NullParameters() ||
                               !isscimlstructure(p)
            nothing, x -> (x,)
        else
            Zygote.pullback(p) do p
                t, _, _ = canonicalize(Tunable(), p)
                t
            end
        end

        if originator isa SciMLBase.TrackerOriginator ||
           originator isa SciMLBase.ReverseDiffOriginator
            (NoTangent(), NoTangent(), unthunk(du0),
                repack_adjoint(unthunk(dp))[1], NoTangent(),
                ntuple(_ -> NoTangent(), length(args))...)
        else
            (NoTangent(), NoTangent(), NoTangent(),
                du0, repack_adjoint(unthunk(dp))[1], NoTangent(),
                ntuple(_ -> NoTangent(), length(args))...)
        end
    end
    sol, forward_sensitivity_backpass
end

function DiffEqBase._concrete_solve_adjoint(
        prob::Union{SciMLBase.AbstractDiscreteProblem,
            SciMLBase.AbstractODEProblem,
            SciMLBase.AbstractDAEProblem,
            SciMLBase.AbstractDDEProblem,
            SciMLBase.AbstractSDEProblem,
            SciMLBase.AbstractSDDEProblem,
            SciMLBase.AbstractRODEProblem
        },
        alg, sensealg::ZygoteAdjoint,
        u0, p, originator::SciMLBase.ADOriginator,
        args...; kwargs...)
    kwargs_filtered = NamedTuple(filter(x -> x[1] != :sensealg, kwargs))
    Zygote.pullback(
        (u0, p) -> solve(prob, alg, args...; u0 = u0, p = p,
            sensealg = SensitivityADPassThrough(),
            kwargs_filtered...),
        u0,
        p)
end

# NOTE: This is needed to prevent a method ambiguity error
function DiffEqBase._concrete_solve_adjoint(
        prob::AbstractNonlinearProblem, alg, sensealg::ZygoteAdjoint,
        u0, p, originator::SciMLBase.ADOriginator,
        args...; kwargs...)
    kwargs_filtered = NamedTuple(filter(x -> x[1] != :sensealg, kwargs))
    Zygote.pullback(
        (u0, p) -> solve(prob, alg, args...; u0 = u0, p = p,
            sensealg = SensitivityADPassThrough(),
            kwargs_filtered...),
        u0,
        p)
end

const ENZYME_TRACKED_REAL_ERROR_MESSAGE = """
                                             `Enzyme` is not compatible with `ReverseDiffAdjoint` nor with `TrackerAdjoint`.
                                             Either choose a different adjoint method like `GaussAdjoint`,
                                             or use a different AD system like `ReverseDiff`.
                                             For more details, on these methods see
                                             https://docs.sciml.ai/SciMLSensitivity/stable/.
                                             """

struct EnzymeTrackedRealError <: Exception
end

function Base.showerror(io::IO, e::EnzymeTrackedRealError)
    println(io, ENZYME_TRACKED_REAL_ERROR_MESSAGE)
end

function DiffEqBase._concrete_solve_adjoint(
        prob::Union{SciMLBase.AbstractDiscreteProblem,
            SciMLBase.AbstractODEProblem,
            SciMLBase.AbstractDAEProblem,
            SciMLBase.AbstractDDEProblem,
            SciMLBase.AbstractSDEProblem,
            SciMLBase.AbstractSDDEProblem,
            SciMLBase.AbstractRODEProblem},
        alg, sensealg::TrackerAdjoint,
        u0, p, originator::SciMLBase.ADOriginator,
        args...;
        kwargs...)
    local sol
    if originator isa SciMLBase.EnzymeOriginator
        throw(EnzymeTrackedRealError())
    end

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

    function tracker_adjoint_forwardpass(_u0, _p)
        if (convert_tspan(sensealg) === nothing &&
            ((haskey(kwargs, :callback) && has_continuous_callback(kwargs[:callback])))) ||
           (convert_tspan(sensealg) !== nothing && convert_tspan(sensealg))
            _tspan = convert.(eltype(_p), prob.tspan)
        else
            _tspan = prob.tspan
        end

        if DiffEqBase.isinplace(prob)
            # use Array{TrackedReal} for mutation to work
            # Recurse to all Array{TrackedArray}

            ## Force recompile mode because it's required for the tracked type handling
            if prob.f isa ODEFunction &&
               (prob.f.f isa FunctionWrappersWrappers.FunctionWrappersWrapper ||
                SciMLBase.specialization(prob.f) === SciMLBase.AutoSpecialize)
                f = ODEFunction{isinplace(prob), SciMLBase.FullSpecialize}(unwrapped_f(prob.f))
                _prob = remake(prob, f = f, u0 = map(identity, _u0),
                    p = _p, tspan = _tspan, callback = nothing)
            else
                _prob = remake(prob, u0 = map(identity, _u0), p = _p,
                    tspan = _tspan, callback = nothing)
            end
        else
            # use TrackedArray for efficiency of the tape
            if prob isa
               Union{SciMLBase.AbstractDDEProblem, SciMLBase.AbstractDAEProblem,
                SciMLBase.AbstractSDDEProblem}
                _f = function (u, p, h, t) # For DDE, but also works for (du,u,p,t) DAE
                    out = prob.f(u, p, h, t)
                    if out isa TrackedArray
                        return out
                    else
                        Tracker.collect(out)
                    end
                end

                # Only define `g` for the stochastic ones
                if prob isa SciMLBase.AbstractSDEProblem
                    _g = function (u, p, h, t)
                        out = prob.g(u, p, h, t)
                        if out isa TrackedArray
                            return out
                        else
                            Tracker.collect(out)
                        end
                    end
                    _prob = remake(prob,
                        f = ArrayInterface.parameterless_type(prob.f){false,
                            SciMLBase.FullSpecialize
                        }(_f,
                            _g),
                        u0 = _u0, p = SciMLStructures.replace(Tunable(), p, _p),
                        tspan = _tspan, callback = nothing)
                else
                    _prob = remake(prob,
                        f = ArrayInterface.parameterless_type(prob.f){false,
                            SciMLBase.FullSpecialize
                        }(_f),
                        u0 = _u0, p = SciMLStructures.replace(Tunable(), p, _p),
                        tspan = _tspan, callback = nothing)
                end
            elseif prob isa
                   Union{SciMLBase.AbstractODEProblem, SciMLBase.AbstractSDEProblem}
                _f = function (u, p, t)
                    out = prob.f(u, p, t)
                    if out isa TrackedArray
                        return out
                    else
                        Tracker.collect(out)
                    end
                end
                if prob isa SciMLBase.AbstractSDEProblem
                    _g = function (u, p, t)
                        out = prob.g(u, p, t)
                        if out isa TrackedArray
                            return out
                        else
                            Tracker.collect(out)
                        end
                    end
                    _prob = remake(prob,
                        f = ArrayInterface.parameterless_type(prob.f){false,
                            SciMLBase.FullSpecialize
                        }(_f,
                            _g),
                        u0 = _u0, p = SciMLStructures.replace(Tunable(), p, _p),
                        tspan = _tspan, callback = nothing)
                else
                    _prob = remake(prob,
                        f = ArrayInterface.parameterless_type(prob.f){false,
                            SciMLBase.FullSpecialize
                        }(_f),
                        u0 = _u0, p = SciMLStructures.replace(Tunable(), p, _p),
                        tspan = _tspan, callback = nothing)
                end
            else
                error("TrackerAdjont does not currently support the specified problem type. Please open an issue.")
            end
        end

        kwargs_filtered = NamedTuple(filter(x -> x[1] != :sensealg, kwargs))
        sol = solve(_prob, alg, args...; sensealg = DiffEqBase.SensitivityADPassThrough(),
            kwargs_filtered...)
        sol = SciMLBase.sensitivity_solution(sol, state_values(sol), current_time(sol))
        @reset sol.prob = prob

        if state_values(sol, 1) isa Array
            return Array(sol)
        else
            tmp = vec(state_values(sol, 1))
            for i in 2:length(sol)
                tmp = hcat(tmp, vec(state_values(sol, i)))
            end
            return reshape(tmp, size(sol))
        end
        #adapt(typeof(u0),arr)
        sol
    end

    out, pullback = Tracker.forward(tracker_adjoint_forwardpass, u0, tunables)
    function tracker_adjoint_backpass(ybar)
        tmp = if eltype(ybar) <: Number && u0 isa Array
            Array(ybar) # can also be a ODESolution
        elseif eltype(ybar) <: Number # CuArray{Floats}
            ybar
        elseif ybar[1] isa Array
            return Array(ybar)
        else
            tmp = vec(ybar.u[1])
            for i in 2:length(ybar.u)
                tmp = hcat(tmp, vec(ybar.u[i]))
            end
            return reshape(tmp, size(ybar.u[1])..., length(ybar.u))
        end
        u0bar, pbar = pullback(tmp)
        _u0bar = u0bar isa Tracker.TrackedArray ? Tracker.data(u0bar) : Tracker.data.(u0bar)

        if originator isa SciMLBase.TrackerOriginator ||
           originator isa SciMLBase.ReverseDiffOriginator
            (NoTangent(), NoTangent(), _u0bar, Tracker.data(pbar), NoTangent(),
                ntuple(_ -> NoTangent(), length(args))...)
        else
            (NoTangent(), NoTangent(), NoTangent(),
                _u0bar, Tracker.data(pbar), NoTangent(),
                ntuple(_ -> NoTangent(), length(args))...)
        end
    end

    u = u0 isa Tracker.TrackedArray ? Tracker.data.(state_values(sol)) :
        Tracker.data.(Tracker.data.(state_values(sol)))
    SciMLBase.sensitivity_solution(sol, u, Tracker.data.(current_time(sol))),
    tracker_adjoint_backpass
end

const REVERSEDIFF_ADJOINT_GPU_COMPATIBILITY_MESSAGE = """
                                                      ReverseDiffAdjoint is not compatible GPU-based array types. Use a different
                                                      sensitivity analysis method, like InterpolatingAdjoint or TrackerAdjoint,
                                                      in order to combine with GPUs.
                                                      """

struct ReverseDiffGPUStateCompatibilityError <: Exception end

function Base.showerror(io::IO, e::ReverseDiffGPUStateCompatibilityError)
    print(io, FORWARDDIFF_SENSITIVITY_PARAMETER_COMPATIBILITY_MESSAGE)
end

const SCIMLSTRUCTURES_ERROR_MESSAGE = """
                                         `p` is not a SciMLStructure. This is required for adjoint sensitivity analysis. For more information,
                                         see the documentation on SciMLStructures.jl for the definition of the SciMLStructures interface.
                                         In particular, adjoint sensitivities only applies to `Tunable`.
                                         """

struct SciMLStructuresCompatibilityError <: Exception
end

function Base.showerror(io::IO, e::SciMLStructuresCompatibilityError)
    println(io, SCIMLSTRUCTURES_ERROR_MESSAGE)
end

function DiffEqBase._concrete_solve_adjoint(
        prob::Union{SciMLBase.AbstractDiscreteProblem,
            SciMLBase.AbstractODEProblem,
            SciMLBase.AbstractDAEProblem,
            SciMLBase.AbstractDDEProblem,
            SciMLBase.AbstractSDEProblem,
            SciMLBase.AbstractSDDEProblem,
            SciMLBase.AbstractRODEProblem},
        alg, sensealg::ReverseDiffAdjoint,
        u0, p, originator::SciMLBase.ADOriginator,
        args...; kwargs...)
    if typeof(u0) isa GPUArraysCore.AbstractGPUArray
        throw(ReverseDiffGPUStateCompatibilityError())
    end

    if !(u0 isa AbstractVector)
        error("Sensitivity algorithm ReverseDiffAdjoint only supports vector u0")
    end

    if originator isa SciMLBase.EnzymeOriginator
        throw(EnzymeTrackedRealError())
    end

    t = eltype(prob.tspan)[]
    u = typeof(u0)[]

    local sol

    function reversediff_adjoint_forwardpass(_u0, _p)
        if (convert_tspan(sensealg) === nothing &&
            ((haskey(kwargs, :callback) && has_continuous_callback(kwargs[:callback])))) ||
           (convert_tspan(sensealg) !== nothing && convert_tspan(sensealg))
            _tspan = convert.(eltype(_p), prob.tspan)
        else
            _tspan = prob.tspan
        end

        if DiffEqBase.isinplace(prob)
            # use Array{TrackedReal} for mutation to work
            # Recurse to all Array{TrackedArray}

            ## Force recompile mode because it's required for the tracked type handling
            if prob.f isa ODEFunction &&
               (prob.f.f isa FunctionWrappersWrappers.FunctionWrappersWrapper ||
                SciMLBase.specialization(prob.f) === SciMLBase.AutoSpecialize)
                f = ODEFunction{isinplace(prob), SciMLBase.FullSpecialize}(unwrapped_f(prob.f))
                _prob = remake(prob, f = f, u0 = reshape([x for x in _u0], size(_u0)),
                    p = _p,
                    tspan = _tspan,
                    callback = nothing)
            else
                _prob = remake(prob, u0 = reshape([x for x in _u0], size(_u0)), p = _p,
                    tspan = _tspan)
            end
        else
            # use TrackedArray for efficiency of the tape
            _f(args...) = ArrayInterface.aos_to_soa(prob.f(args...))
            if prob isa SDEProblem
                _g(args...) = ArrayInterface.aos_to_soa(prob.g(args...))
                _prob = remake(prob,
                    f = ArrayInterface.parameterless_type(prob.f){
                        SciMLBase.isinplace(prob),
                        true}(_f,
                        _g),
                    u0 = _u0, p = _p, tspan = _tspan, callback = nothing)
            else
                _prob = remake(prob,
                    f = ArrayInterface.parameterless_type(prob.f){
                        SciMLBase.isinplace(prob),
                        true}(_f),
                    u0 = _u0, p = _p, tspan = _tspan, callback = nothing)
            end
        end

        kwargs_filtered = NamedTuple(filter(x -> x[1] != :sensealg, kwargs))
        sol = solve(_prob, alg, args...; sensealg = DiffEqBase.SensitivityADPassThrough(),
            kwargs_filtered...)
        t = current_time(sol)
        if DiffEqBase.isinplace(prob)
            u = map.(ReverseDiff.value, state_values(sol))
        else
            u = map(ReverseDiff.value, state_values(sol))
        end
        sol = SciMLBase.sensitivity_solution(sol, state_values(sol), t)
        Array(sol)
    end

    tape = ReverseDiff.GradientTape(reversediff_adjoint_forwardpass, (u0, p))
    tu, tp = ReverseDiff.input_hook(tape)
    output = ReverseDiff.output_hook(tape)
    ReverseDiff.value!(tu, u0)
    p isa SciMLBase.NullParameters || ReverseDiff.value!(tp, p)
    ReverseDiff.forward_pass!(tape)

    function reversediff_adjoint_backpass(ybar)
        _ybar = if ybar isa AbstractVectorOfArray
            Array(ybar)
        elseif eltype(ybar) <: AbstractArray
            Array(VectorOfArray(ybar))
        else
            ybar
        end
        ReverseDiff.increment_deriv!(output, _ybar)
        ReverseDiff.reverse_pass!(tape)

        if originator isa SciMLBase.TrackerOriginator ||
           originator isa SciMLBase.ReverseDiffOriginator
            (NoTangent(), NoTangent(), ReverseDiff.deriv(tu), ReverseDiff.deriv(tp),
                NoTangent(), ntuple(_ -> NoTangent(), length(args))...)
        else
            (NoTangent(), NoTangent(), NoTangent(), ReverseDiff.deriv(tu),
                ReverseDiff.deriv(tp), NoTangent(),
                ntuple(_ -> NoTangent(), length(args))...)
        end
    end
    u = u0 isa ReverseDiff.TrackedArray ? ReverseDiff.value(state_values(sol)) :
        ReverseDiff.value.(state_values(sol))
    SciMLBase.sensitivity_solution(sol, u, ReverseDiff.value.(current_time(sol))),
    reversediff_adjoint_backpass
end

function DiffEqBase._concrete_solve_adjoint(prob::SciMLBase.AbstractODEProblem, alg,
        sensealg::AbstractShadowingSensitivityAlgorithm,
        u0, p, originator::SciMLBase.ADOriginator,
        args...; save_start = true, save_end = true,
        saveat = eltype(prob.tspan)[],
        save_idxs = nothing,
        kwargs...)
    if haskey(kwargs, :callback)
        error("Sensitivity analysis based on Least Squares Shadowing is not compatible with callbacks. Please select another `sensealg`.")
    else
        _prob = remake(prob, f = unwrapped_f(prob.f), u0 = u0, p = p)
    end

    sol = solve(_prob, alg, args...; save_start = save_start, save_end = save_end,
        saveat = saveat, kwargs...)

    if saveat isa Number
        if _prob.tspan[2] > _prob.tspan[1]
            ts = _prob.tspan[1]:convert(typeof(_prob.tspan[2]), abs(saveat)):_prob.tspan[2]
        else
            ts = _prob.tspan[2]:convert(typeof(_prob.tspan[2]), abs(saveat)):_prob.tspan[1]
        end
        _out = sol(ts)
        out = if save_idxs === nothing
            out = SciMLBase.sensitivity_solution(
                sol, state_values(_out), current_time(sol))
        else
            outuf = getu(_out, save_idxs)
            out = SciMLBase.sensitivity_solution(sol, outuf(_out, 1:length(_out)), ts)
        end
        # only_end
        (length(ts) == 1 && ts[1] == _prob.tspan[2]) &&
            error("Sensitivity analysis based on Least Squares Shadowing requires a long-time averaged quantity.")
    elseif isempty(saveat)
        no_start = !save_start
        no_end = !save_end
        sol_idxs = 1:length(sol)
        no_start && (sol_idxs = sol_idxs[2:end])
        no_end && (sol_idxs = sol_idxs[1:(end - 1)])
        only_end = length(sol_idxs) <= 1
        _u = state_values(sol, sol_idxs)
        u = save_idxs === nothing ? _u : [x[save_idxs] for x in _u]
        ts = current_time(sol, sol_idxs)
        out = SciMLBase.sensitivity_solution(sol, u, ts)
    else
        _saveat = saveat isa Array ? sort(saveat) : saveat # for minibatching
        ts = _saveat
        _out = sol(ts)

        out = if save_idxs === nothing
            out = SciMLBase.sensitivity_solution(sol, state_values(_out), ts)
        else
            outuf = getu(_out, save_idxs)
            out = SciMLBase.sensitivity_solution(sol, outuf(_out, 1:length(_out)), ts)
        end
        # only_end
        (length(ts) == 1 && ts[1] == _prob.tspan[2]) &&
            error("Sensitivity analysis based on Least Squares Shadowing requires a long-time averaged quantity.")
    end

    _save_idxs = save_idxs === nothing ? Colon() : save_idxs

    function adjoint_sensitivity_backpass(Δ)
        function df(_out, u, p, t, i)
            if Δ isa AbstractArray{<:AbstractArray} || Δ isa AbstractVectorOfArray
                x = Δ isa AbstractVectorOfArray ? Δ.u[i] : Δ[i]
                if _save_idxs isa Number
                    _out[_save_idxs] = x[_save_idxs]
                elseif _save_idxs isa Colon
                    vec(_out) .= vec(x)
                else
                    vec(@view(_out[_save_idxs])) .= vec(x[_save_idxs])
                end
            else
                if _save_idxs isa Number
                    _out[_save_idxs] = adapt(ArrayInterface.parameterless_type(u0),
                        reshape(Δ, prod(size(Δ)[1:(end - 1)]),
                            size(Δ)[end])[_save_idxs, i])
                elseif _save_idxs isa Colon
                    vec(_out) .= vec(adapt(ArrayInterface.parameterless_type(u0),
                        reshape(Δ, prod(size(Δ)[1:(end - 1)]),
                            size(Δ)[end])[:, i]))
                else
                    vec(@view(_out[_save_idxs])) .= vec(adapt(
                        ArrayInterface.parameterless_type(u0),
                        reshape(Δ,
                            prod(size(Δ)[1:(end - 1)]),
                            size(Δ)[end])[:, i]))
                end
            end
        end

        if sensealg isa ForwardLSS
            lss_problem = ForwardLSSProblem(sol, sensealg, t = ts, dgdu_discrete = df)
            dp = shadow_forward(lss_problem)
        elseif sensealg isa AdjointLSS
            adjointlss_problem = AdjointLSSProblem(sol, sensealg, t = ts,
                dgdu_discrete = df)
            dp = shadow_adjoint(adjointlss_problem)
        elseif sensealg isa NILSS
            nilss_prob = NILSSProblem(_prob, sensealg, t = ts, dgdu_discrete = df)
            dp = shadow_forward(nilss_prob, alg)
        elseif sensealg isa NILSAS
            nilsas_prob = NILSASProblem(_prob, sensealg, t = ts, dgdu_discrete = df)
            dp = shadow_adjoint(nilsas_prob, alg)
        else
            error("No concrete_solve implementation found for sensealg `$sensealg`. Did you spell the sensitivity algorithm correctly? Please report this error.")
        end

        if originator isa SciMLBase.TrackerOriginator ||
           originator isa SciMLBase.ReverseDiffOriginator
            (NoTangent(), NoTangent(), NoTangent(), dp, NoTangent(),
                ntuple(_ -> NoTangent(), length(args))...)
        else
            (NoTangent(), NoTangent(), NoTangent(), NoTangent(), dp, NoTangent(),
                ntuple(_ -> NoTangent(), length(args))...)
        end
    end
    out, adjoint_sensitivity_backpass
end

function DiffEqBase._concrete_solve_adjoint(
        prob::AbstractNonlinearProblem,
        alg, sensealg::SteadyStateAdjoint,
        u0, p, originator::SciMLBase.ADOriginator,
        args...; save_idxs = nothing, kwargs...)
    _prob = remake(prob, u0 = u0, p = p)
    sol = solve(_prob, alg, args...; kwargs...)
    _save_idxs = save_idxs === nothing ? Colon() : save_idxs

    if save_idxs === nothing
        out = sol
    else
        out = SciMLBase.sensitivity_solution(sol, sol[_save_idxs])
    end

    function steadystatebackpass(Δ)
        # Δ = dg/dx or diffcache.dg_val
        # del g/del p = 0
        function df(_out, u, p, t, i)
            if _save_idxs isa Number
                _out[_save_idxs] = Δ[_save_idxs]
            elseif Δ isa Number
                @. _out[_save_idxs] = Δ
            else
                @. _out[_save_idxs] = Δ[_save_idxs]
            end
        end
        dp = adjoint_sensitivities(sol, alg; sensealg = sensealg, dgdu = df)

        if originator isa SciMLBase.TrackerOriginator ||
           originator isa SciMLBase.ReverseDiffOriginator
            (NoTangent(), NoTangent(), NoTangent(), dp, NoTangent(),
                ntuple(_ -> NoTangent(), length(args))...)
        else
            (NoTangent(), NoTangent(), NoTangent(), NoTangent(), dp, NoTangent(),
                ntuple(_ -> NoTangent(), length(args))...)
        end
    end
    out, steadystatebackpass
end

function fix_endpoints(sensealg, sol, ts)
    @warn "Endpoints do not match. Return code: $(sol.retcode). Likely your time range is not a multiple of `saveat`. sol.t[end]: $(last(current_time(sol))), ts[end]: $(ts[end])"
    ts = collect(ts)
    push!(ts, last(current_time(sol)))
end
