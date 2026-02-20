# Not in FiniteDiff because `u` -> scalar isn't used anywhere else,
# but could be upstreamed.
mutable struct UGradientWrapper{fType, tType, P} <: Function
    f::fType
    t::tType
    p::P
end

(ff::UGradientWrapper)(uprev) = ff.f(uprev, ff.p, ff.t)

mutable struct ParamGradientWrapper{fType, tType, uType} <: Function
    f::fType
    t::tType
    u::uType
end

(ff::ParamGradientWrapper)(p) = ff.f(ff.u, p, ff.t)

# the next four definitions are only needed in case of non-diagonal SDEs

mutable struct ParamNonDiagNoiseGradientWrapper{fType, tType, uType} <: Function
    f::fType
    t::tType
    u::uType
end

(ff::ParamNonDiagNoiseGradientWrapper)(p) = vec(ff.f(ff.u, p, ff.t))

mutable struct ParamNonDiagNoiseJacobianWrapper{fType, tType, uType, duType} <: Function
    f::fType
    t::tType
    u::uType
    du::duType
end

function (ff::ParamNonDiagNoiseJacobianWrapper)(p)
    du1 = similar(p, size(ff.du))
    du1 .= 0
    ff.f(du1, ff.u, p, ff.t)
    return vec(du1)
end

function (ff::ParamNonDiagNoiseJacobianWrapper)(du1, p)
    ff.f(du1, ff.u, p, ff.t)
    return vec(du1)
end

mutable struct UNonDiagNoiseGradientWrapper{fType, tType, P} <: Function
    f::fType
    t::tType
    p::P
end

(ff::UNonDiagNoiseGradientWrapper)(uprev) = vec(ff.f(uprev, ff.p, ff.t))

mutable struct UNonDiagNoiseJacobianWrapper{fType, tType, P, duType} <: Function
    f::fType
    t::tType
    p::P
    du::duType
end

function (ff::UNonDiagNoiseJacobianWrapper)(uprev)
    (du1 = similar(ff.du); ff.f(du1, uprev, ff.p, ff.t); vec(du1))
end

function (ff::UNonDiagNoiseJacobianWrapper)(du1, uprev)
    ff.f(du1, uprev, ff.p, ff.t)
    return vec(du1)
end

# RODE wrappers
mutable struct RODEUJacobianWrapper{fType, tType, P, WType} <: Function
    f::fType
    t::tType
    p::P
    W::WType
end

(ff::RODEUJacobianWrapper)(du1, uprev) = ff.f(du1, uprev, ff.p, ff.t, ff.W)
function (ff::RODEUJacobianWrapper)(uprev)
    (du1 = similar(uprev); ff.f(du1, uprev, ff.p, ff.t, ff.W); du1)
end

mutable struct RODEUDerivativeWrapper{F, tType, P, WType} <: Function
    f::F
    t::tType
    p::P
    W::WType
end
(ff::RODEUDerivativeWrapper)(u) = ff.f(u, ff.p, ff.t, ff.W)

mutable struct RODEParamGradientWrapper{fType, tType, uType, WType} <: Function
    f::fType
    t::tType
    u::uType
    W::WType
end

(ff::RODEParamGradientWrapper)(p) = ff.f(ff.u, p, ff.t, ff.W)

mutable struct RODEParamJacobianWrapper{fType, tType, uType, WType} <: Function
    f::fType
    t::tType
    u::uType
    W::WType
end

(ff::RODEParamJacobianWrapper)(du1, p) = ff.f(du1, ff.u, p, ff.t, ff.W)

function (ff::RODEParamJacobianWrapper)(p)
    du1 = similar(p, size(ff.u))
    ff.f(du1, ff.u, p, ff.t, ff.W)
    return du1
end

function determine_chunksize(u, alg::AbstractOverloadingSensitivityAlgorithm)
    return determine_chunksize(u, get_chunksize(alg))
end

function determine_chunksize(u, CS)
    if CS != 0
        return CS
    else
        return ForwardDiff.pickchunksize(length(u))
    end
end

function jacobian(
        f, x::AbstractArray{<:Number},
        alg::AbstractOverloadingSensitivityAlgorithm
    )
    if alg_autodiff(alg)
        uf = unwrapped_f(f)
        J = ForwardDiff.jacobian(uf, x)
    else
        T = if f isa ParamGradientWrapper
            promote_type(eltype(f.u), eltype(x))
        elseif f isa UGradientWrapper
            promote_type(eltype(f.p), eltype(x))
        else
            T = eltype(x)
        end
        J = FiniteDiff.finite_difference_jacobian(f, x, Val(:forward), T)
    end
    return J
end

function jacobian!(
        J::Nothing, f, x::AbstractArray{<:Number},
        fx::Union{Nothing, AbstractArray{<:Number}},
        alg::AbstractOverloadingSensitivityAlgorithm, jac_config::Nothing
    )
    @assert isempty(x)
    return J
end
function jacobian!(J::PreallocationTools.DiffCache, x::SciMLBase.UJacobianWrapper, args...)
    return jacobian!(J.du, x, args...)
end
function jacobian!(
        J::AbstractMatrix{<:Number}, f, x::AbstractArray{<:Number},
        fx::Union{Nothing, AbstractArray{<:Number}},
        alg::AbstractOverloadingSensitivityAlgorithm, jac_config
    )
    if alg_autodiff(alg)
        uf = unwrapped_f(f)
        if fx === nothing
            ForwardDiff.jacobian!(J, uf, x)
        else
            ForwardDiff.jacobian!(J, uf, fx, x, jac_config)
        end
    else
        FiniteDiff.finite_difference_jacobian!(J, f, x, jac_config)
    end
    return nothing
end

function derivative!(
        df::AbstractArray{<:Number}, f,
        x::Number,
        alg::AbstractOverloadingSensitivityAlgorithm, der_config
    )
    if alg_autodiff(alg)
        ForwardDiff.derivative!(df, f, x) # der_config doesn't work
    else
        FiniteDiff.finite_difference_derivative!(df, f, x, der_config)
    end
    return nothing
end

function gradient!(
        df::AbstractArray{<:Number}, f,
        x::Union{Number, AbstractArray{<:Number}},
        alg::AbstractOverloadingSensitivityAlgorithm, grad_config
    )
    if alg_autodiff(alg)
        ForwardDiff.gradient!(df, f, x, grad_config)
    else
        FiniteDiff.finite_difference_gradient!(df, f, x, grad_config)
    end
    return nothing
end

"""
jacobianvec!(Jv, f, x, v, alg, (buffer, seed)) -> nothing

``Jv <- J(f(x))v``
"""
function jacobianvec!(
        Jv::AbstractArray{<:Number}, f, x::AbstractArray{<:Number},
        v, alg::AbstractOverloadingSensitivityAlgorithm, config
    )
    if alg_autodiff(alg)
        buffer, seed = config
        TD = typeof(first(seed))
        T = typeof(first(seed).partials)
        @.. seed = TD(x, T(tuple(v)))
        uf = unwrapped_f(f)
        uf(buffer, seed)
        Jv .= ForwardDiff.partials.(buffer, 1)
    else
        buffer1, buffer2 = config
        f(buffer1, x)
        T = eltype(x)
        # Should it be min? max? mean?
        ϵ = sqrt(eps(real(T))) * max(one(real(T)), abs(norm(x)))
        @. x += ϵ * v
        f(buffer2, x)
        @. x -= ϵ * v
        @. Jv = (buffer2 - buffer1) / ϵ
    end
    return nothing
end
function jacobianmat!(
        JM::AbstractMatrix{<:Number}, f, x::AbstractArray{<:Number},
        M, alg::AbstractOverloadingSensitivityAlgorithm, config
    )
    buffer, seed = config
    T = eltype(seed)
    numparams = length(ForwardDiff.partials(seed[1]))
    for i in eachindex(seed)
        seed[i] = T(x[i], ForwardDiff.Partials(ntuple(j -> M[i, j], numparams)))
    end
    f(buffer, seed)
    for (j, dual) in enumerate(buffer)
        for (i, partial) in enumerate(ForwardDiff.partials(dual))
            JM[j, i] = partial
        end
    end
    return nothing
end
function vecjacobian!(
        dλ, y, λ, p, t, S::TS;
        dgrad = nothing, dy = nothing,
        W = nothing
    ) where {TS <: SensitivityFunction}
    _vecjacobian!(dλ, y, λ, p, t, S, S.sensealg.autojacvec, dgrad, dy, W)
    return
end

function vecjacobian(
        y, λ, p, t, S::TS;
        dgrad = nothing, dy = nothing,
        W = nothing
    ) where {TS <: SensitivityFunction}
    return _vecjacobian(y, λ, p, t, S, S.sensealg.autojacvec, dgrad, dy, W)
end

function _vecjacobian!(
        dλ, y, λ, p, t, S::TS, isautojacvec::Bool, dgrad, dy,
        W
    ) where {TS <: SensitivityFunction}
    (; sensealg, f) = S
    prob = getprob(S)

    (; J, uf, f_cache, jac_config) = S.diffcache

    if J isa DiffCache && dλ !== nothing
        J = get_tmp(J, dλ)
    end

    if !(prob isa AbstractNonlinearProblem) && dλ !== nothing
        if W === nothing
            if SciMLBase.has_jac(f)
                f.jac(J, y, p, t) # Calculate the Jacobian into J
            else
                if typeof(t) !== typeof(uf.t)
                    # Handle the case of ForwardDiff.Dual from Rosenbrock
                    _uf = SciMLBase.UJacobianWrapper(f, t, p)
                    # This is really slow and allocates, but it's a fallback only for a
                    # rare case so it can be optimized in the future
                    _f_cache = DiffEqBase.isinplace(prob) ? deepcopy(y) : nothing
                    _jac_config = build_jac_config(sensealg, _uf, t * dλ)
                    jacobian!(J, _uf, y, _f_cache, sensealg, _jac_config)
                else
                    uf.t = t
                    uf.p = p
                    if inplace_sensitivity(S)
                        jacobian!(J, uf, y, f_cache, sensealg, jac_config)
                    else
                        J = jacobian(uf, y, sensealg)
                    end
                end
            end
        else
            if SciMLBase.has_jac(f)
                f.jac(J, y, p, t, W) # Calculate the Jacobian into J
            else
                uf.t = t
                uf.p = p
                uf.W = W
                jacobian!(J, uf, y, f_cache, sensealg, jac_config)
            end
        end
        mul!(dλ', λ', J)
    end
    if dgrad !== nothing && !isempty(dgrad)
        (; pJ, pf, paramjac_config) = S.diffcache

        if TS <: Union{CallbackSensitivityFunction, CallbackSensitivityFunctionPSwap}
            if isscimlstructure(p) && !(p isa AbstractArray)
                _tunables_p, _, _ = canonicalize(Tunable(), p)
            elseif isfunctor(p)
                _tunables_p, _ = Functors.functor(p)
            else
                _tunables_p = p
            end
        elseif p isa AbstractArray
            # ForwardDiff/FiniteDiff need p in its original shape (e.g. Matrix);
            # canonicalize may flatten non-vector arrays, so use p directly.
            _tunables_p = p
        else
            _tunables_p = S.diffcache.tunables
        end

        if W === nothing
            if SciMLBase.has_paramjac(f)
                # Calculate the parameter Jacobian into pJ
                f.paramjac(pJ, y, p, t)
            else
                pf.t = t
                pf.u = y
                if inplace_sensitivity(S)
                    jacobian!(pJ, pf, _tunables_p, f_cache, sensealg, paramjac_config)
                else
                    temp = jacobian(pf, _tunables_p, sensealg)
                    pJ .= temp
                end
            end
        else
            if SciMLBase.has_paramjac(f)
                # Calculate the parameter Jacobian into pJ
                f.paramjac(pJ, y, p, t, W)
            else
                pf.t = t
                pf.u = y
                pf.W = W
                if inplace_sensitivity(S)
                    jacobian!(pJ, pf, _tunables_p, f_cache, sensealg, paramjac_config)
                else
                    temp = jacobian(pf, _tunables_p, sensealg)
                    pJ .= temp
                end
            end
        end
        mul!(dgrad', λ', pJ)
    end
    if dy !== nothing
        if W === nothing
            if inplace_sensitivity(S)
                f(dy, y, p, t)
            else
                recursive_copyto!(dy, vec(f(y, p, t)))
            end
        else
            if inplace_sensitivity(S)
                f(dy, y, p, t, W)
            else
                recursive_copyto!(dy, vec(f(y, p, t, W)))
            end
        end
    end
    return
end

const TRACKERVJP_NOTHING_MESSAGE = """
`nothing` returned from a Tracker vector-Jacobian product (vjp) calculation.
This indicates that your function `f` is not a function of `p` or `u`, i.e. that
the derivative is constant zero. In many cases this is due to an error in
the model definition, for example accidentally using a global parameter
instead of the one in the model (`f(u,p,t)= _p .* u`).

One common cause of this is using Flux neural networks with implicit parameters,
for example `f(u,p,t) = NN(u)` does not use `p` and therefore will have a zero
derivative. The answer is to use `Flux.destructure` in this case, for example:

```julia
p,re = Flux.destructure(NN)
f(u,p,t) = re(p)(u)
prob = ODEProblem(f,u0,tspan,p)
```

Note that restructuring outside of `f`, i.e. `reNN = re(p); f(u,p,t) = reNN(u)` will
also trigger a zero gradient. The `p` must be used inside of `f`, not globally outside.

If this zero gradient with respect to `u` or `p` is intended, then one can set
`TrackerVJP(allow_nothing=true)` to override this error message. For example:

```julia
solve(prob,alg,sensealg=InterpolatingAdjoint(autojacvec=TrackerVJP(allow_nothing=true)))
```
"""

struct TrackerVJPNothingError <: Exception end

function Base.showerror(io::IO, e::TrackerVJPNothingError)
    return print(io, TRACKERVJP_NOTHING_MESSAGE)
end

function _vecjacobian!(
        dλ, y, λ, p, t, S::TS, isautojacvec::TrackerVJP, dgrad, dy,
        W
    ) where {TS <: SensitivityFunction}
    (; sensealg) = S
    f = unwrapped_f(S.f)

    if inplace_sensitivity(S)
        if W === nothing
            _dy, back = Tracker.forward(y, p) do u, p
                out_ = map(zero, u)
                f(out_, u, p, t)
                Tracker.collect(out_)
            end
        else
            _dy, back = Tracker.forward(y, p) do u, p
                out_ = map(zero, u)
                f(out_, u, p, t, W)
                Tracker.collect(out_)
            end
        end

        if !(typeof(_dy) isa TrackedArray) && !(eltype(_dy) <: Tracker.TrackedReal) &&
                !sensealg.autojacvec.allow_nothing
            throw(TrackerVJPNothingError())
        end

        # Grab values from `_dy` before `back` in case mutated
        dy !== nothing && recursive_copyto!(dy, Tracker.data(_dy))

        tmp1, tmp2 = Tracker.data.(back(λ))
        dλ !== nothing && recursive_copyto!(dλ, tmp1)
        dgrad !== nothing && recursive_copyto!(dgrad, tmp2)
    else
        if W === nothing
            _dy, back = Tracker.forward(y, p) do u, p
                Tracker.collect(f(u, p, t))
            end
        else
            _dy, back = Tracker.forward(y, p) do u, p
                Tracker.collect(f(u, p, t, W))
            end
        end

        if !(typeof(_dy) isa TrackedArray) && !(eltype(_dy) <: Tracker.TrackedReal) &&
                !sensealg.autojacvec.allow_nothing
            throw(TrackerVJPNothingError())
        end

        # Grab values from `_dy` before `back` in case mutated
        dy !== nothing && recursive_copyto!(dy, Tracker.data(_dy))

        tmp1, tmp2 = Tracker.data.(back(λ))
        dλ !== nothing && recursive_copyto!(dλ, tmp1)
        dgrad !== nothing && recursive_copyto!(dgrad, tmp2)
    end
    return
end

function _vecjacobian!(
        dλ, y, λ, p, t, S::TS, isautojacvec::ReverseDiffVJP, dgrad, dy,
        W
    ) where {TS <: SensitivityFunction}
    (; sensealg) = S
    prob = getprob(S)
    f = unwrapped_f(S.f)

    if p isa SciMLBase.NullParameters
        _p = similar(y, (0,))
    else
        _p = p
    end

    if TS <: Union{CallbackSensitivityFunction, CallbackSensitivityFunctionPSwap}
        if p === nothing || p isa SciMLBase.NullParameters
            tunables, repack = p, identity
        elseif isscimlstructure(p) && !(p isa AbstractArray)
            tunables, repack, _ = canonicalize(Tunable(), p)
        elseif isfunctor(p)
            error(
                "ReverseDiffVJP does not support Functors.jl parameter structs. " *
                    "Use ZygoteVJP() instead, or make `p` a SciMLStructure. See SciMLStructures.jl."
            )
        else
            tunables, repack = p, identity
        end
    else
        (; tunables, repack) = S.diffcache
    end

    u0 = state_values(prob)
    if prob isa AbstractNonlinearProblem ||
            (
            eltype(λ) <: eltype(u0) && t isa eltype(u0) &&
                compile_tape(sensealg.autojacvec)
        )
        tape = S.diffcache.paramjac_config

        ## These other cases happen due to autodiff in stiff ODE solvers
    elseif inplace_sensitivity(S)
        _y = eltype(y) === eltype(λ) ? y : convert.(promote_type(eltype(y), eltype(λ)), y)
        if W === nothing
            if isscimlstructure(_p) && !(_p isa AbstractArray)
                _tunables, _repack, _ = canonicalize(Tunable(), _p)
            else
                _tunables, _repack = _p, identity
            end
            _is_pswap = TS <: CallbackSensitivityFunctionPSwap
            tape = ReverseDiff.GradientTape((_y, _tunables, [t])) do u, p, t
                du1 = _is_pswap ? similar(p, length(p)) : similar(u, size(u))
                du1 .= false
                f(du1, u, _repack(p), first(t))
                return vec(du1)
            end
        else
            _W = eltype(W) === eltype(λ) ? W :
                convert.(promote_type(eltype(W), eltype(λ)), W)
            tape = ReverseDiff.GradientTape((_y, _p, [t], _W)) do u, p, t, Wloc
                du1 = p !== nothing && p !== SciMLBase.NullParameters() ?
                    similar(p, size(u)) : similar(u)
                f(du1, u, p, first(t), Wloc)
                return vec(du1)
            end
        end
    else
        _y = eltype(y) === eltype(λ) ? y : convert.(promote_type(eltype(y), eltype(λ)), y)
        if W === nothing
            if isscimlstructure(_p) && !(_p isa AbstractArray)
                _tunables, _repack, _ = canonicalize(Tunable(), _p)
            else
                _tunables, _repack = _p, identity
            end
            tape = ReverseDiff.GradientTape((_y, _tunables, [t])) do u, p, t
                vec(f(u, _repack(p), first(t)))
            end
        else
            _W = eltype(W) === eltype(λ) ? W :
                convert.(promote_type(eltype(W), eltype(λ)), W)
            tape = ReverseDiff.GradientTape((_y, _p, [t], _W)) do u, p, t, Wloc
                vec(f(u, p, first(t), Wloc))
            end
        end
    end

    if prob isa AbstractNonlinearProblem
        tu, tp = ReverseDiff.input_hook(tape)
    else
        if W === nothing
            tu, tp, tt = ReverseDiff.input_hook(tape)
        else
            tu, tp, tt, tW = ReverseDiff.input_hook(tape)
        end
    end
    output = ReverseDiff.output_hook(tape)
    ReverseDiff.unseed!(tu) # clear any "leftover" derivatives from previous calls
    ReverseDiff.unseed!(tp)
    if !(prob isa AbstractNonlinearProblem)
        ReverseDiff.unseed!(tt)
    end
    W !== nothing && ReverseDiff.unseed!(tW)
    ReverseDiff.value!(tu, y)
    p isa SciMLBase.NullParameters || ReverseDiff.value!(tp, tunables)
    if !(prob isa AbstractNonlinearProblem)
        ReverseDiff.value!(tt, [t])
    end
    W !== nothing && ReverseDiff.value!(tW, W)
    ReverseDiff.forward_pass!(tape)
    ReverseDiff.increment_deriv!(output, λ)
    ReverseDiff.reverse_pass!(tape)
    dλ !== nothing && copyto!(vec(dλ), ReverseDiff.deriv(tu))
    dgrad !== nothing && copyto!(vec(dgrad), ReverseDiff.deriv(tp))
    ReverseDiff.pull_value!(output)
    dy !== nothing && copyto!(vec(dy), ReverseDiff.value(output))
    return
end

const ZYGOTEVJP_NOTHING_MESSAGE = """
`nothing` returned from a Zygote vector-Jacobian product (vjp) calculation.
This indicates that your function `f` is not a function of `p` or `u`, i.e. that
the derivative is constant zero. In many cases this is due to an error in
the model definition, for example accidentally using a global parameter
instead of the one in the model (`f(u,p,t)= _p .* u`).

One common cause of this is using Flux neural networks with implicit parameters,
for example `f(u,p,t) = NN(u)` does not use `p` and therefore will have a zero
derivative. The answer is to use `Flux.destructure` in this case, for example:

```julia
p,re = Flux.destructure(NN)
f(u,p,t) = re(p)(u)
prob = ODEProblem(f,u0,tspan,p)
```

Note that restructuring outside of `f`, i.e. `reNN = re(p); f(u,p,t) = reNN(u)` will
also trigger a zero gradient. The `p` must be used inside of `f`, not globally outside.

If this zero gradient with respect to `u` or `p` is intended, then one can set
`ZygoteVJP(allow_nothing=true)` to override this error message, for example:

```julia
solve(prob,alg,sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP(allow_nothing=true)))
```
"""

struct ZygoteVJPNothingError <: Exception end

function Base.showerror(io::IO, e::ZygoteVJPNothingError)
    return print(io, ZYGOTEVJP_NOTHING_MESSAGE)
end

function _vecjacobian!(
        dλ, y, λ, p, t, S::TS, isautojacvec::ZygoteVJP, dgrad, dy,
        W
    ) where {TS <: SensitivityFunction}
    (; sensealg) = S
    prob = getprob(S)
    f = unwrapped_f(S.f)

    if TS <: Union{CallbackSensitivityFunction, CallbackSensitivityFunctionPSwap}
        _needs_repack = !(p === nothing || p isa SciMLBase.NullParameters) &&
            ((isscimlstructure(p) && !(p isa AbstractArray)) || isfunctor(p))
        if _needs_repack
            if isscimlstructure(p) && !(p isa AbstractArray)
                tunables, repack, _ = canonicalize(Tunable(), p)
            else
                tunables, repack = Functors.functor(p)
            end
        else
            tunables, repack = p, identity
        end
    else
        (; tunables, repack) = S.diffcache
    end

    if inplace_sensitivity(S)
        if W === nothing
            _dy, back = Zygote.pullback(y, tunables) do u, tunables
                out_ = Zygote.Buffer(similar(u))
                f(out_, u, repack(tunables), t)
                vec(copy(out_))
            end
        else
            _dy, back = Zygote.pullback(y, tunables) do u, tunables
                out_ = Zygote.Buffer(similar(u))
                f(out_, u, repack(tunables), t, W)
                vec(copy(out_))
            end
        end

        # Grab values from `_dy` before `back` in case mutated
        dy !== nothing && recursive_copyto!(dy, _dy)

        tmp1, tmp2 = back(λ)
        dλ !== nothing && recursive_copyto!(dλ, tmp1)
        if dgrad !== nothing
            if tmp2 === nothing && !sensealg.autojacvec.allow_nothing
                throw(ZygoteVJPNothingError())
            else
                !isempty(dgrad) && recursive_copyto!(dgrad, tmp2)
            end
        end
    else
        if W === nothing
            _dy, back = Zygote.pullback(y, tunables) do u, tunables
                vec(f(u, repack(tunables), t))
            end
        else
            _dy, back = Zygote.pullback(y, tunables) do u, tunables
                vec(f(u, repack(tunables), t, W))
            end
        end

        # Grab values from `_dy` before `back` in case mutated
        dy !== nothing && recursive_copyto!(dy, _dy)

        tmp1, tmp2 = back(λ)
        if tmp1 === nothing && !sensealg.autojacvec.allow_nothing
            throw(ZygoteVJPNothingError())
        elseif tmp1 !== nothing
            dλ !== nothing && recursive_copyto!(dλ, tmp1)
        end

        if dgrad !== nothing
            if tmp2 === nothing && !sensealg.autojacvec.allow_nothing
                throw(ZygoteVJPNothingError())
            elseif tmp2 !== nothing
                !isempty(dgrad) && recursive_copyto!(dgrad, tmp2)
            end
        end
    end
    return
end

function _vecjacobian(
        y, λ, p, t, S::TS, isautojacvec::ZygoteVJP, dgrad, dy,
        W
    ) where {TS <: SensitivityFunction}
    (; sensealg) = S
    prob = getprob(S)
    f = unwrapped_f(S.f)

    if W === nothing
        _dy, back = Zygote.pullback(y, p) do u, p
            vec(f(u, p, t))
        end
    else
        _dy, back = Zygote.pullback(y, p) do u, p
            vec(f(u, p, t, W))
        end
    end

    # Grab values from `_dy` before `back` in case mutated
    dy !== nothing && recursive_copyto!(dy, _dy)

    tmp1, tmp2 = back(λ)
    if tmp1 === nothing && !sensealg.autojacvec.allow_nothing
        throw(ZygoteVJPNothingError())
    elseif tmp1 !== nothing
        dλ = vec(tmp1)
    end

    if dgrad !== nothing
        if tmp2 === nothing && !sensealg.autojacvec.allow_nothing
            throw(ZygoteVJPNothingError())
        elseif tmp2 !== nothing
            recursive_copyto!(dgrad, tmp2)
        end
    end
    return dy, dλ, dgrad
end

function gclosure1(f, du, u, p, t)
    Base.copyto!(du, f(u, p, t))
    return nothing
end

function gclosure2(f, du, u, p, t, W)
    Base.copyto!(du, f(u, p, t, W))
    return nothing
end

function _vecjacobian!(
        dλ, y, λ, p, t, S::TS, isautojacvec::EnzymeVJP, dgrad, dy,
        W
    ) where {TS <: SensitivityFunction}
    (; sensealg) = S
    f = unwrapped_f(S.f)

    prob = getprob(S)

    if TS <: Union{CallbackSensitivityFunction, CallbackSensitivityFunctionPSwap}
        _p = p
        if _p === nothing || _p isa SciMLBase.NullParameters
            tunables, repack, trivial_repack = _p, identity, true
        elseif isscimlstructure(_p) && !(_p isa AbstractArray)
            tunables, repack, aliases = canonicalize(Tunable(), _p)
            trivial_repack = aliases && _p isa AbstractVector
        elseif isfunctor(_p)
            tunables, repack = Functors.functor(_p)
            trivial_repack = false
        else
            tunables, repack, trivial_repack = _p, identity, true
        end
    else
        _p = parameter_values(prob)
        (; tunables, repack) = S.diffcache
        trivial_repack = _p === nothing || _p isa SciMLBase.NullParameters ||
            (_p isa AbstractVector)
    end

    _tmp1, tmp2, _tmp3, _tmp4, _tmp5, _tmp6, _cached_shadow = S.diffcache.paramjac_config

    if _tmp1 isa LazyBufferCache
        # Use a template that combines y's wrapper type (e.g., ComponentArray) with
        # dλ's element type (e.g., Float64 or ForwardDiff.Dual). This ensures the
        # LazyBufferCache creates buffers with the correct wrapper type for the user's
        # ODE function while supporting ForwardDiff Dual element types.
        _cache_tmpl = eltype(dλ) === eltype(y) ? y : similar(y, eltype(dλ))
        tmp1 = get_tmp(_tmp1, _cache_tmpl)
        tmp3 = get_tmp(_tmp3, _cache_tmpl)
        tmp4 = get_tmp(_tmp4, _cache_tmpl)
        ytmp = get_tmp(_tmp5, _cache_tmpl)
    else
        tmp1 = _tmp1
        tmp3 = _tmp3
        tmp4 = _tmp4
        ytmp = _tmp5
    end

    Enzyme.remake_zero!(tmp1) # should be removed for dλ
    vec(ytmp) .= vec(y)

    #if dgrad !== nothing
    #  tmp2 = dgrad
    #else
    _shadow_p = nothing
    dup = if !(tmp2 isa SciMLBase.NullParameters)
        # tmp2 .= 0
        Enzyme.remake_zero!(tmp2)
        if _cached_shadow !== nothing
            Enzyme.remake_zero!(_cached_shadow)
            _sp = _cached_shadow
        else
            _sp = trivial_repack ? tmp2 : repack(tmp2)
        end
        _shadow_p = _sp
        Enzyme.Duplicated(p, _sp)
    else
        Enzyme.Const(p)
    end
    #end

    #if dy !== nothing
    #      tmp3 = dy
    #else
    Enzyme.remake_zero!(tmp3)
    #end

    vec(tmp4) .= vec(λ)
    isautojacvec = get_jacvec(sensealg)
    # Extract Enzyme mode from EnzymeVJP or use default Enzyme.Reverse
    enzyme_mode = isautojacvec isa EnzymeVJP ? isautojacvec.mode : Enzyme.Reverse

    if inplace_sensitivity(S)
        Enzyme.remake_zero!(_tmp6)

        if W === nothing
            Enzyme.autodiff(
                enzyme_mode, Enzyme.Duplicated(SciMLBase.Void(f), _tmp6),
                Enzyme.Const, Enzyme.Duplicated(tmp3, tmp4),
                Enzyme.Duplicated(ytmp, tmp1),
                dup,
                Enzyme.Const(t)
            )
        else
            Enzyme.autodiff(
                enzyme_mode, Enzyme.Duplicated(SciMLBase.Void(f), _tmp6),
                Enzyme.Const, Enzyme.Duplicated(tmp3, tmp4),
                Enzyme.Duplicated(ytmp, tmp1),
                dup,
                Enzyme.Const(t), Enzyme.Const(W)
            )
        end
        dλ !== nothing && recursive_copyto!(dλ, tmp1)
        if dgrad !== nothing && !(tmp2 isa SciMLBase.NullParameters)
            if !trivial_repack && _shadow_p !== nothing
                grad_tunables, _, _ = canonicalize(Tunable(), _shadow_p)
                recursive_copyto!(dgrad, grad_tunables)
            else
                recursive_copyto!(dgrad, tmp2)
            end
        end
        dy !== nothing && recursive_copyto!(dy, tmp3)
    else
        if W === nothing
            _tmp6 = Enzyme.make_zero(f)
            Enzyme.autodiff(
                enzyme_mode, Enzyme.Const(gclosure1), Enzyme.Const,
                Enzyme.Duplicated(f, _tmp6),
                Enzyme.Duplicated(tmp3, tmp4),
                Enzyme.Duplicated(ytmp, tmp1),
                dup, Enzyme.Const(t)
            )
        else
            _tmp6 = Enzyme.make_zero(f)
            Enzyme.autodiff(
                enzyme_mode, Enzyme.Const(gclosure2), Enzyme.Const,
                Enzyme.Duplicated(f, _tmp6),
                Enzyme.Duplicated(tmp3, tmp4),
                Enzyme.Duplicated(ytmp, tmp1),
                dup, Enzyme.Const(t), Enzyme.Const(W)
            )
        end
        if dy !== nothing
            out_ = if W === nothing
                f(y, p, t)
            else
                f(y, p, t, W)
            end
            recursive_copyto!(dy, out_)
        end
        dλ !== nothing && recursive_copyto!(dλ, tmp1)
        if dgrad !== nothing && !(tmp2 isa SciMLBase.NullParameters)
            if !trivial_repack && _shadow_p !== nothing
                grad_tunables, _, _ = canonicalize(Tunable(), _shadow_p)
                recursive_copyto!(dgrad, grad_tunables)
            else
                recursive_copyto!(dgrad, tmp2)
            end
        end
        dy !== nothing && recursive_copyto!(dy, tmp3)
    end
    return
end

function _vecjacobian!(dλ, y, λ, p, t, S::SensitivityFunction, ::MooncakeVJP, dgrad, dy, W)
    _dy, y_grad, p_grad = mooncake_run_ad(S.diffcache.paramjac_config, y, p, t, λ)
    dy !== nothing && recursive_copyto!(dy, _dy)
    dλ !== nothing && recursive_copyto!(dλ, y_grad)
    dgrad !== nothing && recursive_copyto!(dgrad, p_grad)
    return
end

function _vecjacobian!(dλ, y, λ, p, t, S::SensitivityFunction, ::ReactantVJP, dgrad, dy, W)
    _dy, y_grad, p_grad = reactant_run_ad(S.diffcache.paramjac_config, y, p, t, λ)
    dy !== nothing && recursive_copyto!(dy, _dy)
    dλ !== nothing && recursive_copyto!(dλ, y_grad)
    dgrad !== nothing && recursive_copyto!(dgrad, p_grad)
    return
end

function jacNoise!(
        λ, y, p, t, S::SensitivityFunction;
        dgrad = nothing, dλ = nothing, dy = nothing
    )
    _jacNoise!(λ, y, p, t, S, S.sensealg.autojacvec, dgrad, dλ, dy)
    return
end

function _jacNoise!(
        λ, y, p, t, S::TS, isnoise::Bool, dgrad, dλ,
        dy
    ) where {TS <: SensitivityFunction}
    (; sensealg, f) = S
    prob = getprob(S)

    if dgrad !== nothing
        (; pJ, pf, f_cache, paramjac_noise_config) = S.diffcache
        if SciMLBase.has_paramjac(f)
            # Calculate the parameter Jacobian into pJ
            f.paramjac(pJ, y, p, t)
        else
            pf.t = t
            pf.u = y
            if inplace_sensitivity(S)
                jacobian!(pJ, pf, p, nothing, sensealg, nothing)
                #jacobian!(pJ, pf, p, f_cache, sensealg, paramjac_noise_config)
            else
                temp = jacobian(pf, p, sensealg)
                pJ .= temp
            end
        end

        if SciMLBase.is_diagonal_noise(prob)
            pJt = transpose(λ) .* transpose(pJ)
            recursive_copyto!(dgrad, pJt)
        else
            m = size(prob.noise_rate_prototype)[2]
            for i in 1:m
                tmp = λ' * pJ[((i - 1) * m + 1):(i * m), :]
                if dgrad !== nothing
                    if tmp !== nothing
                        !isempty(dgrad) && (dgrad[:, i] .= vec(tmp))
                    end
                end
            end
        end
    end

    if dλ !== nothing &&
            (isnoisemixing(sensealg) || !SciMLBase.is_diagonal_noise(prob))
        (; J, uf, f_cache, jac_noise_config) = S.diffcache
        if dy !== nothing
            if inplace_sensitivity(S)
                f(dy, y, p, t)
            else
                dy .= f(y, p, t)
            end
        end

        if SciMLBase.has_jac(f)
            f.jac(J, y, p, t) # Calculate the Jacobian into J
        else
            if inplace_sensitivity(S)
                if dy !== nothing
                    ForwardDiff.jacobian!(J, uf, dy, y)
                else
                    if SciMLBase.is_diagonal_noise(prob)
                        dy = similar(y)
                    else
                        dy = similar(prob.noise_rate_prototype)
                        f(dy, y, p, t)
                        ForwardDiff.jacobian!(J, uf, dy, y)
                    end
                    f(dy, y, p, t)
                    ForwardDiff.jacobian!(J, uf, dy, y)
                end
            else
                tmp = ForwardDiff.jacobian(uf, y)
                J .= tmp
            end
            #  uf.t = t
            #  uf.p = p
            #  jacobian!(J, uf, y, nothing, sensealg, nothing)
        end

        if SciMLBase.is_diagonal_noise(prob)
            Jt = transpose(λ) .* transpose(J)
            recursive_copyto!(dλ, Jt)
        else
            for i in 1:m
                tmp = λ' * J[((i - 1) * m + 1):(i * m), :]
                dλ[:, i] .= vec(tmp)
            end
        end
    end
    return
end

function _jacNoise!(
        λ, y, p, t, S::TS, isnoise::ReverseDiffVJP, dgrad, dλ,
        dy
    ) where {TS <: SensitivityFunction}
    (; sensealg) = S
    prob = getprob(S)
    f = unwrapped_f(S.f)

    # number of Wiener processes
    noise_rate_prototype = prob.noise_rate_prototype
    m = noise_rate_prototype === nothing ? length(y) : size(noise_rate_prototype)[2]

    for i in 1:m
        tapei = S.diffcache.paramjac_noise_config[i]
        tu, tp, tt = ReverseDiff.input_hook(tapei)
        output = ReverseDiff.output_hook(tapei)
        ReverseDiff.unseed!(tu) # clear any "leftover" derivatives from previous calls
        ReverseDiff.unseed!(tp)
        ReverseDiff.unseed!(tt)
        ReverseDiff.value!(tu, y)
        ReverseDiff.value!(tp, p)
        ReverseDiff.value!(tt, [t])
        ReverseDiff.forward_pass!(tapei)
        if SciMLBase.is_diagonal_noise(prob)
            ReverseDiff.increment_deriv!(output, λ[i])
        else
            ReverseDiff.increment_deriv!(output, λ)
        end
        ReverseDiff.reverse_pass!(tapei)

        deriv = ReverseDiff.deriv(tp)
        if dgrad !== nothing
            if deriv !== nothing
                !isempty(dgrad) && (dgrad[:, i] .= vec(deriv))
            end
        end
        ReverseDiff.pull_value!(output)

        if SciMLBase.is_diagonal_noise(prob)
            dλ !== nothing && (dλ[:, i] .= vec(ReverseDiff.deriv(tu)))
            dy !== nothing && (dy[i] = ReverseDiff.value(output))
        else
            dλ !== nothing && (dλ[:, i] .= vec(ReverseDiff.deriv(tu)))
            dy !== nothing && (dy[:, i] .= vec(ReverseDiff.value(output)))
        end
    end
    return
end

function _jacNoise!(
        λ, y, p, t, S::TS, isnoise::ZygoteVJP, dgrad, dλ,
        dy
    ) where {TS <: SensitivityFunction}
    (; sensealg) = S
    prob = getprob(S)
    (; tunables, repack) = S.diffcache
    f = unwrapped_f(S.f)

    # number of Wiener processes
    noise_rate_prototype = prob.noise_rate_prototype
    m = noise_rate_prototype === nothing ? length(y) : size(noise_rate_prototype)[2]

    if SciMLBase.is_diagonal_noise(prob)
        if VERSION < v"1.9" # pre "stack" function
            for i in 1:m
                if inplace_sensitivity(S)
                    _dy, back = Zygote.pullback(y, p) do u, p
                        out_ = Zygote.Buffer(zero(u))
                        f(out_, u, repack(p), t)
                        copy(out_[i])
                    end
                else
                    _dy, back = Zygote.pullback(y, p) do u, p
                        f(u, repack(p), t)[i]
                    end
                end
                tmp1, tmp2 = back(λ[i]) #issue: tmp2 = zeros(p)
                if dgrad !== nothing
                    if tmp2 !== nothing
                        !isempty(dgrad) && (dgrad[:, i] .= vec(tmp2))
                    end
                end
                dλ !== nothing && (dλ[:, i] .= vec(tmp1))
                dy !== nothing && (dy[i] = _dy)
            end
        else
            if inplace_sensitivity(S)
                _dy, back = Zygote.pullback(y, p) do u, p
                    out_ = Zygote.Buffer(zero(u))
                    f(out_, u, repack(p), t)
                    copy(out_)
                end
            else
                _dy, back = Zygote.pullback(y, p) do u, p
                    f(u, repack(p), t)
                end
            end
            out = [back(x) for x in eachcol(Diagonal(λ))]
            if dgrad !== nothing
                tmp2 = last.(out)
                if !(eltype(tmp2) isa Nothing)
                    !isempty(dgrad) && (dgrad .= stack(tmp2))
                end
            end
            dλ !== nothing && (dλ .= stack(first.(out)))
            dy !== nothing && (dy .= _dy)
        end
    else
        if inplace_sensitivity(S)
            for i in 1:m
                _dy, back = Zygote.pullback(y, p) do u, p
                    out_ = Zygote.Buffer(zero(prob.noise_rate_prototype))
                    f(out_, u, repack(p), t)
                    copy(out_[:, i])
                end
                tmp1, tmp2 = back(λ) #issue with Zygote.Buffer
                if dgrad !== nothing
                    if tmp2 !== nothing
                        !isempty(dgrad) && (dgrad[:, i] .= vec(tmp2))
                    end
                end
                dλ !== nothing && (dλ[:, i] .= vec(tmp1))
                dy !== nothing && (dy[:, i] .= vec(_dy))
            end
        else
            for i in 1:m
                _dy, back = Zygote.pullback(y, p) do u, p
                    f(u, repack(p), t)[:, i]
                end
                tmp1, tmp2 = back(λ)
                if dgrad !== nothing
                    if tmp2 !== nothing
                        !isempty(dλ) && (dgrad[:, i] .= vec(tmp2))
                    end
                end
                if tmp1 === nothing
                    # if a column of the noise matrix is zero, Zygote returns nothing.
                    dλ !== nothing && (dλ[:, i] .= false)
                else
                    dλ !== nothing && (dλ[:, i] .= vec(tmp1))
                end
                dy !== nothing && (dy[:, i] .= vec(_dy))
            end
        end
    end
    return
end

function accumulate_cost!(
        dλ, y, p, t, S::TS,
        dgrad = nothing
    ) where {TS <: SensitivityFunction}
    (; dgdu, dgdp, dg_val, g, g_grad_config) = S.diffcache

    if dgdu !== nothing
        if dgdp === nothing
            dgdu(dg_val, y, p, t)
            dλ .-= vec(dg_val)
        else
            dgdu(dg_val[1], y, p, t)
            dλ .-= vec(dg_val[1])
            if dgrad !== nothing
                dgdp(dg_val[2], y, p, t)
                dgrad .-= vec(dg_val[2])
            end
        end
    else
        g[1].t = t
        g[1].p = p
        gradient!(dg_val[1], g[1], y, S.sensealg, g_grad_config[1])
        dλ .-= vec(dg_val[1])
        if dgrad !== nothing
            g[2].t = t
            g[2].u = y
            gradient!(dg_val[2], g[2], p, S.sensealg, g_grad_config[2])
            dgrad .-= vec(dg_val[2])
        end
    end
    return nothing
end

function accumulate_cost(
        dλ, y, p, t, S::TS,
        dgrad = nothing
    ) where {TS <: SensitivityFunction}
    (; dgdu, dgdp) = S.diffcache

    dλ -= dgdu(y, p, t)
    if dgdp !== nothing
        if dgrad !== nothing
            dgrad -= dgdp(y, p, t)
        end
    end
    return dλ, dgrad
end

build_jac_config(alg, uf, u::Nothing) = nothing
function build_jac_config(alg, uf, u)
    if alg_autodiff(alg)
        jac_config = ForwardDiff.JacobianConfig(
            uf, u, u,
            ForwardDiff.Chunk{
                determine_chunksize(
                    u,
                    alg
                ),
            }()
        )
    else
        if diff_type(alg) != Val{:complex}
            jac_config = FiniteDiff.JacobianCache(
                zero(u), zero(u),
                zero(u), diff_type(alg)
            )
        else
            tmp = Complex{eltype(u)}.(u)
            du1 = Complex{eltype(u)}.(du1)
            jac_config = FiniteDiff.JacobianCache(tmp, du1, nothing, diff_type(alg))
        end
    end
    return jac_config
end

function build_param_jac_config(alg, pf, u, p)
    if alg_autodiff(alg)
        tunables, repack, aliases = canonicalize(Tunable(), p)
        jac_config = ForwardDiff.JacobianConfig(
            pf, u, tunables,
            ForwardDiff.Chunk{
                determine_chunksize(
                    tunables,
                    alg
                ),
            }()
        )
    else
        if diff_type(alg) != Val{:complex}
            jac_config = FiniteDiff.JacobianCache(
                similar(p), similar(u),
                similar(u), diff_type(alg)
            )
        else
            tmp = Complex{eltype(p)}.(p)
            du1 = Complex{eltype(u)}.(u)
            jac_config = FiniteDiff.JacobianCache(tmp, du1, nothing, diff_type(alg))
        end
    end
    return jac_config
end

function build_grad_config(alg, tf, du1, t)
    if alg_autodiff(alg)
        grad_config = ForwardDiff.GradientConfig(
            tf, du1,
            ForwardDiff.Chunk{
                determine_chunksize(
                    du1,
                    alg
                ),
            }()
        )
    else
        grad_config = FiniteDiff.GradientCache(du1, t, diff_type(alg))
    end
    return grad_config
end

function build_deriv_config(alg, tf, du1, t)
    if alg_autodiff(alg)
        grad_config = ForwardDiff.DerivativeConfig(tf, du1, t)
    else
        grad_config = FiniteDiff.DerivativeCache(du1, t, diff_type(alg))
    end
    return grad_config
end
