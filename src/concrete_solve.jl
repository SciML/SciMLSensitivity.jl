## High level

# Here is where we can add a default algorithm for computing sensitivities
# Based on problem information!

function inplace_vjp(prob,u0,p,verbose)
  du = copy(u0)
  ez = try
    Enzyme.autodiff(Enzyme.Duplicated(du, du),
                    copy(u0),copy(p),prob.tspan[1]) do out,u,_p,t
      prob.f(out, u, _p, t)
      nothing
    end
    true
  catch
    false
  end
  if ez
    return EnzymeVJP()
  end

  # Determine if we can compile ReverseDiff
  compile = try
      if DiffEqBase.isinplace(prob)
        !hasbranching(prob.f,copy(u0),u0,p,prob.tspan[1])
      else
        !hasbranching(prob.f,u0,p,prob.tspan[1])
      end
    catch
      false
  end

  vjp = try
    ReverseDiff.GradientTape((copy(u0), p, [prob.tspan[1]])) do u,p,t
      du1 = similar(u, size(u))
      prob.f(du1,u,p,first(t))
      return vec(du1)
    end
    ReverseDiffVJP(compile)
  catch
    false
  end
  return vjp
end

function automatic_sensealg_choice(prob::Union{ODEProblem,SDEProblem},u0,p,verbose)

  default_sensealg = if p !== DiffEqBase.NullParameters() &&
                          !(eltype(u0) <: ForwardDiff.Dual) &&
                          !(eltype(p) <: ForwardDiff.Dual) &&
                          !(eltype(u0) <: Complex) &&
                          !(eltype(p) <: Complex) &&
                          length(u0) + length(p) <= 100
    ForwardDiffSensitivity()
  elseif u0 isa GPUArraysCore.AbstractGPUArray || !DiffEqBase.isinplace(prob)
    # only Zygote is GPU compatible and fast
    # so if out-of-place, try Zygote
    if p === nothing || p === DiffEqBase.NullParameters()
      # QuadratureAdjoint skips all p calculations until the end
      # So it's the fastest when there are no parameters
      QuadratureAdjoint(autojacvec=ZygoteVJP())
    else
      InterpolatingAdjoint(autojacvec=ZygoteVJP())
    end
  else
    vjp = inplace_vjp(prob,u0,p,verbose)
    if p === nothing || p === DiffEqBase.NullParameters()
      QuadratureAdjoint(autojacvec=vjp)
    else
      InterpolatingAdjoint(autojacvec=vjp)
    end
  end
  return default_sensealg
end

function automatic_sensealg_choice(prob::Union{NonlinearProblem,SteadyStateProblem}, u0, p, verbose)

  default_sensealg = if u0 isa GPUArraysCore.AbstractGPUArray || !DiffEqBase.isinplace(prob)
    # autodiff = false because forwarddiff fails on many GPU kernels
    # this only effects the Jacobian calculation and is same computation order
    SteadyStateAdjoint(autodiff=false, autojacvec=ZygoteVJP())
  else
    vjp = inplace_vjp(prob,u0,p,verbose)
    SteadyStateAdjoint(autojacvec=vjp)
  end
  return default_sensealg
end

function DiffEqBase._concrete_solve_adjoint(prob::Union{ODEProblem,SDEProblem},
                                            alg,sensealg::Nothing,u0,p,originator::SciMLBase.ADOriginator,args...;
                                            verbose=true,kwargs...)

  if haskey(kwargs,:callback)
    has_cb = kwargs[:callback]!==nothing
  else
    has_cb = false
  end
  default_sensealg = automatic_sensealg_choice(prob,u0,p,verbose)
  if has_cb && typeof(default_sensealg) <: AbstractAdjointSensitivityAlgorithm
    default_sensealg = setvjp(default_sensealg, ReverseDiffVJP())
  end
  DiffEqBase._concrete_solve_adjoint(prob,alg,default_sensealg,u0,p,originator::SciMLBase.ADOriginator,args...;verbose,kwargs...)
end

function DiffEqBase._concrete_solve_adjoint(prob::Union{NonlinearProblem,SteadyStateProblem},alg,
                                            sensealg::Nothing,u0,p,originator::SciMLBase.ADOriginator,args...;
                                            verbose=true,kwargs...)

  default_sensealg = automatic_sensealg_choice(prob, u0, p, verbose)
  DiffEqBase._concrete_solve_adjoint(prob,alg,default_sensealg,u0,p,originator::SciMLBase.ADOriginator,args...;verbose,kwargs...)
end

function DiffEqBase._concrete_solve_adjoint(prob::Union{DiscreteProblem,DDEProblem,
                                                        SDDEProblem,DAEProblem},
                                                        alg,sensealg::Nothing,
                                                        u0,p,originator::SciMLBase.ADOriginator,args...;kwargs...)
  if length(u0) + length(p) > 100
      default_sensealg = ReverseDiffAdjoint()
  else
      default_sensealg = ForwardDiffSensitivity()
  end
  DiffEqBase._concrete_solve_adjoint(prob,alg,default_sensealg,u0,p,originator::SciMLBase.ADOriginator,args...;kwargs...)
end

function DiffEqBase._concrete_solve_adjoint(prob,alg,
                                 sensealg::AbstractAdjointSensitivityAlgorithm,
                                 u0,p,originator::SciMLBase.ADOriginator,args...;save_start=true,save_end=true,
                                 saveat = eltype(prob.tspan)[],
                                 save_idxs = nothing,
                                 kwargs...)

  if !(typeof(p) <: Union{Nothing,SciMLBase.NullParameters,AbstractArray}) || (p isa AbstractArray && !Base.isconcretetype(eltype(p)))
    throw(AdjointSensitivityParameterCompatibilityError())
  end

  # Remove saveat, etc. from kwargs since it's handled separately
  # and letting it jump back in there can break the adjoint
  kwargs_prob = NamedTuple(filter(x->x[1] != :saveat && x[1] != :save_start && x[1] != :save_end && x[1] != :save_idxs,prob.kwargs))

  if haskey(kwargs, :callback)
    cb = track_callbacks(CallbackSet(kwargs[:callback]),prob.tspan[1],prob.u0,prob.p,sensealg)
    _prob = remake(prob;u0=u0,p=p,kwargs = merge(kwargs_prob,(;callback=cb)))
  else
    cb = nothing
    _prob = remake(prob;u0=u0,p=p,kwargs = kwargs_prob)
  end

  # Remove callbacks, saveat, etc. from kwargs since it's handled separately
  kwargs_fwd = NamedTuple{Base.diff_names(Base._nt_names(
  values(kwargs)), (:callback,))}(values(kwargs))

  # Capture the callback_adj for the reverse pass and remove both callbacks
  kwargs_adj = NamedTuple{Base.diff_names(Base._nt_names(values(kwargs)), (:callback_adj,:callback))}(values(kwargs))
  isq = sensealg isa QuadratureAdjoint
  if typeof(sensealg) <: BacksolveAdjoint
    sol = solve(_prob,alg,args...;save_noise=true,
                                  save_start=save_start,save_end=save_end,
                                  saveat=saveat,kwargs_fwd...)
  elseif ischeckpointing(sensealg)
    sol = solve(_prob,alg,args...;save_noise=true,
                                  save_start=true,save_end=true,
                                  saveat=saveat,kwargs_fwd...)
  else
    sol = solve(_prob,alg,args...;save_noise=true,save_start=true,
                                  save_end=true,kwargs_fwd...)
  end

  # Force `save_start` and `save_end` in the forward pass This forces the
  # solver to do the backsolve all the way back to `u0` Since the start aliases
  # `_prob.u0`, this doesn't actually use more memory But it cleans up the
  # implementation and makes `save_start` and `save_end` arg safe.
  if typeof(sensealg) <: BacksolveAdjoint
    # Saving behavior unchanged
    ts = sol.t
    only_end = length(ts) == 1 && ts[1] == _prob.tspan[2]
    out = DiffEqBase.sensitivity_solution(sol,sol.u,ts)
  elseif saveat isa Number
    if _prob.tspan[2] > _prob.tspan[1]
      ts = _prob.tspan[1]:convert(typeof(_prob.tspan[2]),abs(saveat)):_prob.tspan[2]
    else
      ts = _prob.tspan[2]:convert(typeof(_prob.tspan[2]),abs(saveat)):_prob.tspan[1]
    end
    # if _prob.tspan[2]-_prob.tspan[1] is not a multiple of saveat, one looses the last ts value
    sol.t[end] !== ts[end] && (ts = fix_endpoints(sensealg,sol,ts))
    if cb === nothing
      _out = sol(ts)
    else
      _, duplicate_iterator_times = separate_nonunique(sol.t)
      _out, ts = out_and_ts(ts, duplicate_iterator_times, sol)
    end

    out = if save_idxs === nothing
      out = DiffEqBase.sensitivity_solution(sol,_out.u,ts)
    else
      out = DiffEqBase.sensitivity_solution(sol,[_out[i][save_idxs] for i in 1:length(_out)],ts)
    end
    only_end = length(ts) == 1 && ts[1] == _prob.tspan[2]
  elseif isempty(saveat)
    no_start = !save_start
    no_end = !save_end
    sol_idxs = 1:length(sol)
    no_start && (sol_idxs = sol_idxs[2:end])
    no_end && (sol_idxs = sol_idxs[1:end-1])
    only_end = length(sol_idxs) <= 1
    _u = sol.u[sol_idxs]
    u = save_idxs === nothing ? _u : [x[save_idxs] for x in _u]
    ts = sol.t[sol_idxs]
    out = DiffEqBase.sensitivity_solution(sol,u,ts)
  else
    _saveat = saveat isa Array ? sort(saveat) : saveat # for minibatching
    if cb === nothing
      _saveat = eltype(_saveat) <: typeof(prob.tspan[2]) ? convert.(typeof(_prob.tspan[2]),_saveat) : _saveat
      ts = _saveat
      _out = sol(ts)
    else
      _ts, duplicate_iterator_times = separate_nonunique(sol.t)
      _out, ts = out_and_ts(_saveat, duplicate_iterator_times, sol)
    end

    out = if save_idxs === nothing
      out = DiffEqBase.sensitivity_solution(sol,_out.u,ts)
    else
      out = DiffEqBase.sensitivity_solution(sol,[_out[i][save_idxs] for i in 1:length(_out)],ts)
    end
    only_end = length(ts) == 1 && ts[1] == _prob.tspan[2]
  end

  _save_idxs = save_idxs === nothing ? Colon() : save_idxs

  function adjoint_sensitivity_backpass(Δ)
    function df(_out, u, p, t, i)
      outtype = typeof(_out) <: SubArray ? DiffEqBase.parameterless_type(_out.parent) : DiffEqBase.parameterless_type(_out)
      if only_end
        eltype(Δ) <: NoTangent && return
        if typeof(Δ) <: AbstractArray{<:AbstractArray} && length(Δ) == 1 && i == 1
          # user did sol[end] on only_end
          if typeof(_save_idxs) <: Number
            x = vec(Δ[1])
            _out[_save_idxs] .= adapt(outtype,@view(x[_save_idxs]))
          elseif _save_idxs isa Colon
            vec(_out) .= adapt(outtype,vec(Δ[1]))
          else
            vec(@view(_out[_save_idxs])) .= adapt(outtype,vec(Δ[1])[_save_idxs])
        end
        else
          Δ isa NoTangent && return
          if typeof(_save_idxs) <: Number
            x = vec(Δ)
            _out[_save_idxs] .= adapt(outtype,@view(x[_save_idxs]))
          elseif _save_idxs isa Colon
            vec(_out) .= adapt(outtype,vec(Δ))
          else
            x = vec(Δ)
            vec(@view(_out[_save_idxs])) .= adapt(outtype,@view(x[_save_idxs]))
          end
        end
      else
        !Base.isconcretetype(eltype(Δ)) && (Δ[i] isa NoTangent || eltype(Δ) <: NoTangent) && return
        if typeof(Δ) <: AbstractArray{<:AbstractArray} || typeof(Δ) <: DESolution
          x = Δ[i]
          if typeof(_save_idxs) <: Number
            _out[_save_idxs] = @view(x[_save_idxs])
          elseif _save_idxs isa Colon
            vec(_out) .= vec(x)
          else
            vec(@view(_out[_save_idxs])) .= vec(@view(x[_save_idxs]))
          end
        else
          if typeof(_save_idxs) <: Number
            _out[_save_idxs] = adapt(outtype,reshape(Δ, prod(size(Δ)[1:end-1]), size(Δ)[end])[_save_idxs, i])
          elseif _save_idxs isa Colon
            vec(_out) .= vec(adapt(outtype,reshape(Δ, prod(size(Δ)[1:end-1]), size(Δ)[end])[:, i]))
          else
            vec(@view(_out[_save_idxs])) .= vec(adapt(outtype,reshape(Δ, prod(size(Δ)[1:end-1]), size(Δ)[end])[:, i]))
          end
        end
      end
    end

    if haskey(kwargs_adj, :callback_adj)
      cb2 = CallbackSet(cb,kwargs[:callback_adj])
    else
      cb2 = cb
    end

    du0, dp = adjoint_sensitivities(sol,alg,args...,df,ts; sensealg=sensealg,
                                    callback = cb2,
                                    kwargs_adj...)

    du0 = reshape(du0,size(u0))
    dp = p === nothing || p === DiffEqBase.NullParameters() ? nothing : reshape(dp',size(p))

    if originator isa SciMLBase.TrackerOriginator || originator isa SciMLBase.ReverseDiffOriginator
      (NoTangent(),NoTangent(),du0,dp,NoTangent(),ntuple(_->NoTangent(), length(args))...)
    else
      (NoTangent(),NoTangent(),NoTangent(),du0,dp,NoTangent(),ntuple(_->NoTangent(), length(args))...)
    end
  end
  out, adjoint_sensitivity_backpass
end

# Prefer this route since it works better with callback AD
function DiffEqBase._concrete_solve_adjoint(prob, alg, sensealg::AbstractForwardSensitivityAlgorithm,
                                            u0, p, originator::SciMLBase.ADOriginator, args...;
                                            save_idxs=nothing,
                                            kwargs...)

  if !(typeof(p) <: Union{Nothing,SciMLBase.NullParameters,AbstractArray}) || (p isa AbstractArray && !Base.isconcretetype(eltype(p)))
    throw(ForwardSensitivityParameterCompatibilityError())
  end

  if p isa AbstractArray && eltype(p) <: ForwardDiff.Dual && !(eltype(u0) <: ForwardDiff.Dual)
    # Handle double differentiation case
    u0 = eltype(p).(u0)
  end
  _prob = ODEForwardSensitivityProblem(prob.f, u0, prob.tspan, p, sensealg)
  sol = solve(_prob, alg, args...; kwargs...)
  _, du = extract_local_sensitivities(sol, sensealg, Val(true))

  u = if save_idxs === nothing
    [reshape(sol[i][1:length(u0)], size(u0)) for i in 1:length(sol)]
  else
    [sol[i][_save_idxs] for i in 1:length(sol)]
  end
  out = DiffEqBase.sensitivity_solution(sol, u, sol.t)

  function forward_sensitivity_backpass(Δ)
    adj = sum(eachindex(du)) do i
      J = du[i]
      if Δ isa AbstractVector || Δ isa DESolution || Δ isa AbstractVectorOfArray
        v = Δ[i]
      elseif Δ isa AbstractMatrix
        v = @view Δ[:, i]
      else
        v = @view Δ[.., i]
      end
      J'vec(v)
    end

    du0 = @not_implemented(
      "ForwardSensitivity does not differentiate with respect to u0. Change your sensealg."
    )

    if originator isa SciMLBase.TrackerOriginator || originator isa SciMLBase.ReverseDiffOriginator
      (NoTangent(), NoTangent(), du0, adj, NoTangent(), ntuple(_ -> NoTangent(), length(args))...)
    else
      (NoTangent(), NoTangent(), NoTangent(), du0, adj, NoTangent(), ntuple(_ -> NoTangent(), length(args))...)
    end
  end
  out, forward_sensitivity_backpass
end

function DiffEqBase._concrete_solve_forward(prob,alg,
                                 sensealg::AbstractForwardSensitivityAlgorithm,
                                 u0,p,originator::SciMLBase.ADOriginator,args...;save_idxs = nothing,
                                 kwargs...)
   _prob = ODEForwardSensitivityProblem(prob.f,u0,prob.tspan,p,sensealg)
   sol = solve(_prob,args...;kwargs...)
   u,du = extract_local_sensitivities(sol,Val(true))
   _save_idxs = save_idxs === nothing ? (1:length(u0)) : save_idxs
   out = DiffEqBase.sensitivity_solution(sol,[ForwardDiff.value.(sol[i][_save_idxs]) for i in 1:length(sol)],sol.t)
   function _concrete_solve_pushforward(Δself, ::Nothing, ::Nothing, x3, Δp, args...)
     x3 !== nothing && error("Pushforward currently requires no u0 derivatives")
     du * Δp
   end
   out,_concrete_solve_pushforward
end

const FORWARDDIFF_SENSITIVITY_PARAMETER_COMPATABILITY_MESSAGE =
"""
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
  print(io, FORWARDDIFF_SENSITIVITY_PARAMETER_COMPATABILITY_MESSAGE)
end

# Generic Fallback for ForwardDiff
function DiffEqBase._concrete_solve_adjoint(prob,alg,
                                 sensealg::ForwardDiffSensitivity{CS,CTS},
                                 u0,p,originator::SciMLBase.ADOriginator,args...;saveat=eltype(prob.tspan)[],
                                 kwargs...) where {CS,CTS}

  if !(typeof(p) <: Union{Nothing,SciMLBase.NullParameters,AbstractArray}) || (p isa AbstractArray && !Base.isconcretetype(eltype(p)))
    throw(ForwardDiffSensitivityParameterCompatibilityError())
  end

  if saveat isa Number
    _saveat = prob.tspan[1]:saveat:prob.tspan[2]
  else
    _saveat = saveat
  end

  sol = solve(remake(prob,p=p,u0=u0),alg,args...;saveat=_saveat, kwargs...)

  # saveat values
  # seems overcomplicated, but see the PR
  if length(sol.t) == 1
      ts = sol.t
  else
      ts = eltype(sol.t)[]
      if sol.t[2] != sol.t[1]
          push!(ts,sol.t[1])
      end
      for i in 2:length(sol.t)-1
          if sol.t[i] != sol.t[i+1] && sol.t[i] != sol.t[i-1]
              push!(ts,sol.t[i])
          end
      end
      if sol.t[end] != sol.t[end-1]
          push!(ts,sol.t[end])
      end
  end

  function forward_sensitivity_backpass(Δ)
    dp = @thunk begin

        chunk_size = if CS === 0 && length(p) < 12
            length(p)
        elseif CS !== 0
            CS
        else
            12
        end

        num_chunks = length(p) ÷ chunk_size
        num_chunks * chunk_size != length(p) && (num_chunks += 1)

        pparts = typeof(p[1:1])[]
        for j in 0:(num_chunks-1)

            local chunk
            if ((j+1)*chunk_size) <= length(p)
                chunk = ((j*chunk_size+1) : ((j+1)*chunk_size))
                pchunk = vec(p)[chunk]
                pdualpart = seed_duals(pchunk,prob.f,ForwardDiff.Chunk{chunk_size}())
            else
                chunk = ((j*chunk_size+1) : length(p))
                pchunk = vec(p)[chunk]
                pdualpart = seed_duals(pchunk,prob.f,ForwardDiff.Chunk{length(chunk)}())
            end

            pdualvec = if j == 0
                vcat(pdualpart,p[(j+1)*chunk_size+1 : end])
            elseif j == num_chunks-1
                vcat(p[1:j*chunk_size],pdualpart)
            else
                vcat(p[1:j*chunk_size],pdualpart,p[((j+1)*chunk_size)+1 : end])
            end

            pdual = ArrayInterfaceCore.restructure(p,pdualvec)
            u0dual = convert.(eltype(pdualvec),u0)

            if (convert_tspan(sensealg) === nothing && (
                  (haskey(kwargs,:callback) && has_continuous_callback(kwargs[:callback])))) ||
                  (convert_tspan(sensealg) !== nothing && convert_tspan(sensealg))

              tspandual = convert.(eltype(pdual),prob.tspan)
            else
              tspandual = prob.tspan
            end

            if typeof(prob.f) <: ODEFunction && prob.f.jac_prototype !== nothing
              _f = ODEFunction{SciMLBase.isinplace(prob.f),true}(prob.f,jac_prototype = convert.(eltype(u0dual),prob.f.jac_prototype))
            elseif typeof(prob.f) <: SDEFunction && prob.f.jac_prototype !== nothing
              _f = SDEFunction{SciMLBase.isinplace(prob.f),true}(prob.f,jac_prototype = convert.(eltype(u0dual),prob.f.jac_prototype))
            else
              _f = prob.f
            end
            _prob = remake(prob,f=_f,u0=u0dual,p=pdual,tspan=tspandual)

            if _prob isa SDEProblem
              _prob.noise_rate_prototype!==nothing && (_prob = remake(_prob, noise_rate_prototype = convert.(eltype(pdual), _prob.noise_rate_prototype)))
            end

            if saveat isa Number
              _saveat = prob.tspan[1]:saveat:prob.tspan[2]
            else
              _saveat = saveat
            end

            _sol = solve(_prob,alg,args...;saveat=ts, kwargs...)
            _,du = extract_local_sensitivities(_sol, sensealg, Val(true))

            _dp = sum(eachindex(du)) do i
              J = du[i]
              if Δ isa AbstractVector || Δ isa DESolution || Δ isa AbstractVectorOfArray
                v = Δ[i]
              elseif Δ isa AbstractMatrix
                v = @view Δ[:, i]
              else
                v = @view Δ[.., i]
              end
              if !(Δ isa NoTangent)
                ForwardDiff.value.(J'vec(v))
              else
                zero(p)
              end
            end
            push!(pparts,vec(_dp))
        end
        ArrayInterfaceCore.restructure(p,reduce(vcat,pparts))
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

        du0parts = typeof(u0[1:1])[]
        for j in 0:(num_chunks-1)

            local chunk
            if ((j+1)*chunk_size) <= length(u0)
                chunk = ((j*chunk_size+1) : ((j+1)*chunk_size))
                u0chunk = vec(u0)[chunk]
                u0dualpart = seed_duals(u0chunk,prob.f,ForwardDiff.Chunk{chunk_size}())
            else
                chunk = ((j*chunk_size+1) : length(u0))
                u0chunk = vec(u0)[chunk]
                u0dualpart = seed_duals(u0chunk,prob.f,ForwardDiff.Chunk{length(chunk)}())
            end

            u0dualvec = if j == 0
                vcat(u0dualpart,u0[(j+1)*chunk_size+1 : end])
            elseif j == num_chunks-1
                vcat(u0[1:j*chunk_size],u0dualpart)
            else
                vcat(u0[1:j*chunk_size],u0dualpart,u0[((j+1)*chunk_size)+1 : end])
            end

            u0dual = ArrayInterfaceCore.restructure(u0,u0dualvec)
            pdual = convert.(eltype(u0dual),p)

            if (convert_tspan(sensealg) === nothing && (
                  (haskey(kwargs,:callback) && has_continuous_callback(kwargs[:callback])))) ||
                  (convert_tspan(sensealg) !== nothing && convert_tspan(sensealg))

              tspandual = convert.(eltype(pdual),prob.tspan)
            else
              tspandual = prob.tspan
            end

            if typeof(prob.f) <: ODEFunction && prob.f.jac_prototype !== nothing
              _f = ODEFunction{SciMLBase.isinplace(prob.f),true}(prob.f,jac_prototype = convert.(eltype(pdual),prob.f.jac_prototype))
            elseif typeof(prob.f) <: SDEFunction && prob.f.jac_prototype !== nothing
              _f = SDEFunction{SciMLBase.isinplace(prob.f),true}(prob.f,jac_prototype = convert.(eltype(pdual),prob.f.jac_prototype))
            else
              _f = prob.f
            end
            _prob = remake(prob,f=_f,u0=u0dual,p=pdual,tspan=tspandual)

            if _prob isa SDEProblem
              _prob.noise_rate_prototype!==nothing && (_prob = remake(_prob, noise_rate_prototype = convert.(eltype(pdual), _prob.noise_rate_prototype)))
            end

            if saveat isa Number
              _saveat = prob.tspan[1]:saveat:prob.tspan[2]
            else
              _saveat = saveat
            end

            _sol = solve(_prob,alg,args...;saveat=ts, kwargs...)
            _,du = extract_local_sensitivities(_sol, sensealg, Val(true))

            _du0 = sum(eachindex(du)) do i
              J = du[i]
              if Δ isa AbstractVector || Δ isa DESolution || Δ isa AbstractVectorOfArray
                v = Δ[i]
              elseif Δ isa AbstractMatrix
                v = @view Δ[:, i]
              else
                v = @view Δ[.., i]
              end
              if !(Δ isa NoTangent)
                ForwardDiff.value.(J'vec(v))
              else
                zero(u0)
              end
            end
            push!(du0parts,vec(_du0))
        end
        ArrayInterfaceCore.restructure(u0,reduce(vcat,du0parts))
    end

    if originator isa SciMLBase.TrackerOriginator || originator isa SciMLBase.ReverseDiffOriginator
      (NoTangent(),NoTangent(),unthunk(du0),unthunk(dp),NoTangent(),ntuple(_->NoTangent(), length(args))...)
    else
      (NoTangent(),NoTangent(),NoTangent(),du0,dp,NoTangent(),ntuple(_->NoTangent(), length(args))...)
    end
  end
  sol,forward_sensitivity_backpass
end

function DiffEqBase._concrete_solve_adjoint(prob,alg,sensealg::ZygoteAdjoint,
                                 u0,p,originator::SciMLBase.ADOriginator,args...;kwargs...)
    Zygote.pullback((u0,p)->solve(prob,alg,args...;u0=u0,p=p,
                    sensealg = SensitivityADPassThrough(),kwargs...),u0,p)
end

function DiffEqBase._concrete_solve_adjoint(prob,alg,sensealg::TrackerAdjoint,
                                            u0,p,originator::SciMLBase.ADOriginator,args...;
                                            kwargs...)

  local sol
  function tracker_adjoint_forwardpass(_u0,_p)

    if (convert_tspan(sensealg) === nothing && (
          (haskey(kwargs,:callback) && has_continuous_callback(kwargs[:callback])))) ||
          (convert_tspan(sensealg) !== nothing && convert_tspan(sensealg))
      _tspan = convert.(eltype(_p),prob.tspan)
    else
      _tspan = prob.tspan
    end

    if DiffEqBase.isinplace(prob)
      # use Array{TrackedReal} for mutation to work
      # Recurse to all Array{TrackedArray}
      _prob = remake(prob,u0=map(identity,_u0),p=_p,tspan=_tspan)
    else
      # use TrackedArray for efficiency of the tape
      if typeof(prob) <: Union{SciMLBase.AbstractDDEProblem,SciMLBase.AbstractDAEProblem,SciMLBase.AbstractSDDEProblem}
        _f = function (u,p,h,t) # For DDE, but also works for (du,u,p,t) DAE
          out = prob.f(u,p,h,t)
          if out isa TrackedArray
            return out
          else
            Tracker.collect(out)
          end
        end

        # Only define `g` for the stochastic ones
        if typeof(prob) <: SciMLBase.AbstractSDEProblem
          _g = function (u,p,h,t)
            out = prob.g(u,p,h,t)
            if out isa TrackedArray
              return out
            else
              Tracker.collect(out)
            end
          end
          _prob = remake(prob,f=DiffEqBase.parameterless_type(prob.f){false,true}(_f,_g),u0=_u0,p=_p,tspan=_tspan)
        else
          _prob = remake(prob,f=DiffEqBase.parameterless_type(prob.f){false,true}(_f),u0=_u0,p=_p,tspan=_tspan)
        end
      elseif typeof(prob) <: Union{SciMLBase.AbstractODEProblem,SciMLBase.AbstractSDEProblem}
        _f = function (u,p,t)
          out = prob.f(u,p,t)
          if out isa TrackedArray
            return out
          else
            Tracker.collect(out)
          end
        end
        if typeof(prob) <: SciMLBase.AbstractSDEProblem
          _g = function (u,p,t)
            out = prob.g(u,p,t)
            if out isa TrackedArray
              return out
            else
              Tracker.collect(out)
            end
          end
          _prob = remake(prob,f=DiffEqBase.parameterless_type(prob.f){false,true}(_f,_g),u0=_u0,p=_p,tspan=_tspan)
        else
          _prob = remake(prob,f=DiffEqBase.parameterless_type(prob.f){false,true}(_f),u0=_u0,p=_p,tspan=_tspan)
        end
      else
        error("TrackerAdjont does not currently support the specified problem type. Please open an issue.")
      end
    end
    sol = solve(_prob,alg,args...;sensealg=DiffEqBase.SensitivityADPassThrough(),kwargs...)

    if typeof(sol.u[1]) <: Array
      return Array(sol)
    else
      tmp = vec(sol.u[1])
      for i in 2:length(sol.u)
        tmp = hcat(tmp,vec(sol.u[i]))
      end
      return reshape(tmp,size(sol.u[1])...,length(sol.u))
    end
    #adapt(typeof(u0),arr)
    sol
  end

  out,pullback = Tracker.forward(tracker_adjoint_forwardpass,u0,p)
  function tracker_adjoint_backpass(ybar)
    tmp = if eltype(ybar) <: Number && typeof(u0) <: Array
      Array(ybar)
    elseif eltype(ybar) <: Number # CuArray{Floats}
      ybar
    elseif typeof(ybar[1]) <: Array
      return Array(ybar)
    else
      tmp = vec(ybar.u[1])
      for i in 2:length(ybar.u)
        tmp = hcat(tmp,vec(ybar.u[i]))
      end
      return reshape(tmp,size(ybar.u[1])...,length(ybar.u))
    end
    u0bar, pbar = pullback(tmp)
    _u0bar = u0bar isa Tracker.TrackedArray ? Tracker.data(u0bar) : Tracker.data.(u0bar)

    if originator isa SciMLBase.TrackerOriginator || originator isa SciMLBase.ReverseDiffOriginator
      (NoTangent(),NoTangent(),_u0bar,Tracker.data(pbar),NoTangent(),ntuple(_->NoTangent(), length(args))...)
    else
      (NoTangent(),NoTangent(),NoTangent(),_u0bar,Tracker.data(pbar),NoTangent(),ntuple(_->NoTangent(), length(args))...)
    end
  end

  u = u0 isa Tracker.TrackedArray ? Tracker.data.(sol.u) : Tracker.data.(Tracker.data.(sol.u))
  DiffEqBase.sensitivity_solution(sol,u,Tracker.data.(sol.t)),tracker_adjoint_backpass
end

const REVERSEDIFF_ADJOINT_GPU_COMPATABILITY_MESSAGE =
"""
ReverseDiffAdjoint is not compatible GPU-based array types. Use a different
sensitivity analysis method, like InterpolatingAdjoint or TrackerAdjoint,
in order to combine with GPUs.
"""

struct ReverseDiffGPUStateCompatibilityError <: Exception end

function Base.showerror(io::IO, e::ReverseDiffGPUStateCompatibilityError)
  print(io, FORWARDDIFF_SENSITIVITY_PARAMETER_COMPATABILITY_MESSAGE)
end

function DiffEqBase._concrete_solve_adjoint(prob,alg,sensealg::ReverseDiffAdjoint,
                                            u0,p,originator::SciMLBase.ADOriginator,args...;kwargs...)

  if typeof(u0) isa GPUArraysCore.AbstractGPUArray
    throw(ReverseDiffGPUStateCompatibilityError())
  end

  t = eltype(prob.tspan)[]
  u = typeof(u0)[]

  local sol

  function reversediff_adjoint_forwardpass(_u0,_p)

    if (convert_tspan(sensealg) === nothing && (
          (haskey(kwargs,:callback) && has_continuous_callback(kwargs[:callback])))) ||
          (convert_tspan(sensealg) !== nothing && convert_tspan(sensealg))
      _tspan = convert.(eltype(_p),prob.tspan)
    else
      _tspan = prob.tspan
    end

    if DiffEqBase.isinplace(prob)
      # use Array{TrackedReal} for mutation to work
      # Recurse to all Array{TrackedArray}
      _prob = remake(prob,u0=reshape([x for x in _u0],size(_u0)),p=_p,tspan=_tspan)
    else
      # use TrackedArray for efficiency of the tape
      _f(args...) = reduce(vcat,prob.f(args...))
      if prob isa SDEProblem
        _g(args...) = reduce(vcat,prob.g(args...))
        _prob = remake(prob,f=DiffEqBase.parameterless_type(prob.f){SciMLBase.isinplace(prob),true}(_f,_g),u0=_u0,p=_p,tspan=_tspan)
      else
        _prob = remake(prob,f=DiffEqBase.parameterless_type(prob.f){SciMLBase.isinplace(prob),true}(_f),u0=_u0,p=_p,tspan=_tspan)
      end
    end

    sol = solve(_prob,alg,args...;sensealg=DiffEqBase.SensitivityADPassThrough(),kwargs...)
    t = sol.t
    if DiffEqBase.isinplace(prob)
      u = map.(ReverseDiff.value,sol.u)
    else
      u = map(ReverseDiff.value,sol.u)
    end
    Array(sol)
  end

  tape = ReverseDiff.GradientTape(reversediff_adjoint_forwardpass,(u0, p))
  tu, tp = ReverseDiff.input_hook(tape)
  output = ReverseDiff.output_hook(tape)
  ReverseDiff.value!(tu, u0)
  typeof(p) <: DiffEqBase.NullParameters || ReverseDiff.value!(tp, p)
  ReverseDiff.forward_pass!(tape)
  function reversediff_adjoint_backpass(ybar)
    _ybar = if ybar isa VectorOfArray
        Array(ybar)
      elseif eltype(ybar) <: AbstractArray
        Array(VectorOfArray(ybar))
      else
        ybar
      end
    ReverseDiff.increment_deriv!(output, _ybar)
    ReverseDiff.reverse_pass!(tape)

    if originator isa SciMLBase.TrackerOriginator || originator isa SciMLBase.ReverseDiffOriginator
      (NoTangent(),NoTangent(),ReverseDiff.deriv(tu),ReverseDiff.deriv(tp),NoTangent(),ntuple(_->NoTangent(), length(args))...)
    else
      (NoTangent(),NoTangent(),NoTangent(),ReverseDiff.deriv(tu),ReverseDiff.deriv(tp),NoTangent(),ntuple(_->NoTangent(), length(args))...)
    end
  end
  Array(VectorOfArray(u)),reversediff_adjoint_backpass
end


function DiffEqBase._concrete_solve_adjoint(prob,alg,
                                 sensealg::AbstractShadowingSensitivityAlgorithm,
                                 u0,p,originator::SciMLBase.ADOriginator,args...;save_start=true,save_end=true,
                                 saveat = eltype(prob.tspan)[],
                                 save_idxs = nothing,
                                 kwargs...)

  if haskey(kwargs, :callback)
    error("Sensitivity analysis based on Least Squares Shadowing is not compatible with callbacks. Please select another `sensealg`.")
  else
    _prob = remake(prob,u0=u0,p=p)
  end

  sol = solve(_prob,alg,args...;save_start=save_start,save_end=save_end,saveat=saveat,kwargs...)

  if saveat isa Number
    if _prob.tspan[2] > _prob.tspan[1]
      ts = _prob.tspan[1]:convert(typeof(_prob.tspan[2]),abs(saveat)):_prob.tspan[2]
    else
      ts = _prob.tspan[2]:convert(typeof(_prob.tspan[2]),abs(saveat)):_prob.tspan[1]
    end
     _out = sol(ts)
    out = if save_idxs === nothing
      out = DiffEqBase.sensitivity_solution(sol,_out.u,sol.t)
    else
      out = DiffEqBase.sensitivity_solution(sol,[_out[i][save_idxs] for i in 1:length(_out)],ts)
    end
    # only_end
    (length(ts) == 1 && ts[1] == _prob.tspan[2]) && error("Sensitivity analysis based on Least Squares Shadowing requires a long-time averaged quantity.")
  elseif isempty(saveat)
    no_start = !save_start
    no_end = !save_end
    sol_idxs = 1:length(sol)
    no_start && (sol_idxs = sol_idxs[2:end])
    no_end && (sol_idxs = sol_idxs[1:end-1])
    only_end = length(sol_idxs) <= 1
    _u = sol.u[sol_idxs]
    u = save_idxs === nothing ? _u : [x[save_idxs] for x in _u]
    ts = sol.t[sol_idxs]
    out = DiffEqBase.sensitivity_solution(sol,u,ts)
  else
    _saveat = saveat isa Array ? sort(saveat) : saveat # for minibatching
    ts = _saveat
    _out = sol(ts)

    out = if save_idxs === nothing
      out = DiffEqBase.sensitivity_solution(sol,_out.u,ts)
    else
      out = DiffEqBase.sensitivity_solution(sol,[_out[i][save_idxs] for i in 1:length(_out)],ts)
    end
    # only_end
    (length(ts) == 1 && ts[1] == _prob.tspan[2]) && error("Sensitivity analysis based on Least Squares Shadowing requires a long-time averaged quantity.")
  end

  _save_idxs = save_idxs === nothing ? Colon() : save_idxs

  function adjoint_sensitivity_backpass(Δ)
    function df(_out, u, p, t, i)
      if typeof(Δ) <: AbstractArray{<:AbstractArray} || typeof(Δ) <: DESolution
        if typeof(_save_idxs) <: Number
          _out[_save_idxs] = Δ[i][_save_idxs]
        elseif _save_idxs isa Colon
          vec(_out) .= vec(Δ[i])
        else
          vec(@view(_out[_save_idxs])) .= vec(Δ[i][_save_idxs])
        end
      else
        if typeof(_save_idxs) <: Number
          _out[_save_idxs] = adapt(DiffEqBase.parameterless_type(u0),reshape(Δ, prod(size(Δ)[1:end-1]), size(Δ)[end])[_save_idxs, i])
        elseif _save_idxs isa Colon
          vec(_out) .= vec(adapt(DiffEqBase.parameterless_type(u0),reshape(Δ, prod(size(Δ)[1:end-1]), size(Δ)[end])[:, i]))
        else
          vec(@view(_out[_save_idxs])) .= vec(adapt(DiffEqBase.parameterless_type(u0),reshape(Δ, prod(size(Δ)[1:end-1]), size(Δ)[end])[:, i]))
        end
      end
    end

    if sensealg isa ForwardLSS
      lss_problem = ForwardLSSProblem(sol, sensealg, ts, df)
      dp = shadow_forward(lss_problem)
    elseif sensealg isa AdjointLSS
      adjointlss_problem = AdjointLSSProblem(sol, sensealg, ts, df)
      dp = shadow_adjoint(adjointlss_problem)
    elseif sensealg isa NILSS
      nilss_prob = NILSSProblem(_prob, sensealg, ts, df)
      dp = shadow_forward(nilss_prob,alg)
    elseif sensealg isa NILSAS
      nilsas_prob = NILSASProblem(_prob, sensealg, ts, df)
      dp = shadow_adjoint(nilsas_prob,alg)
    else
      error("No concrete_solve implementation found for sensealg `$sensealg`. Did you spell the sensitivity algorithm correctly? Please report this error.")
    end

    if originator isa SciMLBase.TrackerOriginator || originator isa SciMLBase.ReverseDiffOriginator
      (NoTangent(),NoTangent(),NoTangent(),dp,NoTangent(),ntuple(_->NoTangent(), length(args))...)
    else
      (NoTangent(),NoTangent(),NoTangent(),NoTangent(),dp,NoTangent(),ntuple(_->NoTangent(), length(args))...)
    end
  end
  out, adjoint_sensitivity_backpass
end

function DiffEqBase._concrete_solve_adjoint(prob::Union{NonlinearProblem,SteadyStateProblem},
                                            alg,sensealg::SteadyStateAdjoint,
                                            u0,p,originator::SciMLBase.ADOriginator,args...;save_idxs = nothing, kwargs...)

    _prob = remake(prob,u0=u0,p=p)
    sol = solve(_prob,alg,args...;kwargs...)
    _save_idxs = save_idxs === nothing ? Colon() : save_idxs

    if save_idxs === nothing
      out = sol
    else
      out = DiffEqBase.sensitivity_solution(sol,sol[_save_idxs])
    end

    function steadystatebackpass(Δ)
      # Δ = dg/dx or diffcache.dg_val
      # del g/del p = 0
      dp = adjoint_sensitivities(sol,alg;sensealg=sensealg,g=nothing,dg=Δ,save_idxs=save_idxs)

      if originator isa SciMLBase.TrackerOriginator || originator isa SciMLBase.ReverseDiffOriginator
        (NoTangent(),NoTangent(),NoTangent(),dp,NoTangent(),ntuple(_->NoTangent(), length(args))...)
      else
        (NoTangent(),NoTangent(),NoTangent(),NoTangent(),dp,NoTangent(),ntuple(_->NoTangent(), length(args))...)
      end
    end
    out, steadystatebackpass
end

function fix_endpoints(sensealg,sol,ts)
  @warn "Endpoints do not match. Return code: $(sol.retcode). Likely your time range is not a multiple of `saveat`. sol.t[end]: $(sol.t[end]), ts[end]: $(ts[end])"
  ts = collect(ts)
  push!(ts, sol.t[end])
end
