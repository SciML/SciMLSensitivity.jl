## High level

struct SensitivityADPassThrough2 end

# Here is where we can add a default algorithm for computing sensitivities
# Based on problem information!
function DiffEqBase._concrete_solve_adjoint(prob,alg,sensealg::Nothing,u0,p,args...;kwargs...)
  default_sensealg = (isgpu(u0) && !DiffEqBase.isinplace(prob)) ?
                                  InterpolatingAdjoint(autojacvec=ZygoteVJP()) :
                                  InterpolatingAdjoint()
  DiffEqBase._concrete_solve_adjoint(prob,alg,default_sensealg,u0,p,args...;kwargs...)
end

function DiffEqBase._concrete_solve_adjoint(prob::SteadyStateProblem,alg,sensealg::Nothing,u0,p,args...;kwargs...)
  default_sensealg = SteadyStateAdjoint()
  DiffEqBase._concrete_solve_adjoint(prob,alg,default_sensealg,u0,p,args...;kwargs...)
end

function DiffEqBase._concrete_solve_adjoint(prob,alg,
                                 sensealg::AbstractAdjointSensitivityAlgorithm,
                                 u0,p,args...;save_start=true,save_end=true,
                                 saveat = eltype(prob.tspan)[],
                                 save_idxs = nothing,
                                 kwargs...)
  _prob = remake(prob,u0=u0,p=p)

  # Force `save_start` and `save_end` in the forward pass This forces the
  # solver to do the backsolve all the way back to `u0` Since the start aliases
  # `_prob.u0`, this doesn't actually use more memory But it cleans up the
  # implementation and makes `save_start` and `save_end` arg safe.
  kwargs_fwd = NamedTuple{Base.diff_names(Base._nt_names(
  values(kwargs)), (:callback_adj,))}(values(kwargs))
  kwargs_adj = NamedTuple{Base.diff_names(Base._nt_names(values(kwargs)), (:callback_adj,:callback))}(values(kwargs))
  if haskey(kwargs, :callback_adj)
    kwargs_adj = merge(kwargs_adj, NamedTuple{(:callback,)}( [get(kwargs, :callback_adj, nothing)] ))
  end

  if ischeckpointing(sensealg)
    sol = solve(_prob,alg,args...;save_noise=true,save_start=true,save_end=true,saveat=saveat,kwargs...)
  else
    sol = solve(_prob,alg,args...;save_noise=true,save_start=true,save_end=true,kwargs...)
  end

  if saveat isa Number
    if _prob.tspan[2] > _prob.tspan[1]
      ts = _prob.tspan[1]:abs(saveat):_prob.tspan[2]
    else
      ts = _prob.tspan[2]:abs(saveat):_prob.tspan[1]
    end
    _out = sol(ts)
    out = if save_idxs === nothing
      out = DiffEqBase.sensitivity_solution(sol,_out.u,sol.t)
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
    ts = _saveat
    _out = sol(ts)
    out = if save_idxs === nothing
      out = DiffEqBase.sensitivity_solution(sol,_out.u,sol.t)
    else
      out = DiffEqBase.sensitivity_solution(sol,[_out[i][save_idxs] for i in 1:length(_out)],ts)
    end
    only_end = length(ts) == 1 && ts[1] == _prob.tspan[2]
  end

  _save_idxs = save_idxs === nothing ? Colon() : save_idxs

  function adjoint_sensitivity_backpass(Δ)
    function df(_out, u, p, t, i)
      if only_end
        if typeof(_save_idxs) <: Number
          _out[_save_idxs] = -vec(Δ)[_save_idxs]
        else
          _out[_save_idxs] .= -vec(Δ)[_save_idxs]
        end
      else
        if typeof(Δ) <: AbstractArray{<:AbstractArray}
          if typeof(_save_idxs) <: Number
            _out[_save_idxs] = -Δ[i][_save_idxs]
          else
            _out[_save_idxs] .= -Δ[i][_save_idxs]
          end
        else
          if typeof(_save_idxs) <: Number
            _out[_save_idxs] = -adapt(typeof(u0),reshape(Δ, prod(size(Δ)[1:end-1]), size(Δ)[end])[_save_idxs, i])
          else
            _out[_save_idxs] .= -adapt(typeof(u0),reshape(Δ, prod(size(Δ)[1:end-1]), size(Δ)[end])[_save_idxs, i])
          end
        end
      end
    end

    du0, dp = adjoint_sensitivities(sol,alg,args...,df,ts; sensealg=sensealg,
                                    kwargs_adj...)

    du0 = reshape(du0,size(u0))
    dp = reshape(dp',size(p))

    (nothing,nothing,du0,dp,nothing,ntuple(_->nothing, length(args))...)
  end
  out, adjoint_sensitivity_backpass
end

# Prefer this route since it works better with callback AD
function DiffEqBase._concrete_solve_adjoint(prob,alg,sensealg::AbstractForwardSensitivityAlgorithm,
                                 u0,p,args...;save_idxs = nothing,
                                 kwargs...)
   _save_idxs = save_idxs === nothing ? (1:length(u0)) : save_idxs
   _prob = ODEForwardSensitivityProblem(prob.f,u0,prob.tspan,p,sensealg)
   sol = solve(_prob,alg,args...;kwargs...)
   _,du = extract_local_sensitivities(sol, sensealg, Val(true))
   out = DiffEqBase.sensitivity_solution(sol,[sol[i][_save_idxs] for i in 1:length(sol)],sol.t)
   function forward_sensitivity_backpass(Δ)
     adj = sum(eachindex(du)) do i
       J = du[i]
       v = @view Δ[:, i]
       J'v
     end
     (nothing,nothing,nothing,adj,nothing,ntuple(_->nothing, length(args))...)
   end
   out,forward_sensitivity_backpass
end

function DiffEqBase._concrete_solve_forward(prob,alg,
                                 sensealg::AbstractForwardSensitivityAlgorithm,
                                 u0,p,args...;save_idxs = nothing,
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

# Generic Fallback for ForwardDiff
function DiffEqBase._concrete_solve_adjoint(prob,alg,
                                 sensealg::ForwardDiffSensitivity,
                                 u0,p,args...;saveat=eltype(prob.tspan)[],
                                 save_idxs = nothing,
                                 kwargs...)
  _save_idxs = save_idxs === nothing ? (1:length(u0)) : save_idxs
  MyTag = typeof(prob.f)
  pdual = seed_duals(p,MyTag)
  u0dual = convert.(eltype(pdual),u0)

  if (convert_tspan(sensealg) === nothing && (
        (haskey(kwargs,:callback) && has_continuous_callback(kwargs.callback)) ||
        (haskey(prob.kwargs,:callback) && has_continuous_callback(prob.kwargs.callback))
        )) || (convert_tspan(sensealg) !== nothing && convert_tspan(sensealg))

    tspandual = convert.(eltype(pdual),prob.tspan)
  else
    tspandual = prob.tspan
  end

  _prob = remake(prob,u0=u0dual,p=pdual,tspan=tspandual)

  if saveat isa Number
    _saveat = prob.tspan[1]:saveat:prob.tspan[2]
  else
    _saveat = saveat
  end

  sol = solve(_prob,alg,args...;saveat=_saveat,kwargs...)
  _,du = extract_local_sensitivities(sol, sensealg, Val(true))
  out = DiffEqBase.sensitivity_solution(sol,[ForwardDiff.value.(sol[i][_save_idxs]) for i in 1:length(sol)],ForwardDiff.value.(sol.t))
  function forward_sensitivity_backpass(Δ)
    adj = sum(eachindex(du)) do i
      J = du[i]
      v = @view Δ[:, i]
      ForwardDiff.value.(J'v)
    end
    (nothing,nothing,nothing,adj,nothing,ntuple(_->nothing, length(args))...)
  end
  out,forward_sensitivity_backpass
end

function DiffEqBase._concrete_solve_adjoint(prob,alg,sensealg::ZygoteAdjoint,
                                 u0,p,args...;kwargs...)
    Zygote.pullback((u0,p)->solve(prob,alg,args...;u0=u0,p=p,kwargs...),u0,p)
end

function DiffEqBase._concrete_solve_adjoint(prob,alg,sensealg::TrackerAdjoint,
                                            u0,p,args...;
                                            kwargs...)

  local sol
  function tracker_adjoint_forwardpass(_u0,_p)
    if DiffEqBase.isinplace(prob)
      # use Array{TrackedReal} for mutation to work
      # Recurse to all Array{TrackedArray}
      _prob = remake(prob,u0=map(identity,_u0),p=_p)
    else
      # use TrackedArray for efficiency of the tape
      _f(args...) = Tracker.collect(prob.f(args...))
      if prob isa SDEProblem
        _g(args...) = Tracker.collect(prob.g(args...))
        _prob = remake(prob,f=DiffEqBase.parameterless_type(prob.f)(_f,_g),u0=_u0,p=_p)
      else
        _prob = remake(prob,f=DiffEqBase.parameterless_type(prob.f)(_f),u0=_u0,p=_p)
      end
    end
    sol = solve(_prob,alg,args...;sensealg=SensitivityADPassThrough2(),kwargs...)

    if typeof(sol.u[1]) <: Array
      return adapt(typeof(u0),sol)
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
    u0bar, pbar = pullback(ybar)
    _u0bar = u0bar isa Tracker.TrackedArray ? Tracker.data(u0bar) : Tracker.data.(u0bar)
    (nothing,nothing,_u0bar,Tracker.data(pbar),nothing,ntuple(_->nothing, length(args))...)
  end

  u = u0 isa Tracker.TrackedArray ? Tracker.data.(sol.u) : Tracker.data.(Tracker.data.(sol.u))
  DiffEqBase.sensitivity_solution(sol,u,Tracker.data.(sol.t)),tracker_adjoint_backpass
end

function DiffEqBase._concrete_solve_adjoint(prob,alg,sensealg::ReverseDiffAdjoint,
                                            u0,p,args...;kwargs...)

  t = eltype(prob.tspan)[]
  u = typeof(u0)[]

  function reversediff_adjoint_forwardpass(_u0,_p)
    if DiffEqBase.isinplace(prob)
      # use Array{TrackedReal} for mutation to work
      # Recurse to all Array{TrackedArray}
      _prob = remake(prob,u0=map(identity,_u0),p=_p)
    else
      # use TrackedArray for efficiency of the tape
      _f(args...) = ReverseDiff.collect(prob.f(args...))
      if prob isa SDEProblem
        _g(args...) = ReverseDiff.collect(prob.g(args...))
        _prob = remake(prob,f=DiffEqBase.parameterless_type(prob.f)(_f,_g),u0=_u0,p=_p)
      else
        _prob = remake(prob,f=DiffEqBase.parameterless_type(prob.f)(_f),u0=_u0,p=_p)
      end
    end
    sol = solve(_prob,alg,args...;sensealg=SensitivityADPassThrough2(),kwargs...)
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
  ReverseDiff.value!(tp, prob.p)
  ReverseDiff.forward_pass!(tape)
  function tracker_adjoint_backpass(ybar)
    ReverseDiff.increment_deriv!(output, ybar)
    ReverseDiff.reverse_pass!(tape)
    (nothing,nothing,ReverseDiff.deriv(tu),ReverseDiff.deriv(tp),nothing,ntuple(_->nothing, length(args))...)
  end
  DiffEqArray(u,t),tracker_adjoint_backpass
end


function DiffEqBase._concrete_solve_adjoint(prob::SteadyStateProblem,alg,sensealg::SteadyStateAdjoint,
                                 u0,p,args...;save_idxs = nothing, kwargs...)

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
      dp
      (nothing,nothing,nothing,dp,nothing,ntuple(_->nothing, length(args))...)
    end
    out, steadystatebackpass
end
