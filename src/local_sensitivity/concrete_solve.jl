## High level

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
    sol = solve(_prob,alg,args...;save_start=true,save_end=true,saveat=saveat,kwargs...)
  else
    sol = solve(_prob,alg,args...;save_start=true,save_end=true,kwargs...)
  end

  if saveat isa Number
    if _prob.tspan[2] > _prob.tspan[1]
      ts = _prob.tspan[1]:abs(saveat):_prob.tspan[2]
    else
      ts = _prob.tspan[2]:abs(saveat):_prob.tspan[1]
    end
    _out = sol(ts)
    out = save_idxs === nothing ? _out : DiffEqArray([x[save_idxs] for x in _out.u],ts)
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
    out = DiffEqArray(u,ts)
  else
    _saveat = saveat isa Array ? sort(saveat) : saveat # for minibatching
    ts = _saveat
    _out = sol(ts)
    out = save_idxs === nothing ? _out : DiffEqArray([x[save_idxs] for x in _out.u],ts)
    only_end = length(ts) == 1 && ts[1] == _prob.tspan[2]
  end

  function adjoint_sensitivity_backpass(Δ)
    function df(_out, u, p, t, i)
      if only_end
        _out[:] .= -vec(Δ)
      else
        if typeof(Δ) <: AbstractArray{<:AbstractArray}
          _out[:] .= -Δ[i]
        else
          _out[:] .= -adapt(typeof(u0),reshape(Δ, prod(size(Δ)[1:end-1]), size(Δ)[end])[:, i])
        end
      end
    end

    du0, dp = adjoint_sensitivities(sol,alg,args...,df,ts; sensealg=sensealg,
                                    kwargs_adj...)

    du0 = reshape(du0,size(u0))
    dp = reshape(dp',size(p))

    (nothing,nothing,du0,dp,ntuple(_->nothing, length(args))...)
  end
  out, adjoint_sensitivity_backpass
end

# Prefer this route since it works better with callback AD
function DiffEqBase._concrete_solve_adjoint(prob,alg,sensealg::AbstractForwardSensitivityAlgorithm,
                                 u0,p,args...;
                                 kwargs...)
   _prob = ODEForwardSensitivityProblem(prob.f,u0,prob.tspan,p,sensealg)
   sol = solve(_prob,alg,args...;kwargs...)
   u,du = extract_local_sensitivities(sol, Val(true))
   function forward_sensitivity_backpass(Δ)
     adj = sum(eachindex(du)) do i
       J = du[i]
       v = @view Δ[:, i]
       J'v
     end
     (nothing,nothing,nothing,adj,ntuple(_->nothing, length(args))...)
   end
   DiffEqArray([u[:,i] for i in 1:size(u,2)],sol.t),forward_sensitivity_backpass
end

function DiffEqBase._concrete_solve_forward(prob,alg,
                                 sensealg::AbstractForwardSensitivityAlgorithm,
                                 u0,p,args...;
                                 kwargs...)
   _prob = ODEForwardSensitivityProblem(prob.f,u0,prob.tspan,p,sensealg)
   sol = solve(_prob,args...;kwargs...)
   u,du = extract_local_sensitivities(sol,Val(true))
   function _concrete_solve_pushforward(Δself, ::Nothing, ::Nothing, x3, Δp, args...)
     x3 !== nothing && error("Pushforward currently requires no u0 derivatives")
     du * Δp
   end
   DiffEqArray([u[:,i] for i in 1:size(u,2)],sol.t),_concrete_solve_pushforward
end

# Generic Fallback for ForwardDiff
function DiffEqBase._concrete_solve_adjoint(prob,alg,
                                 sensealg::ForwardDiffSensitivity,
                                 u0,p,args...;saveat=eltype(prob.tspan)[],
                                 save_idxs = nothing,
                                 kwargs...)
  save_idxs !== nothing && error("save_idxs is currently incompatible with ForwardDiffSensitivity")
  MyTag = typeof(prob.f)
  pdual = seed_duals(p,MyTag)
  u0dual = convert.(eltype(pdual),u0)

  if (convert_tspan(sensealg) === nothing && (
        (haskey(kwargs,:callback) && has_continuous_callback(kwargs.callback)) ||
        (haskey(prob.kwargs,:callback) && has_continuous_callback(prob.kwargs.callback))
        )) || (convert_tspan(sensealg) !== nothing && convert_tspan(alg))

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

  u,du = extract_local_sensitivities(sol, sensealg, Val(true))
  function forward_sensitivity_backpass(Δ)
    adj = sum(eachindex(du)) do i
      J = du[i]
      v = @view Δ[:, i]
      J'v
    end
    (nothing,nothing,nothing,adj,ntuple(_->nothing, length(args))...)
  end
  DiffEqArray([u[:,i] for i in 1:size(u,2)],sol.t),forward_sensitivity_backpass
end

function DiffEqBase._concrete_solve_adjoint(prob,alg,sensealg::ZygoteAdjoint,
                                 u0,p,args...;kwargs...)
    Zygote.pullback((u0,p)->_concrete_solve(prob,alg,u0,p,args...;kwargs...),u0,p)
end

function DiffEqBase._concrete_solve_adjoint(prob,alg,sensealg::TrackerAdjoint,
                                            u0,p,args...;kwargs...)

  t = eltype(prob.tspan)[]
  u = typeof(u0)[]

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
    sol = solve(_prob,alg,args...;kwargs...)
    t = sol.t
    if DiffEqBase.isinplace(prob)
      u = map.(Tracker.data,sol.u)
    else
      u = map(Tracker.data,sol.u)
    end

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
  end

  sol,pullback = Tracker.forward(tracker_adjoint_forwardpass,u0,p)
  function tracker_adjoint_backpass(ybar)
    u0bar, pbar = pullback(ybar)
    _u0bar = u0bar isa Tracker.TrackedArray ? Tracker.data(u0bar) : Tracker.data.(u0bar)
    (nothing,nothing,_u0bar,Tracker.data(pbar),ntuple(_->nothing, length(args))...)
  end
  DiffEqArray(u,t),tracker_adjoint_backpass
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
    sol = solve(_prob,alg,args...;kwargs...)
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
    (nothing,nothing,ReverseDiff.deriv(tu),ReverseDiff.deriv(tp)',ntuple(_->nothing, length(args))...)
  end
  DiffEqArray(u,t),tracker_adjoint_backpass
end


function DiffEqBase._concrete_solve_adjoint(prob::SteadyStateProblem,alg,sensealg::SteadyStateAdjoint,
                                 u0,p,args...;kwargs...)

    #_prob = remake(prob,u0=u0,p=p)
    # sol = solve(_prob,alg)
    sol = solve(prob,alg)
    function steadystatebackpass(Δ)
      # Δ = dg/dx or diffcache.dg_val
      # del g/del p = 0
      dp = adjoint_sensitivities(sol,alg;sensealg=sensealg,g=nothing,dg=Δ)
      (nothing,nothing,nothing,dp,ntuple(_->nothing, length(args))...)
    end
    sol.u, steadystatebackpass
end
