## High level

# Here is where we can add a default algorithm for computing sensitivities
# Based on problem information!
function DiffEqBase._concrete_solve_adjoint(prob,alg,sensealg::Nothing,u0,p,args...;kwargs...)
  default_sensealg = InterpolatingAdjoint()
  DiffEqBase._concrete_solve_adjoint(prob,alg,default_sensealg,u0,p,args...;kwargs...)
end

function DiffEqBase._concrete_solve_adjoint(prob,alg,
                                 sensealg::AbstractAdjointSensitivityAlgorithm,
                                 u0,p,args...;save_start=true,save_end=true,
                                 saveat = eltype(prob.tspan)[],
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
    out = sol(ts)
    only_end = length(ts) == 1 && ts[1] == _prob.tspan[2]
  elseif isempty(saveat)
    no_start = !save_start
    no_end = !save_end
    sol_idxs = 1:length(sol)
    no_start && (sol_idxs = sol_idxs[2:end])
    no_end && (sol_idxs = sol_idxs[1:end-1])
    # If didn't save start, take off first. If only wanted the end, return vector
    only_end = length(sol_idxs) <= 1
    u = sol[sol_idxs]
    only_end && (sol_idxs = length(sol))
    out = only_end ? sol[end] : reduce((x,y)->cat(x,y,dims=ndims(u)),u.u)
    ts = sol.t[sol_idxs]
  else
    ts = saveat
    out = sol(ts)
    only_end = length(ts) == 1 && ts[1] == _prob.tspan[2]
  end

  function adjoint_sensitivity_backpass(Δ)
    function df(out, u, p, t, i)
      if only_end
        out[:] .= -vec(Δ)
      else
        out[:] .= -reshape(Δ, :, size(Δ)[end])[:, i]
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
                                 u0,p,args...;kwargs...)
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
   u,forward_sensitivity_backpass
end

function DiffEqBase._concrete_solve_forward(prob,alg,
                                 sensealg::AbstractForwardSensitivityAlgorithm,
                                 u0,p,args...;kwargs...)
   _prob = ODEForwardSensitivityProblem(prob.f,u0,prob.tspan,p,sensealg)
   sol = solve(_prob,args...;kwargs...)
   u,du = extract_local_sensitivities(sol,Val(true))
   function _concrete_solve_pushforward(Δself, ::Nothing, ::Nothing, x3, Δp, args...)
     x3 !== nothing && error("Pushforward currently requires no u0 derivatives")
     du * Δp
   end
   DiffEqArray(u,sol.t),_concrete_solve_pushforward
end

# Generic Fallback for ForwardDiff
function DiffEqBase._concrete_solve_adjoint(prob,alg,
                                 sensealg::ForwardDiffSensitivity,
                                 u0,p,args...;saveat=eltype(prob.tspan)[],
                                 kwargs...)

  MyTag = typeof(prob.f)
  pdual = seed_duals(p,MyTag)
  u0dual = convert.(eltype(pdual),u0)
  if convert_tspan(sensealg)
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
  u,forward_sensitivity_backpass
end

function DiffEqBase._concrete_solve_adjoint(prob,alg,sensealg::ZygoteAdjoint,
                                 u0,p,args...;kwargs...)
    Zygote.pullback((u0,p)->_concrete_solve(prob,alg,u0,p,args...;kwargs...),u0,p)
end

function DiffEqBase._concrete_solve_adjoint(prob,alg,sensealg::TrackerAdjoint,
                                            u0,p,args...;kwargs...)

  t = eltype(prob.tspan)[]
  u = typeof(u0)[]
  function tracker_adjoint_forwardpass(u0,p)
    if DiffEqBase.isinplace(prob)
      # use Array{TrackedReal} for mutation to work
      # Recurse to all Array{TrackedArray}
      _prob = remake(prob,u0=map(identity,u0),p=p)
    else
      # use TrackedArray for efficiency of the tape
      _prob = remake(prob,u0=u0,p=p)
    end
    sol = solve(_prob,alg,args...;kwargs...)
    t = sol.t
    if DiffEqBase.isinplace(prob)
      u = map.(Tracker.data,sol.u)
    else
      u = map(Tracker.data,sol.u)
    end
    adapt(typeof(u0),sol)
  end

  sol,pullback = Tracker.forward(tracker_adjoint_forwardpass,u0,p)
  function tracker_adjoint_backpass(ybar)
    u0bar, pbar = pullback(ybar)
    _u0bar = u0bar isa Tracker.TrackedArray ? Tracker.data(u0bar) : Tracker.data.(u0bar)
    (nothing,nothing,_u0bar,Tracker.data(pbar),ntuple(_->nothing, length(args))...)
  end
  DiffEqArray(u,t),tracker_adjoint_backpass
end
