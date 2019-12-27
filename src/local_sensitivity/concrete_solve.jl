## High level

function concrete_solve(prob::DiffEqBase.DEProblem,alg::DiffEqBase.DEAlgorithm,
                        u0=prob.u0,p=prob.p,args...;kwargs...)
  _concrete_solve(prob,alg,u0,p,args...;kwargs...)
end

function _concrete_solve(prob::DiffEqBase.DEProblem,alg::DiffEqBase.DEAlgorithm,
                        u0=prob.u0,p=prob.p,args...;kwargs...)
  sol = solve(remake(prob,u0=u0,p=p),alg,args...;kwargs...)
  RecursiveArrayTools.DiffEqArray(reduce(hcat,sol.u),sol.t)
end

ZygoteRules.@adjoint function concrete_solve(prob,alg,u0,p,args...;
                                             sensealg=nothing,kwargs...)
  _concrete_solve_adjoint(prob,alg,sensealg,u0,p,args...;kwargs...)
end

_concrete_solve_adjoint(prob,alg,sensealg::Nothing,u0,p,args...;kwargs...) =
  _concrete_solve_adjoint(prob,alg,InterpolatingAdjoint(),u0,p,args...;kwargs...)

function _concrete_solve_adjoint(prob,alg,sensealg::AbstractAdjointSensitivityAlgorithm,
                                 u0,p,args...;save_start=true,save_end=true,kwargs...)
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
  sol = solve(_prob,args...;save_start=true,save_end=true,kwargs...)

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

  function adjoint_sensitivity_backpass(Δ)
    function df(out, u, p, t, i)
      if only_end
        out[:] .= -vec(Δ)
      else
        out[:] .= -reshape(Δ, :, size(Δ)[end])[:, i]
      end
    end

    ts = sol.t[sol_idxs]
    du0, dp = adjoint_sensitivities_u0(sol,args...,df,ts;
                    kwargs_adj...)

    (nothing,nothing,reshape(dp,size(p)), reshape(du0,size(u0)), ntuple(_->nothing, length(args))...)
  end
  RecursiveArrayTools.DiffEqArray(out,sol.t), adjoint_sensitivity_backpass
end

function _concrete_solve_adjoint(prob,alg,sensealg::AbstractForwardSensitivityAlgorithm,
                                 u0,p,args...;kwargs...)
   _prob = ODEForwardSensitivityProblem(prob.f,u0,prob.tspan,p,sensealg)
   sol = solve(_prob,args...;kwargs...)
   u,du = extract_local_sensitivities(sol,Val(true))
   function forward_sensitivity_backpass(Δ)
     (nothing,nothing,Δ'*du,nothing,ntuple(_->nothing, length(args))...)
   end
   DiffEqArray(u,sol.t),forward_sensitivity_backpass
end

function _concrete_solve_adjoint(prob,alg,sensealg::ZygoteAdjoint,
                                 u0,p,args...;kwargs...)
    Zygote.pullback(_concrete_solve,prob,alg,u0,p,args...;kwargs...)
end

function _concrete_solve_adjoint(prob,alg,sensealg::TrackerAdjoint,
                                 u0,p,args...;kwargs...)

  function tracker_adjoint_forwardpass(u0,p)
    if DiffEqBase.isinplace(prob)
      # use Array{TrackedReal} for mutation to work
      # Recurse to all Array{TrackedArray}
      _prob = remake(prob,u0=map(identity,u0),p=p)
    else
      # use TrackedArray for efficiency of the tape
      _prob = remake(prob,u0=u0,p=p)
    end
    solve(_prob,args...;kwargs...)
  end

  sol,pullback = Tracker.forward(tracker_adjoint_forwardpass,u0,p)
  function tracker_adjoint_backpass(ybar)
    u0bar,pbar = pullback(ybar)
    _u0bar = u0bar isa Tracker.TrackedArray ? Tracker.data(u0bar) : Tracker.data.(u0bar)
    (nothing,nothing,_u0bar,Tracker.data(pbar),ntuple(_->nothing, length(args))...)
  end
  DiffEqArray(Tracker.data(sol),sol.t),tracker_adjoint_backpass
end
