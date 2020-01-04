struct ODEInterpolatingAdjointSensitivityFunction{rateType,uType,uType2,UF,PF,G,JC,GC,DG,TJ,PJT,PJC,SType,CPS,CV,Alg<:InterpolatingAdjoint} <: SensitivityFunction
  uf::UF
  pf::PF
  g::G
  J::TJ
  pJ::PJT
  dg_val::uType
  jac_config::JC
  g_grad_config::GC
  paramjac_config::PJC
  sensealg::Alg
  f_cache::rateType
  discrete::Bool
  y::uType2
  sol::SType
  dg::DG
  checkpoint_sol::CPS
  colorvec::CV
end

mutable struct CheckpointSolution{S,I,T}
  cpsol::S # solution in a checkpoint interval
  intervals::I # checkpoint intervals
  cursor::Int # sol.prob.tspan = intervals[cursor]
  tols::T
end

@noinline function ODEInterpolatingAdjointSensitivityFunction(g,u0,p,sensealg,discrete,sol,dg,checkpoints,prob,colorvec,tols)
  numparams = p isa Zygote.Params ? sum(length.(p)) : length(p)
  numindvar = length(u0)
  @unpack f, tspan = prob
  checkpointing = sensealg.checkpointing isa Bool ? sensealg.checkpointing : !sol.dense
  (checkpointing && checkpoints === nothing) && error("checkpoints must be passed when checkpointing is enabled.")
  isautojacvec = DiffEqBase.has_jac(f) ? false : get_jacvec(sensealg)
  J = isautojacvec ? nothing : similar(u0, numindvar, numindvar)

  if !discrete
    if dg != nothing
      pg = nothing
      pg_config = nothing
    else
      pg = UGradientWrapper(g,tspan[1],p)
      pg_config = build_grad_config(sensealg,pg,u0,p)
    end
  else
    pg = nothing
    pg_config = nothing
  end

  if DiffEqBase.has_jac(f) || isautojacvec
    jac_config = nothing
    uf = nothing
  else
    uf = DiffEqDiffTools.UJacobianWrapper(f,tspan[1],p)
    jac_config = build_jac_config(sensealg,uf,u0)
  end

  y = copy(sol.u[end])

  if DiffEqBase.has_paramjac(f) || isautojacvec
    paramjac_config = nothing
    pf = nothing
  else
    pf = DiffEqDiffTools.ParamJacobianWrapper(f,tspan[1],y)
    paramjac_config = build_param_jac_config(sensealg,pf,y,p)
  end

  pJ = isautojacvec ? nothing : similar(prob.u0, numindvar, numparams)

  checkpoint_sol = if checkpointing
    intervals = map(tuple, @view(checkpoints[1:end-1]), @view(checkpoints[2:end]))
    interval_end = intervals[end][end]
    prob.tspan[1] > interval_end && push!(intervals, (interval_end, prob.tspan[1]))
    cursor = lastindex(intervals)
    interval = intervals[cursor]
    cpsol = solve(remake(prob, tspan=interval, u0=sol(interval[1])), sol.alg; tols...)
    CheckpointSolution(cpsol, intervals, cursor, tols)
  else
    nothing
  end

  dg_val = similar(u0, numindvar) # number of funcs size
  f_cache = deepcopy(u0)

  return ODEInterpolatingAdjointSensitivityFunction(uf,pf,pg,J,pJ,dg_val,
                               jac_config,pg_config,paramjac_config,
                               sensealg,f_cache,
                               discrete,y,sol,dg,
                               checkpoint_sol,colorvec)
end

function findcursor(checkpoint_sol, t)
  # equivalent with `findfirst(x->x[1] <= t <= x[2], checkpoint_sol.intervals)`
  intervals = checkpoint_sol.intervals
  lt(x, t) = <(x[2], t)
  return searchsortedfirst(intervals, t, lt=lt)
end

# u = λ'
# add tstop on all the checkpoints
function (S::ODEInterpolatingAdjointSensitivityFunction)(du,u,p,t)
  @unpack y, sol, checkpoint_sol, discrete = S
  idx = length(y)
  f = sol.prob.f

  if checkpoint_sol === nothing
    sol(y,t)
  else
    intervals = checkpoint_sol.intervals
    interval = intervals[checkpoint_sol.cursor]
    if !(interval[1] <= t <= interval[2])
      cursor′ = findcursor(checkpoint_sol, t)
      interval = intervals[cursor′]
      cpsol_t = checkpoint_sol.cpsol.t
      sol(y, interval[1])
      prob′ = remake(sol.prob, tspan=intervals[cursor′], u0=y)
      cpsol′ = solve(prob′, sol.alg; dt=abs(cpsol_t[end] - cpsol_t[end-1]), checkpoint_sol.tols...)
      checkpoint_sol.cpsol = cpsol′
      checkpoint_sol.cursor = cursor′
    end
    checkpoint_sol.cpsol(y, t)
  end

  λ     = @view u[1:idx]
  dλ    = @view du[1:idx]
  grad  = @view u[idx+1:end]
  dgrad = @view du[idx+1:end]

  vecjacobian!(dλ, λ, p, t, S, dgrad=dgrad)

  dλ .*= -one(eltype(λ))

  discrete || accumulate_dgdu!(dλ, y, p, t, S)
  return nothing
end

# g is either g(t,u,p) or discrete g(t,u,i)
@noinline function ODEAdjointProblem(sol,sensealg::InterpolatingAdjoint,
                                     g,t=nothing,dg=nothing;
                                     checkpoints=nothing,
                                     callback=CallbackSet(),
                                     reltol=nothing, abstol=nothing,
                                     kwargs...)
  prob = remake(sol.prob, tspan=reverse(sol.prob.tspan))
  f = prob.f
  discrete = t != nothing

  p = prob.p
  p === DiffEqBase.NullParameters() && error("Your model does not have parameters, and thus it is impossible to calculate the derivative of the solution with respect to the parameters. Your model must have parameters to use parameter sensitivity calculations!")
  p isa Zygote.Params && sensealg.autojacvec == false && error("Use of Zygote.Params requires autojacvec=true")
  numparams = p isa Zygote.Params ? sum(length.(p)) : length(p)

  u0 = zero(prob.u0)

  len = length(u0)+numparams
  λ = similar(u0, len)
  sense = ODEInterpolatingAdjointSensitivityFunction(g,u0,
                                                     p,sensealg,discrete,
                                                     sol,dg,checkpoints,prob,
                                                     f.colorvec,
                                                     (reltol=reltol,abstol=abstol))

  init_cb = t !== nothing && prob.tspan[1] == t[end]
  cb = generate_callbacks(sense, g, λ, t, callback, init_cb)
  z0 = vec(zero(λ))
  ODEProblem(sense,z0,prob.tspan,p,callback=cb)
end
