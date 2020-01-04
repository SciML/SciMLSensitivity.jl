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

mutable struct CheckpointSolution{S,I}
  cpsol::S # solution in a checkpoint interval
  intervals::I # checkpoint intervals
  cursor::Int # sol.prob.tspan = intervals[cursor]
end

@noinline function ODEInterpolatingAdjointSensitivityFunction(g,u0,p,sensealg,discrete,sol,dg,checkpoints,prob,colorvec)
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

  y = copy(sol(tspan[1]))

  if DiffEqBase.has_paramjac(f) || isautojacvec
    paramjac_config = nothing
    pf = nothing
  else
    pf = DiffEqDiffTools.ParamJacobianWrapper(f,tspan[1],y)
    paramjac_config = build_param_jac_config(sensealg,pf,y,p)
  end

  pJ = isautojacvec ? nothing : similar(sol.prob.u0, numindvar, numparams)

  integrator = if checkpointing
    integ = init(sol.prob, sol.alg, save_on=false)
    integ.u = y
    integ
  else
    nothing
  end
  dg_val = similar(u0, numindvar) # number of funcs size
  f_cache = deepcopy(u0)

  return ODEInterpolatingAdjointSensitivityFunction(uf,pf,pg,J,pJ,dg_val,
                               jac_config,pg_config,paramjac_config,
                               sensealg,f_cache,
                               discrete,y,sol,dg,checkpointing,checkpoints,
                               integrator,colorvec)
end

# u = λ'
# add tstop on all the checkpoints
function (S::ODEInterpolatingAdjointSensitivityFunction)(du,u,p,t)
  @unpack y, sol, checkpointing, discrete = S
  idx = length(y)
  f = sol.prob.f

  if checkpointing
    @unpack integrator, checkpoints = S
    # assuming that in the forward direction `t0` < `t1`, and the
    # `checkpoints` vector is sorted with respect to the forward direction
    tidx = findlast(x->x <= t, checkpoints)
    t0 = checkpoints[tidx]
    dt = t-t0
    if abs(dt) > integrator.opts.dtmin
      sol(integrator.u, t0)
      copyto!(integrator.uprev, integrator.u)
      integrator.t = t0
      # set `iter` to some arbitrary integer so that there won't be max maxiters error
      integrator.iter=100
      u_modified!(integrator, true)
      step!(integrator, dt, true)
      if !DiffEqBase.isinplace(sol.prob) # `integrator.u` is aliased to `y`
        y .= integrator.u
      end
    else
      sol(y,t)
    end
  else
    sol(y,t)
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
                                     callback=CallbackSet())
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
                                                     f.colorvec)

  init_cb = t !== nothing && prob.tspan[1] == t[end]
  cb = generate_callbacks(sense, g, λ, t, callback, init_cb)
  z0 = vec(zero(λ))
  ODEProblem(sense,z0,prob.tspan,p,callback=cb)
end
