struct ODEBacksolveSensitivityFunction{rateType,uType,uType2,UF,PF,G,JC,GC,DG,TJ,PJT,PJC,CP,SType,CV,Alg<:BacksolveAdjoint} <: SensitivityFunction
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
  checkpoints::CP
  colorvec::CV
end

@noinline function ODEBacksolveSensitivityFunction(g,u0,p,sensealg,discrete,sol,dg,checkpoints,tspan,colorvec)
  numparams = p isa Zygote.Params ? sum(length.(p)) : length(p)
  numindvar = length(u0)
  f = sol.prob.f
  isautojacvec = get_jacvec(sensealg)
  J = isautojacvec ? nothing : similar(u0, numindvar, numindvar)

  if !discrete
    if dg != nothing
      pg = nothing
      pg_config = nothing
    else
      pg = UGradientWrapper(g,tspan[2],p)
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
    uf = DiffEqDiffTools.UJacobianWrapper(f,tspan[2],p)
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

  pJ = isautojacvec ? nothing : similar(sol.prob.u0, numindvar, numparams)

  dg_val = similar(u0, numindvar) # number of funcs size
  f_cache = deepcopy(u0)

  return ODEBacksolveSensitivityFunction(uf,pf,pg,J,pJ,dg_val,
                               jac_config,pg_config,paramjac_config,
                               sensealg,f_cache,
                               discrete,y,sol,dg,checkpoints,colorvec)
end

# u = λ'
function (S::ODEBacksolveSensitivityFunction)(du,u,p,t)
  @unpack y, sol, discrete = S
  idx = length(y)
  f = sol.prob.f

  λ     = @view u[1:idx]
  dλ    = @view du[1:idx]
  grad  = @view u[idx+1:end-idx]
  dgrad = @view du[idx+1:end-idx]
  _y    = @view u[end-idx+1:end]
  dy    = @view du[end-idx+1:end]
  copyto!(vec(y), _y)

  vecjacobian!(dλ, λ, p, t, S, dgrad=dgrad, dy=dy)

  dλ .*= -one(eltype(λ))

  discrete || accumulate_dgdu!(dλ, y, p, t, S)
  return nothing
end

# g is either g(t,u,p) or discrete g(t,u,i)
@noinline function ODEAdjointProblem(sol,sensealg::BacksolveAdjoint,
                                     g,t=nothing,dg=nothing;
                                     checkpoints=sol.t,
                                     callback=CallbackSet())
  f = sol.prob.f
  tspan = (sol.prob.tspan[2],sol.prob.tspan[1])
  discrete = t != nothing

  p = sol.prob.p
  p === DiffEqBase.NullParameters() && error("Your model does not have parameters, and thus it is impossible to calculate the derivative of the solution with respect to the parameters. Your model must have parameters to use parameter sensitivity calculations!")
  p isa Zygote.Params && sensealg.autojacvec == false && error("Use of Zygote.Params requires autojacvec=true")
  numparams = p isa Zygote.Params ? sum(length.(p)) : length(p)

  u0 = zero(sol.prob.u0)
  len = length(u0)+numparams
  λ = similar(u0, len)
  sense = ODEBacksolveSensitivityFunction(g,u0,
                                        p,sensealg,discrete,
                                        sol,dg,checkpoints,tspan,f.colorvec)

  init_cb = t !== nothing && sol.prob.tspan[2] == t[end]
  cb = generate_callbacks(sense, g, λ, t, callback, init_cb)
  z0 = [vec(zero(λ)); vec(sense.y)]
  ODEProblem(sense,z0,tspan,p,callback=cb)
end
