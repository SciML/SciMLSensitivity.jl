using Flux.Tracker: gradient

struct ODEAdjointSensitivityFunction{rateType,uType,uType2,UF,PF,G,JC,GC,A,DG,TJ,PJT,PJC,CP,SType,INT,CV} <: SensitivityFunction
  uf::UF
  pf::PF
  g::G
  J::TJ
  pJ::PJT
  dg_val::uType
  jac_config::JC
  g_grad_config::GC
  paramjac_config::PJC
  alg::A
  f_cache::rateType
  discrete::Bool
  y::uType2
  sol::SType
  dg::DG
  checkpoints::CP
  integrator::INT
  colorvec::CV
end

@noinline function ODEAdjointSensitivityFunction(g,u0,p,alg,discrete,sol,dg,checkpoints,tspan,colorvec)
  numparams = length(p)
  numindvar = length(u0)
  # if there is an analytical Jacobian provided, we are not going to do automatic `jac*vec`
  f = sol.prob.f
  isautojacvec = DiffEqBase.has_jac(f) ? false : get_jacvec(alg)
  J = isautojacvec ? nothing : similar(u0, numindvar, numindvar)

  if !discrete
    if dg != nothing || isautojacvec
      pg = nothing
      pg_config = nothing
    else
      pg = UGradientWrapper(g,tspan[2],p)
      pg_config = build_grad_config(alg,pg,u0,p)
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
    jac_config = build_jac_config(alg,uf,u0)
  end

  y = copy(sol(tspan[1])) # TODO: Has to start at interpolation value!
  paramjac_config = nothing
  pf = nothing
  if !isquad(alg)
    if DiffEqBase.has_paramjac(f) || isautojacvec
      paramjac_config = nothing
    else
      pf = DiffEqDiffTools.ParamJacobianWrapper(f,tspan[1],y)
      paramjac_config = build_param_jac_config(alg,pf,y,p)
    end
  end

  pJ = if !isquad(alg)
    isautojacvec ? nothing : similar(sol.prob.u0, numindvar, numparams)
  else
    nothing
  end
  integrator = if ischeckpointing(alg)
    integ = init(sol.prob, sol.alg, save_on=false)
    integ.u = y
    integ
  else
    nothing
  end
  dg_val = similar(u0, numindvar) # number of funcs size
  f_cache = deepcopy(u0)

  return ODEAdjointSensitivityFunction(uf,pf,pg,J,pJ,dg_val,
                               jac_config,pg_config,paramjac_config,
                               alg,f_cache,
                               discrete,y,sol,dg,checkpoints,integrator,colorvec)
end

# u = λ'
function (S::ODEAdjointSensitivityFunction)(du,u,p,t)
  @unpack y, sol, J, uf, alg, f_cache, jac_config, discrete, dg, dg_val, g, g_grad_config = S
  idx = length(y)
  f = sol.prob.f
  isautojacvec = DiffEqBase.has_jac(f) ? false : get_jacvec(alg)

  if isbcksol(alg)
    λ     = @view u[1:idx]
    dλ    = @view du[1:idx]
    grad  = @view u[idx+1:end-idx]
    dgrad = @view du[idx+1:end-idx]
    _y    = @view u[end-idx+1:end]
    dy    = @view du[end-idx+1:end]
    copyto!(vec(y), _y)
    isautojacvec || f(dy, _y, p, t)
  else
    if ischeckpointing(alg)
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
        # `integrator.u` is aliased to `y`
      else
        sol(y,t)
      end
    else
      sol(y,t)
    end
    if isquad(alg)
      λ     = u
      dλ    = du
    else
      λ     = @view u[1:idx]
      dλ    = @view du[1:idx]
      grad  = @view u[idx+1:end]
      dgrad = @view du[idx+1:end]
    end
  end

  if !isautojacvec
    if DiffEqBase.has_jac(f)
      f.jac(J,y,p,t) # Calculate the Jacobian into J
    else
      uf.t = t
      jacobian!(J, uf, y, f_cache, alg, jac_config)
    end
    mul!(dλ',λ',J)
  elseif isquad(alg)
    _dy, back = Tracker.forward(y) do u
      if DiffEqBase.isinplace(sol.prob)
        out_ = map(zero, u)
        f(out_, u, p, t)
        Tracker.collect(out_)
      else
        vec(f(u, p, t))
      end
    end
    dλ[:] = Tracker.data(back(λ)[1])
    isbcksol(alg) && (dy[:] = vec(Tracker.data(_dy)))
  else
    _dy, back = Tracker.forward(y, sol.prob.p) do u, p
      if DiffEqBase.isinplace(sol.prob)
        out_ = map(zero, u)
        f(out_, u, p, t)
        Tracker.collect(out_)
      else
        vec(f(u, p, t))
      end
    end
    dλ[:], dgrad[:] = Tracker.data.(back(λ))
    isbcksol(alg) && (dy[:] = vec(Tracker.data(_dy)))
  end

  dλ .*= -one(eltype(λ))

  if !discrete
    if dg != nothing
      dg(dg_val,y,p,t)
    else
      g.t = t
      gradient!(dg_val, g, y, alg, g_grad_config)
    end
    dλ .+= dg_val
  end

  if !isquad(alg) && !isautojacvec
    @unpack pJ, pf, paramjac_config = S
    if DiffEqBase.has_paramjac(f)
      f.paramjac(pJ,y,sol.prob.p,t) # Calculate the parameter Jacobian into pJ
    else
      jacobian!(pJ, pf, sol.prob.p, f_cache, alg, paramjac_config)
    end
    mul!(dgrad',λ',pJ)
  end
  nothing
end

# g is either g(t,u,p) or discrete g(t,u,i)
@noinline function ODEAdjointProblem(sol,g,t=nothing,dg=nothing,
                           alg=SensitivityAlg();
                           checkpoints=sol.t,
                           callback=CallbackSet())
  f = sol.prob.f
  tspan = (sol.prob.tspan[2],sol.prob.tspan[1])
  discrete = t != nothing

  p = sol.prob.p
  p === nothing && error("You must have parameters to use parameter sensitivity calculations!")

  u0 = zero(sol.prob.u0)

  len = isquad(alg) ? length(u0) : length(u0)+length(p)
  λ = similar(u0, len)
  sense = ODEAdjointSensitivityFunction(g,u0,
                                        p,alg,discrete,
                                        sol,dg,checkpoints,tspan,f.colorvec)

  cb = generate_callbacks(sense, g, λ, t, callback)
  z0 = isbcksol(alg) ? [vec(zero(λ)); vec(sense.y)] : vec(zero(λ))
  ODEProblem(sense,z0,tspan,p,callback=cb)
end

function generate_callbacks(sensefun, g, λ, t, callback)
  if sensefun.discrete
    @unpack alg, y, sol = sensefun
    prob = sol.prob
    cur_time = Ref(length(t))
    function time_choice(integrator)
      cur_time[] > 0 ? t[cur_time[]] : nothing
    end
    affect! = let isq = isquad(alg), λ=λ, t=t, y=y, cur_time=cur_time, idx=length(prob.u0)
      function (integrator)
        p, u = integrator.p, integrator.u
        λ  = isq ? λ : @view(λ[1:idx])
        g(λ,y,p,t[cur_time[]],cur_time[])
        if isq
          u .+= λ
        else
          u = @view u[1:idx]
          u .= λ .+ @view integrator.u[1:idx]
        end
        u_modified!(integrator,true)
        cur_time[] -= 1
      end
    end
    cb = IterativeCallback(time_choice,affect!,eltype(prob.tspan);initial_affect=true)

    _cb = CallbackSet(cb,callback)
  else
    _cb = callback
  end
  return _cb
end

struct AdjointSensitivityIntegrand{pType,uType,rateType,S,AS,PF,PJC,A,PJT}
  sol::S
  adj_sol::AS
  p::pType
  y::uType
  λ::uType
  pf::PF
  f_cache::rateType
  pJ::PJT
  paramjac_config::PJC
  alg::A
end

function AdjointSensitivityIntegrand(sol,adj_sol,alg=SensitivityAlg())
  prob = sol.prob
  @unpack f, p, tspan, u0 = prob
  y = similar(sol.prob.u0)
  λ = similar(adj_sol.prob.u0)
  # we need to alias `y`
  pf = DiffEqDiffTools.ParamJacobianWrapper(f,tspan[1],y)
  f_cache = similar(y)
  isautojacvec = DiffEqBase.has_paramjac(f) ? false : get_jacvec(alg)
  pJ = isautojacvec ? nothing : similar(u0,length(u0),length(p))

  if DiffEqBase.has_paramjac(f) || isautojacvec
    paramjac_config = nothing
  else
    paramjac_config = build_param_jac_config(alg,pf,y,p)
  end
  AdjointSensitivityIntegrand(sol,adj_sol,p,y,λ,pf,f_cache,pJ,paramjac_config,alg)
end

function (S::AdjointSensitivityIntegrand)(out,t)
  @unpack y, λ, pJ, pf, p, f_cache, paramjac_config, alg, sol, adj_sol = S
  f = sol.prob.f
  sol(y,t)
  adj_sol(λ,t)
  λ .*= -one(eltype(λ))
  isautojacvec = DiffEqBase.has_paramjac(f) ? false : get_jacvec(alg)
  # y is aliased
  pf.t = t

  if !isautojacvec
    if DiffEqBase.has_paramjac(f)
      f.paramjac(pJ,y,p,t) # Calculate the parameter Jacobian into pJ
    else
      jacobian!(pJ, pf, p, f_cache, alg, paramjac_config)
    end
    mul!(out',λ',pJ)
  else
    _, back = Tracker.forward(y, p) do u, p
      if DiffEqBase.isinplace(sol.prob)
        out_ = map(zero, u)
        f(out_, u, p, t)
        Tracker.collect(out_)
      else
        vec(f(u, p, t))
      end
    end
    out[:] = vec(Tracker.data(back(λ)[2]))
  end
  out'
end

function (S::AdjointSensitivityIntegrand)(t)
  out = similar(S.p)
  S(out,t)
end

function adjoint_sensitivities_u0(sol,alg,g,t=nothing,dg=nothing;
                                  abstol=1e-6,reltol=1e-3,
                                  iabstol=abstol, ireltol=reltol,sensealg=SensitivityAlg(checkpointing=!sol.dense,quad=false),
                                  checkpoints=sol.t,
                                  kwargs...)
  isquad(sensealg) && error("Can't get sensitivities of u0 with quadrature.")
  adj_prob = ODEAdjointProblem(sol,g,t,dg,sensealg,checkpoints=checkpoints)
  adj_sol = solve(adj_prob,alg;abstol=abstol,reltol=reltol,kwargs...,save_everystep=false,save_start=false,saveat=eltype(sol[1])[])

  -adj_sol[end][1:length(sol.prob.u0)],
    adj_sol[end][(1:length(sol.prob.p)) .+ length(sol.prob.u0)]'
end

function adjoint_sensitivities(sol,alg,g,t=nothing,dg=nothing;
                               abstol=1e-6,reltol=1e-3,
                               iabstol=abstol, ireltol=reltol,sensealg=SensitivityAlg(checkpointing=!sol.dense),
                               checkpoints=sol.t,
                               kwargs...)
  adj_prob = ODEAdjointProblem(sol,g,t,dg,sensealg,checkpoints=checkpoints)
  isq = isquad(sensealg)
  adj_sol = solve(adj_prob,alg;abstol=abstol,reltol=reltol,
                               save_everystep=isq,save_start=isq,kwargs...)
  !isq && return adj_sol[end][(1:length(sol.prob.p)) .+ length(sol.prob.u0)]'
  integrand = AdjointSensitivityIntegrand(sol,adj_sol,sensealg)

  if t === nothing
    res,err = quadgk(integrand,sol.prob.tspan[1],sol.prob.tspan[2],
                   atol=iabstol,rtol=ireltol)
  else
    res = zero(integrand.p)'
    for i in 1:length(t)-1
      res .+= quadgk(integrand,t[i],t[i+1],
                     atol=iabstol,rtol=ireltol)[1]
    end
    if t[1] != sol.prob.tspan[1]
      res .+= quadgk(integrand,sol.prob.tspan[1],t[1],
                     atol=iabstol,rtol=ireltol)[1]
    end
  end
  res
end
