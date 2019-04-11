using Flux.Tracker: gradient

struct ODEAdjointSensitivityFunction{dgType,rateType,uType,F,J,PJ,UF,PF,G,JC,GC,A,DG,MM,TJ,PJT,PJC,CP,SType,INT} <: SensitivityFunction
  f::F
  jac::J
  paramjac::PJ
  uf::UF
  pf::PF
  g::G
  J::TJ
  pJ::PJT
  dg_val::dgType
  jac_config::JC
  g_grad_config::GC
  paramjac_config::PJC
  alg::A
  numparams::Int
  numindvar::Int
  f_cache::rateType
  discrete::Bool
  y::uType
  sol::SType
  dg::DG
  mass_matrix::MM
  checkpoints::CP
  integrator::INT
end

@noinline function ODEAdjointSensitivityFunction(f,jac,paramjac,uf,pf,g,u0,
                                      jac_config,g_grad_config,paramjac_config,
                                      p,f_cache,alg,discrete,y,sol,dg,mm,checkpoints)
  numparams::Int = length(p)
  numindvar::Int = length(u0)
  # if there is an analytical Jacobian provided, we are not going to do automatic `jac*vec`
  isautojacvec = DiffEqBase.has_jac(f) ? false : get_jacvec(alg)
  J = isautojacvec ? nothing : similar(sol.prob.u0, numindvar, numindvar)
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
  return ODEAdjointSensitivityFunction(f,jac,paramjac,uf,pf,g,J,pJ,dg_val,
                               jac_config,g_grad_config,paramjac_config,
                               alg,numparams,numindvar,f_cache,
                               discrete,y,sol,dg,mm,checkpoints,integrator)
end

# u = λ'
function (S::ODEAdjointSensitivityFunction)(du,u,p,t)
  idx = length(S.y)
  y = S.y
  isautojacvec = DiffEqBase.has_jac(S.f) ? false : get_jacvec(S.alg)
  sol = S.sol

  if isbcksol(S.alg)
    λ     = @view u[1:idx]
    dλ    = @view du[1:idx]
    grad  = @view u[idx+1:end-idx]
    dgrad = @view du[idx+1:end-idx]
    _y    = @view u[end-idx+1:end]
    dy    = @view du[end-idx+1:end]
    copyto!(y, _y)
    isautojacvec || sol.prob.f(dy, _y, p, t)
  else
    if ischeckpointing(S.alg)
      # assuming that in the forward direction `t0` < `t1`, and the
      # `checkpoints` vector is sorted with respect to the forward direction
      tidx = findlast(x->x <= t, S.checkpoints)
      t0 = S.checkpoints[tidx]
      dt = t-t0
      integrator = S.integrator
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
    if isquad(S.alg)
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
    if DiffEqBase.has_jac(S.f)
      S.f.jac(S.J,y,p,t) # Calculate the Jacobian into J
    else
      S.uf.t = t
      jacobian!(S.J, S.uf, y, S.f_cache, S.alg, S.jac_config)
    end
    mul!(dλ',λ',S.J)
  elseif isquad(S.alg)
    _dy, back = Tracker.forward(y) do u
      out_ = map(zero, u)
      S.f(out_, u, p, t)
      Tracker.collect(out_)
    end
    dλ[:] = Tracker.data(back(λ)[1])
    isbcksol(S.alg) && (dy[:] = Tracker.data(_dy))
  else
    _dy, back = Tracker.forward(y, S.sol.prob.p) do u, p
      out_ = map(zero, u)
      S.f(out_, u, p, t)
      Tracker.collect(out_)
    end
    dλ[:], dgrad[:] = map(Tracker.data, back(λ))
    isbcksol(S.alg) && (dy[:] = Tracker.data(_dy))
  end

  dλ .*= -one(eltype(λ))

  if !S.discrete
    if S.dg != nothing
      S.dg(S.dg_val,y,p,t)
    else
      S.g.t = t
      gradient!(S.dg_val, S.g, y, S.alg, S.g_grad_config)
    end
    dλ .+= S.dg_val
  end

  if !isquad(S.alg) && !isautojacvec
    if DiffEqBase.has_paramjac(S.f)
      S.f.paramjac(S.pJ,y,S.sol.prob.p,t) # Calculate the parameter Jacobian into pJ
    else
      jacobian!(S.pJ, S.pf, S.sol.prob.p, S.f_cache, S.alg, S.paramjac_config)
    end
    mul!(dgrad',λ',S.pJ)
  end
  nothing
end

# g is either g(t,u,p) or discrete g(t,u,i)
@noinline function ODEAdjointProblem(sol,g,t=nothing,dg=nothing,
                           alg=SensitivityAlg();
                           checkpoints=sol.t,
                           callback=CallbackSet(),mass_matrix=I)
  f = sol.prob.f
  tspan = (sol.prob.tspan[2],sol.prob.tspan[1])
  t != nothing && (tspan = (t[end],t[1]))
  discrete = t != nothing

  isinplace = DiffEqBase.isinplace(sol.prob)
  p = sol.prob.p
  # if there is an analytical Jacobian provided, we are not going to do automatic `jac*vec`
  isautojacvec = DiffEqBase.has_jac(f) ? false : get_jacvec(alg)
  p === nothing && error("You must have parameters to use parameter sensitivity calculations!")

  u0 = zero(sol.prob.u0)

  if DiffEqBase.has_jac(f) || isautojacvec
    jac_config = nothing
    uf = nothing
  else
    jac_config = build_jac_config(alg,uf,u0)
    uf = DiffEqDiffTools.UJacobianWrapper(f,tspan[2],p)
  end

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

  len::Int = isquad(alg) ? length(u0) : length(u0)+length(p)
  λ = similar(u0, len)
  sense = ODEAdjointSensitivityFunction(f,f.jac,f.paramjac,
                                       uf,pf,pg,u0,jac_config,pg_config,paramjac_config,
                                       p,deepcopy(u0),alg,discrete,
                                       y,sol,dg,mass_matrix,checkpoints)

  if discrete
    cur_time = Ref(length(t))
    function time_choice(integrator)
      cur_time[] > 0 ? t[cur_time[]] : nothing
    end
    affect! = let isq = isquad(alg), λ=λ, t=t, y=y, cur_time=cur_time, idx=length(u0)
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
    cb = IterativeCallback(time_choice,affect!,eltype(tspan);initial_affect=true)

    _cb = CallbackSet(cb,callback)
  else
    _cb = callback
  end

  z0 = isbcksol(alg) ? [vec(zero(λ)); vec(y)] : vec(zero(λ))
  ODEProblem(sense,z0,tspan,p,callback=_cb)
end

struct AdjointSensitivityIntegrand{pType,uType,rateType,S,AS,F,PF,PJC,A,PJT}
  sol::S
  adj_sol::AS
  f::F
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
  f = sol.prob.f
  tspan = sol.prob.tspan
  p = sol.prob.p
  # we need to copy here, because later, we will call
  # `ReverseDiff.compile(ReverseDiff.GradientTape(pf′, (y, p)))`
  y = similar(sol.prob.u0)
  λ = similar(adj_sol.prob.u0)
  # we need to alias `y`
  pf = DiffEqDiffTools.ParamJacobianWrapper(f,tspan[1],y)
  f_cache = similar(y)
  isautojacvec = DiffEqBase.has_paramjac(f) ? false : get_jacvec(alg)
  pJ = isautojacvec ? nothing : Matrix{eltype(sol.prob.u0)}(undef,length(sol.prob.u0),length(p))

  if DiffEqBase.has_paramjac(f) || isautojacvec
    paramjac_config = nothing
  else
    paramjac_config = build_param_jac_config(alg,pf,y,p)
  end
  AdjointSensitivityIntegrand(sol,adj_sol,f,p,y,λ,pf,f_cache,pJ,paramjac_config,alg)
end

function (S::AdjointSensitivityIntegrand)(out,t)
  y = S.y
  S.sol(y,t)
  λ = S.λ
  S.adj_sol(λ,t)
  λ .*= -one(eltype(λ))
  isautojacvec = DiffEqBase.has_paramjac(S.f) ? false : get_jacvec(S.alg)
  # y is aliased
  S.pf.t = t

  if !isautojacvec
    if DiffEqBase.has_paramjac(S.f)
      S.f.paramjac(S.pJ,y,S.p,t) # Calculate the parameter Jacobian into pJ
    else
      jacobian!(S.pJ, S.pf, S.p, S.f_cache, S.alg, S.paramjac_config)
    end
    mul!(out',λ',S.pJ)
  else
    _, back = Tracker.forward(y, S.p) do u, p
      out_ = map(zero, u)
      S.f(out_, u, p, t)
      Tracker.collect(out_)
    end
    out[:] = Tracker.data(back(λ)[2])
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
  adj_sol = solve(adj_prob,alg;abstol=abstol,reltol=reltol,save_everystep=false,save_start=false,kwargs...)
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
  end
  res
end
