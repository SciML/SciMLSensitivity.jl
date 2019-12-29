struct ODEInterpolatingAdjointSensitivityFunction{rateType,uType,uType2,UF,PF,G,JC,GC,DG,TJ,PJT,PJC,CP,SType,INT,CV} <: SensitivityFunction
  uf::UF
  pf::PF
  g::G
  J::TJ
  pJ::PJT
  dg_val::uType
  jac_config::JC
  g_grad_config::GC
  paramjac_config::PJC
  sensealg::InterpolatingAdjoint
  f_cache::rateType
  discrete::Bool
  y::uType2
  sol::SType
  dg::DG
  checkpointing::Bool
  checkpoints::CP
  integrator::INT
  colorvec::CV
end

@noinline function ODEInterpolatingAdjointSensitivityFunction(g,u0,p,sensealg,discrete,sol,dg,checkpoints,tspan,colorvec)
  numparams = p isa Zygote.Params ? sum(length.(p)) : length(p)
  numindvar = length(u0)
  # if there is an analytical Jacobian provided, we are not going to do automatic `jac*vec`
  f = sol.prob.f
  checkpointing = sensealg.checkpointing isa Bool ? sensealg.checkpointing : !sol.dense
  isautojacvec = DiffEqBase.has_jac(f) ? false : get_jacvec(sensealg)
  J = isautojacvec ? nothing : similar(u0, numindvar, numindvar)

  if !discrete
    if dg != nothing || isautojacvec
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

  y = copy(sol(tspan[1])) # TODO: Has to start at interpolation value!

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
                               discrete,y,sol,dg,checkpointing,checkpoints,integrator,colorvec)
end

# u = λ'
function (S::ODEInterpolatingAdjointSensitivityFunction)(du,u,p,t)
  @unpack y, sol, J, uf, sensealg, checkpointing, f_cache, jac_config, discrete, dg, dg_val, g, g_grad_config = S
  idx = length(y)
  f = sol.prob.f
  isautojacvec = DiffEqBase.has_jac(f) ? false : get_jacvec(sensealg)

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
      # `integrator.u` is aliased to `y`
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

  if !isautojacvec
    if DiffEqBase.has_jac(f)
      f.jac(J,y,p,t) # Calculate the Jacobian into J
    else
      uf.t = t
      jacobian!(J, uf, y, f_cache, sensealg, jac_config)
    end
    mul!(dλ',λ',J)
  else
    if DiffEqBase.isinplace(sol.prob)
      _dy, back = Tracker.forward(y, sol.prob.p) do u, p
        out_ = map(zero, u)
        f(out_, u, p, t)
        Tracker.collect(out_)
      end
      dλ[:], dgrad[:] = Tracker.data.(back(λ))
    elseif !(sol.prob.p isa Zygote.Params)
      _dy, back = Zygote.pullback(y, sol.prob.p) do u, p
        vec(f(u, p, t))
      end
      tmp1,tmp2 = back(λ)
      dλ[:] .= tmp1
      dgrad[:] .= tmp2
    else # Not in-place and p is a Params

      # This is the hackiest hack of the west specifically to get Zygote
      # Implicit parameters to work. This should go away ASAP!

      _dy, back = Zygote.pullback(y, S.sol.prob.p) do u, p
        vec(f(u, p, t))
      end

      _idy, iback = Zygote.pullback(S.sol.prob.p) do
        vec(f(y, p, t))
      end

      igs = iback(λ)
      vs = zeros(Float32, sum(length.(S.sol.prob.p)) - length(y))
      i = 1
      for p in S.sol.prob.p
        g = igs[p]
        g isa AbstractArray || continue
        vs[i:i+length(g)-1] = g
        i += length(g)
      end
      eback = back(λ)
      dλ[:] = eback[1]
      dgrad[:] = vcat(eback[1], vs)
    end
  end

  dλ .*= -one(eltype(λ))

  if !discrete
    if dg != nothing
      dg(dg_val,y,p,t)
    else
      g.t = t
      gradient!(dg_val, g, y, sensealg, g_grad_config)
    end
    dλ .+= dg_val
  end

  if !isautojacvec
    @unpack pJ, pf, paramjac_config = S
    if DiffEqBase.has_paramjac(f)
      f.paramjac(pJ,y,sol.prob.p,t) # Calculate the parameter Jacobian into pJ
    else
      jacobian!(pJ, pf, sol.prob.p, f_cache, sensealg, paramjac_config)
    end
    mul!(dgrad',λ',pJ)
  end
  nothing
end

# g is either g(t,u,p) or discrete g(t,u,i)
@noinline function ODEAdjointProblem(sol,sensealg::InterpolatingAdjoint,
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
  sense = ODEInterpolatingAdjointSensitivityFunction(g,u0,
                                                     p,sensealg,discrete,
                                                     sol,dg,checkpoints,tspan,
                                                     f.colorvec)

  init_cb = t !== nothing && sol.prob.tspan[2] == t[end]
  cb = generate_callbacks(sense, g, λ, t, callback, init_cb)
  z0 = vec(zero(λ))
  ODEProblem(sense,z0,tspan,p,callback=cb)
end
