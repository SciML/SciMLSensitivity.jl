struct AdjointDiffCache{UF,PF,G,TJ,PJT,uType,JC,GC,PJC,rateType,DG}
  uf::UF
  pf::PF
  g::G
  J::TJ
  pJ::PJT
  dg_val::uType
  jac_config::JC
  g_grad_config::GC
  paramjac_config::PJC
  f_cache::rateType
  dg::DG
end

"""
    adjointdiffcache(g,sensealg,discrete,sol,dg)

return (AdjointDiffCache, y)
"""
function adjointdiffcache(g,sensealg,discrete,sol,dg;quad=false)
  prob = sol.prob
  @unpack f, u0, p, tspan = prob
  numparams = length(p)
  numindvar = length(u0)
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
    uf = DiffEqBase.UJacobianWrapper(f,tspan[2],p)
    jac_config = build_jac_config(sensealg,uf,u0)
  end

  y = DiffEqBase.dualcache(copy(sol.u[end]))

  if DiffEqBase.has_paramjac(f) || isautojacvec || quad
    paramjac_config = nothing
    pf = nothing
  else
    pf = DiffEqBase.ParamJacobianWrapper(f,tspan[1],y)
    paramjac_config = build_param_jac_config(sensealg,pf,y,p)
  end

  pJ = (quad || isautojacvec) ? nothing : similar(u0, numindvar, numparams)

  dg_val = similar(u0, numindvar) # number of funcs size
  f_cache = deepcopy(u0)

  return (AdjointDiffCache(uf,pf,pg,J,pJ,dg_val,
                          jac_config,pg_config,paramjac_config,
                          f_cache,dg), y)
end

getprob(S::SensitivityFunction) = S isa ODEBacksolveSensitivityFunction ? S.prob : S.sol.prob

function generate_callbacks(sensefun, g, λ, t, callback, init_cb)
  sensefun.discrete || return callback

  @unpack sensealg, y = sensefun
  prob = getprob(sensefun)
  cur_time = Ref(length(t))
  time_choice = let cur_time=cur_time, t=t
    integrator -> cur_time[] > 0 ? t[cur_time[]] : nothing
  end
  affect! = let isq = (sensealg isa QuadratureAdjoint), λ=λ, t=t, y=y, cur_time=cur_time, idx=length(prob.u0)
    function (integrator)
      p, u = integrator.p, integrator.u
      λ  = isq ? λ : @view(λ[1:idx])
      g(λ,y,p,t[cur_time[]],cur_time[])
      if isq
        u .+= integrator.f.mass_matrix \ λ
      else
        u = @view u[1:idx]
        u .= λ .+ @view integrator.u[1:idx]
      end
      u_modified!(integrator,true)
      cur_time[] -= 1
      return nothing
    end
  end
  cb = IterativeCallback(time_choice,affect!,eltype(prob.tspan);initial_affect=init_cb)
  return CallbackSet(cb,callback)
end
