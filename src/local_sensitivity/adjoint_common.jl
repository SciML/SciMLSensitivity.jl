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

  y = copy(sol.u[end])

  if sensealg.autojacvec isa ReverseDiffVJP
    if DiffEqBase.isinplace(prob)
      tape = ReverseDiff.GradientTape((y, prob.p)) do u,p
        du1 = similar(p, size(u))
        f(du1,u,p,tspan[2])
        return vec(du1)
      end
    else
      tape = ReverseDiff.GradientTape((y, prob.p)) do u,p
        vec(f(u,p,tspan[2]))
      end
    end

    if compile_tape(sensealg.autojacvec)
      paramjac_config = ReverseDiff.compile(tape)
    else
      paramjac_config = tape
    end

    pf = nothing
  elseif DiffEqBase.has_paramjac(f) || isautojacvec || quad
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

using NLsolve: nlsolve
## handle UniformScaling
#issingular(F) = det(F) <= 1e-12 # not the best... but..., meh for now
#issingular(::UniformScaling) = false
function generate_callbacks(sensefun, g, λ, t, callback, init_cb)
  # a trick to make other algorithms still work
  odefun, sensefun = sensefun isa ODEFunction ? (sensefun, sensefun.f) : (nothing, sensefun)
  sensefun.discrete || return callback

  if odefun !== nothing
    mass_matrix = odefun.mass_matrix
    if mass_matrix isa UniformScaling
      factorized_mass_matrix = mass_matrix
    else
      diffvar_idxs = 1:findlast(x->any(!iszero, @view(mass_matrix[:, x])), axes(mass_matrix, 2))
      algevar_idxs = diffvar_idxs[end]+1:size(mass_matrix, 1)
      M̃ = @view mass_matrix[diffvar_idxs, diffvar_idxs]
      factorized_mass_matrix = lu(M̃, check=false)
    end
  else
    factorized_mass_matrix = nothing
  end

  @unpack sensealg, y = sensefun
  prob = getprob(sensefun)
  cur_time = Ref(length(t))
  time_choice = let cur_time=cur_time, t=t
    integrator -> cur_time[] > 0 ? t[cur_time[]] : nothing
  end
  affect! = let isq = (sensealg isa QuadratureAdjoint), λ=λ, t=t, y=y, cur_time=cur_time, idx=length(prob.u0), F=factorized_mass_matrix
    function (integrator)
      p, u = integrator.p, integrator.u
      λ  = isq ? λ : @view(λ[1:idx])
      gᵤ = similar(λ)
      g(gᵤ,y,p,t[cur_time[]],cur_time[])
      if isq
        # TODO: doesn't lead to stable solution
        if !isempty(algevar_idxs)
          origin_f = integrator.f.f.sol.prob.f
          uf = DiffEqBase.UJacobianWrapper(origin_f,integrator.t,integrator.p)
          J = ForwardDiff.jacobian(uf, u)
          dfdd = J[diffvar_idxs, diffvar_idxs]
          dhdd = J[algevar_idxs, diffvar_idxs]
          dfda = J[diffvar_idxs, algevar_idxs]
          dhda = J[algevar_idxs, algevar_idxs]
          Δλa = -dhda'\gᵤ[algevar_idxs]
          Δλd = dhdd'Δλa + gᵤ[diffvar_idxs]
          #u[algevar_idxs] .+= Δλa
          @show J
        else
          Δλd = gᵤ
        end
        if factorized_mass_matrix !== nothing
          F !== I && ldiv!(F, Δλd)
        end
        u[diffvar_idxs] .+= Δλd
      else
        u = @view u[1:idx]
        u .= gᵤ .+ @view integrator.u[1:idx]
      end
      u_modified!(integrator,true)
      cur_time[] -= 1
      return nothing
    end
  end
  cb = IterativeCallback(time_choice,affect!,eltype(prob.tspan);initial_affect=init_cb)
  return CallbackSet(cb,callback)
end
