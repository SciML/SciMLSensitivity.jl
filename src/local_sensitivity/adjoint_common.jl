struct AdjointDiffCache{UF,PF,G,TJ,PJT,uType,JC,GC,PJC,JNC,PJNC,rateType,DG,DI,AI,FM}
  uf::UF
  pf::PF
  g::G
  J::TJ
  pJ::PJT
  dg_val::uType
  jac_config::JC
  g_grad_config::GC
  paramjac_config::PJC
  jac_noise_config::JNC
  paramjac_noise_config::PJNC
  f_cache::rateType
  dg::DG
  diffvar_idxs::DI
  algevar_idxs::AI
  factorized_mass_matrix::FM
  issemiexplicitdae::Bool
end

"""
    adjointdiffcache(g,sensealg,discrete,sol,dg;quad=false)

return (AdjointDiffCache, y)
"""
function adjointdiffcache(g,sensealg,discrete,sol,dg,f;quad=false,noiseterm=false)
  prob = sol.prob
  if prob isa DiffEqBase.SteadyStateProblem
    @unpack u0, p = prob
    tspan = (nothing, nothing)
  #elseif prob isa SDEProblem
  #  @unpack tspan, u0, p = prob
  else
    @unpack u0, p, tspan = prob
  end
  numparams = p === nothing || p === DiffEqBase.NullParameters() ? 0 : length(p)
  numindvar = length(u0)
  isautojacvec = get_jacvec(sensealg)

  issemiexplicitdae = false
  mass_matrix = sol.prob.f.mass_matrix
  if mass_matrix isa UniformScaling
    factorized_mass_matrix = mass_matrix'
  elseif mass_matrix isa Tuple{UniformScaling,UniformScaling}
    factorized_mass_matrix = (I',I')
  else
    mass_matrix = mass_matrix'
    diffvar_idxs = findall(x->any(!iszero, @view(mass_matrix[:, x])), axes(mass_matrix, 2))
    algevar_idxs = setdiff(eachindex(u0), diffvar_idxs)
    # TODO: operator
    M̃ = @view mass_matrix[diffvar_idxs, diffvar_idxs]
    factorized_mass_matrix = lu(M̃, check=false)
    issuccess(factorized_mass_matrix) || error("The submatrix corresponding to the differential variables of the mass matrix must be nonsingular!")
    isempty(algevar_idxs) || (issemiexplicitdae = true)
  end
  if !issemiexplicitdae
    diffvar_idxs = eachindex(u0)
    algevar_idxs = 1:0
  end

  J = (issemiexplicitdae || !isautojacvec || prob isa DiffEqBase.SteadyStateProblem) ? similar(u0, numindvar, numindvar) : nothing

  if !discrete
    if dg != nothing
      pg = nothing
      pg_config = nothing
      if dg isa Tuple && length(dg) == 2
        dg_val = (similar(u0, numindvar),similar(u0, numparams))
        dg_val[1] .= false
        dg_val[2] .= false
      else
        dg_val = similar(u0, numindvar) # number of funcs size
        dg_val .= false
      end
    else
      pg = UGradientWrapper(g,tspan[2],p)
      pg_config = build_grad_config(sensealg,pg,u0,p)
      dg_val = similar(u0, numindvar) # number of funcs size
      dg_val .= false
    end
  else
    dg_val = nothing
    pg = nothing
    pg_config = nothing
  end

  if DiffEqBase.has_jac(f) || (J === nothing)
    jac_config = nothing
    uf = nothing
  else
    if DiffEqBase.isinplace(prob)
      uf = DiffEqBase.UJacobianWrapper(f,tspan[2],p)
      jac_config = build_jac_config(sensealg,uf,u0)
    else
      uf = DiffEqBase.UDerivativeWrapper(f,tspan[2],p)
      jac_config = nothing
    end
  end

  if prob isa DiffEqBase.SteadyStateProblem
    y = copy(sol.u)
  else
    y = copy(sol.u[end])
  end

  if sensealg.autojacvec isa ReverseDiffVJP ||
    (sensealg.autojacvec isa Bool && sensealg.autojacvec && DiffEqBase.isinplace(prob))

    if prob isa DiffEqBase.SteadyStateProblem
       if DiffEqBase.isinplace(prob)
        tape = ReverseDiff.GradientTape((y, prob.p)) do u,p
          du1 = p !== nothing && p !== DiffEqBase.NullParameters() ? similar(p, size(u)) : similar(u)
          f(du1,u,p,nothing)
          return vec(du1)
        end
      else
        tape = ReverseDiff.GradientTape((y, prob.p)) do u,p
          vec(f(u,p,nothing))
        end
      end
    elseif  noiseterm && (!StochasticDiffEq.is_diagonal_noise(prob) || isnoisemixing(sensealg))
      tape = nothing
    else
      if DiffEqBase.isinplace(prob)
        tape = ReverseDiff.GradientTape((y, prob.p, [tspan[2]])) do u,p,t
            du1 = p !== nothing && p !== DiffEqBase.NullParameters() ? similar(p, size(u)) : similar(u)
            f(du1,u,p,first(t))
          return vec(du1)
       end
       else
         tape = ReverseDiff.GradientTape((y, prob.p, [tspan[2]])) do u,p,t
           vec(f(u,p,first(t)))
         end
       end
    end
    if compile_tape(sensealg.autojacvec)
      paramjac_config = ReverseDiff.compile(tape)
    else
      paramjac_config = tape
    end

    pf = nothing
  elseif (DiffEqBase.has_paramjac(f) || isautojacvec || quad)
    paramjac_config = nothing
    pf = nothing
  else
    if DiffEqBase.isinplace(prob)
      pf = DiffEqBase.ParamJacobianWrapper(f,tspan[1],y)
      paramjac_config = build_param_jac_config(sensealg,pf,y,p)
    else
      pf = ParamGradientWrapper(f,tspan[2],y)
      paramjac_config = nothing
    end
  end

  pJ = (quad || isautojacvec) ? nothing : similar(u0, numindvar, numparams)

  f_cache = DiffEqBase.isinplace(prob) ? deepcopy(u0) : nothing

  if noiseterm
    if (sensealg.noise isa ReverseDiffNoise ||
      (sensealg.noise isa Bool && sensealg.noise && DiffEqBase.isinplace(prob)))

      jac_noise_config = nothing
      paramjac_noise_config = []

      if DiffEqBase.isinplace(prob)
        for i in 1:numindvar
          function noisetape(indx)
            if StochasticDiffEq.is_diagonal_noise(prob)
              ReverseDiff.GradientTape((y, prob.p, [tspan[2]])) do u,p,t
                du1 = similar(p, size(u))
                f(du1,u,p,first(t))
                return du1[indx]
              end
            else
              ReverseDiff.GradientTape((y, prob.p, [tspan[2]])) do u,p,t
                du1 = similar(p, size(prob.noise_rate_prototype))
                f(du1,u,p,first(t))
                return du1[:,indx]
              end
            end
          end
          tapei = noisetape(i)
          if compile_tape(sensealg.noise)
            push!(paramjac_noise_config, ReverseDiff.compile(tapei))
          else
            push!(paramjac_noise_config, tapei)
          end
        end
      else
        for i in 1:numindvar
          function noisetapeoop(indx)
            if StochasticDiffEq.is_diagonal_noise(prob)
              ReverseDiff.GradientTape((y, prob.p, [tspan[2]])) do u,p,t
                f(u,p,first(t))[indx]
              end
            else
              ReverseDiff.GradientTape((y, prob.p, [tspan[2]])) do u,p,t
                f(u,p,first(t))[:,indx]
              end
            end
          end
          tapei = noisetapeoop(i)
          if compile_tape(sensealg.noise)
            push!(paramjac_noise_config, ReverseDiff.compile(tapei))
          else
            push!(paramjac_noise_config, tapei)
          end
        end
      end
    elseif (sensealg.noise isa Bool && !sensealg.noise)
      if DiffEqBase.isinplace(prob)
        if StochasticDiffEq.is_diagonal_noise(prob)
          pf = DiffEqBase.ParamJacobianWrapper(f,tspan[1],y)
          if isnoisemixing(sensealg)
            uf = DiffEqBase.UJacobianWrapper(f,tspan[2],p)
            jac_noise_config = build_jac_config(sensealg,uf,u0)
          else
            jac_noise_config = nothing
          end
        else
          pf = ParamNonDiagNoiseJacobianWrapper(f,tspan[1],y,prob.noise_rate_prototype)
          uf = UNonDiagNoiseJacobianWrapper(f,tspan[2],p,prob.noise_rate_prototype)
          jac_noise_config = build_jac_config(sensealg,uf,u0)
        end
        paramjac_noise_config = build_param_jac_config(sensealg,pf,y,p)
      else
        if StochasticDiffEq.is_diagonal_noise(prob)
          pf = ParamGradientWrapper(f,tspan[2],y)
          if isnoisemixing(sensealg)
            uf = DiffEqBase.UDerivativeWrapper(f,tspan[2],p)
          end
        else
          pf = ParamNonDiagNoiseGradientWrapper(f,tspan[1],y)
          uf = UNonDiagNoiseGradientWrapper(f,tspan[2],p)
        end
        paramjac_noise_config = nothing
        jac_noise_config = nothing
      end
      if StochasticDiffEq.is_diagonal_noise(prob)
        pJ = similar(u0, numindvar, numparams)
        if isnoisemixing(sensealg)
          J = similar(u0, numindvar, numindvar)
        end
      else
        pJ = similar(u0, numindvar*numindvar, numparams)
        J = similar(u0, numindvar*numindvar, numindvar)
      end

    else
      paramjac_noise_config = nothing
      jac_noise_config = nothing
    end
  else
    paramjac_noise_config = nothing
    jac_noise_config = nothing
  end

  adjoint_cache = AdjointDiffCache(uf,pf,pg,J,pJ,dg_val,
                          jac_config,pg_config,paramjac_config,
                          jac_noise_config,paramjac_noise_config,
                          f_cache,dg,diffvar_idxs,algevar_idxs,
                          factorized_mass_matrix,issemiexplicitdae)

  return adjoint_cache, y
end

getprob(S::SensitivityFunction) = (S isa ODEBacksolveSensitivityFunction) ? S.prob : S.sol.prob

function generate_callbacks(sensefun, g, λ, t, callback, init_cb)
  sensefun.discrete || return callback

  @unpack sensealg, y = sensefun
  @unpack diffvar_idxs, algevar_idxs, factorized_mass_matrix, issemiexplicitdae, J, uf, f_cache, jac_config = sensefun.diffcache
  prob = getprob(sensefun)
  cur_time = Ref(length(t))
  time_choice = let cur_time=cur_time, t=t
    integrator -> cur_time[] > 0 ? t[cur_time[]] : nothing
  end
  affect! = let isq = (sensealg isa QuadratureAdjoint), λ=λ, t=t, y=y, cur_time=cur_time, idx=length(prob.u0), F=factorized_mass_matrix
    function (integrator)
      p, u = integrator.p, integrator.u
      # Warning: alias here! Be careful with λ
      gᵤ = isq ? λ : @view(λ[1:idx])
      g(gᵤ,y,p,t[cur_time[]],cur_time[])
      if issemiexplicitdae
        jacobian!(J, uf, y, f_cache, sensealg, jac_config)
        dhdd = J[algevar_idxs, diffvar_idxs]
        dhda = J[algevar_idxs, algevar_idxs]
        # TODO: maybe need a `conj`
        Δλa = -dhda'\gᵤ[algevar_idxs]
        Δλd = dhdd'Δλa + gᵤ[diffvar_idxs]
      else
        Δλd = gᵤ
      end

      if factorized_mass_matrix !== nothing
        F !== I && F !== (I,I) && ldiv!(F, Δλd)
      end
      u[diffvar_idxs] .+= Δλd
      u_modified!(integrator,true)
      cur_time[] -= 1
      return nothing
    end
  end
  cb = IterativeCallback(time_choice,affect!,eltype(prob.tspan);initial_affect=init_cb)
  return CallbackSet(cb,callback)
end
