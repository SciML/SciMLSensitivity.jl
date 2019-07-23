struct ODELocalSensitivityFunction{iip,F,A,Tt,J,JP,PJ,TW,TWt,UF,PF,JC,PJC,Alg,fc,JM,pJM,MM,CV} <: DiffEqBase.AbstractODEFunction{iip}
  f::F
  analytic::A
  tgrad::Tt
  jac::J
  jac_prototype::JP
  paramjac::PJ
  invW::TW
  invW_t::TWt
  uf::UF
  pf::PF
  J::JM
  pJ::pJM
  jac_config::JC
  paramjac_config::PJC
  alg::Alg
  numparams::Int
  numindvar::Int
  f_cache::fc
  mass_matrix::MM
  isautojacvec::Bool
  colorvec::CV
end

function ODELocalSensitivityFunction(f,analytic,tgrad,jac,jac_prototype,paramjac,invW,invW_t,uf,pf,u0,
                                    jac_config,paramjac_config,alg,p,f_cache,mm,
                                    isautojacvec,colorvec)
  numparams = length(p)
  numindvar = length(u0)
  J = isautojacvec ? nothing : Matrix{eltype(u0)}(undef,numindvar,numindvar)
  pJ = Matrix{eltype(u0)}(undef,numindvar,numparams) # number of funcs size
  ODELocalSensitivityFunction{isinplace(f),typeof(f),typeof(analytic),
                             typeof(tgrad),typeof(jac),typeof(jac_prototype),typeof(paramjac),
                             typeof(invW),typeof(invW_t),typeof(uf),
                             typeof(pf),typeof(jac_config),
                             typeof(paramjac_config),typeof(alg),
                             typeof(f_cache),
                             typeof(J),typeof(pJ),typeof(mm),typeof(f.colorvec)}(
                             f,analytic,tgrad,jac,jac_prototype,paramjac,invW,invW_t,uf,pf,J,pJ,
                             jac_config,paramjac_config,alg,
                             numparams,numindvar,f_cache,mm,isautojacvec,colorvec)
end

function (S::ODELocalSensitivityFunction)(du,u,p,t)
  y = @view u[1:S.numindvar] # These are the independent variables
  dy = @view du[1:S.numindvar]
  S.f(dy,y,p,t) # Make the first part be the ODE

  # Now do sensitivities
  # Compute the Jacobian

  if !S.isautojacvec
    if DiffEqBase.has_jac(S.f)
      S.jac(S.J,y,p,t) # Calculate the Jacobian into J
    else
      S.uf.t = t
      jacobian!(S.J, S.uf, y, S.f_cache, S.alg, S.jac_config)
    end
  end

  if DiffEqBase.has_paramjac(S.f)
    S.paramjac(S.pJ,y,p,t) # Calculate the parameter Jacobian into pJ
  else
    S.pf.t = t
    S.pf.u .= y
    jacobian!(S.pJ, S.pf, p, S.f_cache, S.alg, S.paramjac_config)
  end

  # Compute the parameter derivatives
  for i in eachindex(p)
    Sj = @view u[i*S.numindvar+1:(i+1)*S.numindvar]
    dp = @view du[i*S.numindvar+1:(i+1)*S.numindvar]
    if !S.isautojacvec
      mul!(dp,S.J,Sj)
    else
      jacobianvec!(dp, S.uf, y, Sj, S.alg, S.jac_config)
    end
    dp .+= @view S.pJ[:,i]
  end
end

function ODELocalSensitivityProblem(f::DiffEqBase.AbstractODEFunction,u0,
                                    tspan,p=nothing,
                                    alg = SensitivityAlg(autojacvec=true);
                                    callback=CallbackSet(),mass_matrix=I)
  isinplace = DiffEqBase.isinplace(f)
  # if there is an analytical Jacobian provided, we are not going to do automatic `jac*vec`
  isautojacvec = get_jacvec(alg)
  p == nothing && error("You must have parameters to use parameter sensitivity calculations!")
  uf = DiffEqDiffTools.UJacobianWrapper(f,tspan[1],p)
  pf = DiffEqDiffTools.ParamJacobianWrapper(f,tspan[1],copy(u0))
  if isautojacvec
    if alg_autodiff(alg)
      # if we are using automatic `jac*vec`, then we need to use a `jac_config`
      # that is a tuple in the form of `(seed, buffer)`
      jac_config_seed = ForwardDiff.Dual{:___jac_tag}.(u0,u0)
      jac_config_buffer = similar(jac_config_seed)
      jac_config = jac_config_seed, jac_config_buffer
    else
      jac_config = (similar(u0),similar(u0))
    end
  elseif DiffEqBase.has_jac(f)
    jac_config = nothing
  else
    jac_config = build_jac_config(alg,uf,u0)
  end

  if DiffEqBase.has_paramjac(f)
    paramjac_config = nothing
  else
    paramjac_config = build_param_jac_config(alg,pf,u0,p)
  end

  # TODO: Use user tgrad. iW can be safely ignored here.
  sense = ODELocalSensitivityFunction(f,f.analytic,nothing,f.jac,f.jac_prototype,f.paramjac,nothing,nothing,
                                     uf,pf,u0,jac_config,
                                     paramjac_config,alg,
                                     p,similar(u0),mass_matrix,
                                     isautojacvec,f.colorvec)
  sense_u0 = [u0;zeros(sense.numindvar*sense.numparams)]
  ODEProblem(sense,sense_u0,tspan,p;callback=callback)
end

function ODELocalSensitivityProblem(f,args...;kwargs...)
  ODELocalSensitivityProblem(ODEFunction(f),args...;kwargs...)
end

function extract_local_sensitivities(sol)
  x = sol[1:sol.prob.f.numindvar,:]
  x,[sol[sol.prob.f.numindvar*j+1:sol.prob.f.numindvar*(j+1),:] for j in 1:(length(sol.prob.p))]
end

function extract_local_sensitivities(sol,i::Integer)
  x = sol[1:sol.prob.f.numindvar,i]
  x,[sol[sol.prob.f.numindvar*j+1:sol.prob.f.numindvar*(j+1),i] for j in 1:(length(sol.prob.p))]
end

function extract_local_sensitivities(sol,t)
  tmp = sol(t)
  x = tmp[1:sol.prob.f.numindvar]
  x,[tmp[sol.prob.f.numindvar*j+1:sol.prob.f.numindvar*(j+1)] for j in 1:(length(sol.prob.p))]
end

function extract_local_sensitivities(tmp,sol,t)
  sol(tmp,t)
  x = @view tmp[1:sol.prob.f.numindvar]
  x,[@view(tmp[sol.prob.f.numindvar*j+1:sol.prob.f.numindvar*(j+1)]) for j in 1:(length(sol.prob.p))]
end
