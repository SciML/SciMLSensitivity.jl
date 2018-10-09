struct ODELocalSensitvityFunction{iip,F,A,J,PJ,UF,PF,JC,PJC,Alg,fc,uEltype,MM} <: DiffEqBase.AbstractODEFunction{iip}
  f::F
  analytic::A
  jac::J
  paramjac::PJ
  uf::UF
  pf::PF
  J::Matrix{uEltype}
  pJ::Matrix{uEltype}
  jac_config::JC
  paramjac_config::PJC
  alg::Alg
  numparams::Int
  numindvar::Int
  f_cache::fc
  mass_matrix::MM
  isautojacvec::Bool
end

function ODELocalSensitvityFunction(f,analytic,jac,paramjac,uf,pf,u0,
                                    jac_config,paramjac_config,alg,p,f_cache,mm,
                                    isautojacvec)
  numparams = length(p)
  numindvar = length(u0)
  J = isautojacvec ? nothing : Matrix{eltype(u0)}(undef,numindvar,numindvar)
  pJ = Matrix{eltype(u0)}(undef,numindvar,numparams) # number of funcs size
  ODELocalSensitvityFunction{isinplace(f),typeof(f),typeof(analytic),
                             typeof(jac),typeof(paramjac),typeof(uf),
                             typeof(pf),typeof(jac_config),
                             typeof(paramjac_config),typeof(alg),
                             typeof(f_cache),
                             eltype(u0),typeof(mm)}(
                             f,analytic,jac,paramjac,uf,pf,J,pJ,
                             jac_config,paramjac_config,alg,
                             numparams,numindvar,f_cache,mm,isautojacvec)
end

function (S::ODELocalSensitvityFunction)(du,u,p,t)
  y = @view u[1:S.numindvar] # These are the independent variables
  dy = @view du[1:S.numindvar]
  S.f(dy,y,p,t) # Make the first part be the ODE

  # Now do sensitivities
  # Compute the Jacobian

  # TODO: priority
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
    if S.isautojacvec
      mul!(dp,S.J,Sj)
      jacobianvec!(dp, S.uf, y, Sj, S.f_cache, S.alg, S.jac_config)
    else
      mul!(dp,S.J,Sj)
    end
    dp .+= @view S.pJ[:,i]
  end
end

function ODELocalSensitivityProblem(f::DiffEqBase.AbstractODEFunction,u0,
                                    tspan,p=nothing,
                                    alg = SensitivityAlg();
                                    callback=CallbackSet(),mass_matrix=I)
  isinplace = DiffEqBase.isinplace(f)
  isautojacvec = get_jacvec(alg)
  p == nothing && error("You must have parameters to use parameter sensitivity calculations!")
  uf = isautojacvec ? nothing : DiffEqDiffTools.UJacobianWrapper(f,tspan[1],p)
  pf = DiffEqDiffTools.ParamJacobianWrapper(f,tspan[1],copy(u0))
  if DiffEqBase.has_jac(f)
    jac_config = nothing
  elseif isautojacvec
    jac_config = ForwardDiff.Dual{:___jac_tag}.(u0,u0)
    jac_config = jac_config, similar(jac_config)
  else
    jac_config = build_jac_config(alg,uf,u0)
  end

  if DiffEqBase.has_paramjac(f)
    paramjac_config = nothing
  else
    paramjac_config = build_param_jac_config(alg,pf,u0,p)
  end

  sense = ODELocalSensitvityFunction(f,f.analytic,f.jac,f.paramjac,
                                     uf,pf,u0,jac_config,
                                     paramjac_config,alg,
                                     p,similar(u0),mass_matrix,
                                     isautojacvec)
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
