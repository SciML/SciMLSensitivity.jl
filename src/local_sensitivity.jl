struct ODELocalSensitvityFunction{iip,F,A,J,PJ,UF,PF,JC,PJC,fc,uEltype} <: DiffEqBase.AbstractODEFunction{iip}
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
  numparams::Int
  numindvar::Int
  f_cache::fc
end

function ODELocalSensitvityFunction(f,analytic,jac,paramjac,uf,pf,u0,
                                    jac_config,paramjac_config,p,f_cache)
  numparams = length(p)
  numindvar = length(u0)
  J = Matrix{eltype(u0)}(undef,numindvar,numindvar)
  pJ = Matrix{eltype(u0)}(undef,numindvar,numparams) # number of funcs size
  ODELocalSensitvityFunction{isinplace(f),typeof(f),typeof(analytic),
                             typeof(jac),typeof(paramjac),typeof(uf),
                             typeof(pf),typeof(jac_config),
                             typeof(paramjac_config),typeof(f_cache),
                             eltype(u0)}(f,analytic,jac,paramjac,uf,pf,J,pJ,
                             jac_config,paramjac_config,
                             numparams,numindvar,f_cache)
end

function (S::ODELocalSensitvityFunction)(du,u,p,t)
  y = @view u[1:S.numindvar] # These are the independent variables
  dy = @view du[1:S.numindvar]
  S.f(dy,y,p,t) # Make the first part be the ODE

  # Now do sensitivities
  # Compute the Jacobian

  if DiffEqBase.has_jac(S.f)
    S.jac(S.J,y,p,t) # Calculate the Jacobian into J
  else
    S.uf.t = t
    jacobian!(S.J, S.uf, y, S.f_cache, S.alg, S.jac_config)
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
    mul!(dp,S.J,Sj)
    dp .+= @view S.pJ[:,i]
  end
end

struct ODELocalSensitivityProblem{uType,tupType,isinplace,P,F,C,MM} <: DiffEqBase.AbstractODEProblem{uType,tupType,isinplace}
  f::F
  u0::uType
  tspan::tupType
  p::P
  callback::C
  mass_matrix::MM
end

ODELocalSensitivityProblem(;f,u0,tspan,p,callback,mass_matrix) =
ODELocalSensitivityProblem(f,u0,tspan,p,callback,mass_matrix)

function ODELocalSensitivityProblem(f::DiffEqBase.AbstractODEFunction,u0,
                                    tspan,p=nothing;
                                    callback=CallbackSet(),mass_matrix=I)
  isinplace = DiffEqBase.isinplace(f)
  p == nothing && error("You must have parameters to use parameter sensitivity calculations!")
  uf = DiffEqDiffTools.UJacobianWrapper(f,tspan[1],p)
  pf = DiffEqDiffTools.ParamJacobianWrapper(f,tspan[1],copy(u0))
  if DiffEqBase.has_jac(f)
    jac_config = nothing
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
                                     paramjac_config,p,similar(u0))
  sense_u0 = [u0;zeros(sense.numindvar*sense.numparams)]
  ODEProblem(sense,sense_u0,tspan,p;callback=callback,mass_matrix=mass_matrix)
end

function ODELocalSensitivityProblem(f,args...;kwargs...)
  ODELocalSensitivityProblem(ODEFunction(f),args...;kwargs...)
end

function extract_local_sensitivities(sol)
  x = sol[1:sol.prob.f.numindvar,:]
  x,[sol[sol.prob.numindvar*j+1:sol.prob.f.numindvar*(j+1),:] for j in 1:(length(sol.prob.p))]
end

function extract_local_sensitivities(sol,i::Integer)
  x = sol[1:sol.prob.f.numindvar,i]
  x,[sol[sol.prob.numindvar*j+1:sol.prob.f.numindvar*(j+1),i] for j in 1:(length(sol.prob.p))]
end

function extract_local_sensitivities(sol,t)
  tmp = sol(t)
  x = tmp[1:sol.prob.f.numindvar]
  x,[tmp[sol.prob.numindvar*j+1:sol.prob.f.numindvar*(j+1)] for j in 1:(length(sol.prob.p))]
end

function extract_local_sensitivities(tmp,sol,t)
  sol(tmp,t)
  x = @view tmp[1:sol.prob.f.numindvar]
  x,[@view(tmp[sol.prob.numindvar*j+1:sol.f.prob.numindvar*(j+1)]) for j in 1:(length(sol.prob.p))]
end
