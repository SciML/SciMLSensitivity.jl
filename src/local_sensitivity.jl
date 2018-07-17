struct ODELocalSensitvityFunction{F,UF,PF,JC,PJC,A,fc,uEltype} <: SensitivityFunction
  f::F
  uf::UF
  pf::PF
  J::Matrix{uEltype}
  pJ::Matrix{uEltype}
  jac_config::JC
  paramjac_config::PJC
  alg::A
  numparams::Int
  numindvar::Int
  f_cache::fc
end

function ODELocalSensitvityFunction(f,uf,pf,u0,jac_config,paramjac_config,p,f_cache,alg)
  numparams = length(p)
  numindvar = length(u0)
  J = Matrix{eltype(u0)}(numindvar,numindvar)
  pJ = Matrix{eltype(u0)}(numindvar,numparams) # number of funcs size
  ODELocalSensitvityFunction(f,uf,pf,J,pJ,jac_config,paramjac_config,
                             alg,numparams,numindvar,f_cache)
end

function (S::ODELocalSensitvityFunction)(du,u,p,t)
  y = @view u[1:S.numindvar] # These are the independent variables
  dy = @view du[1:S.numindvar]
  S.f(dy,y,p,t) # Make the first part be the ODE

  # Now do sensitivities
  # Compute the Jacobian

  if DiffEqBase.has_jac(S.f)
    S.f(Val{:jac},S.J,y,p,t) # Calculate the Jacobian into J
  else
    S.uf.t = t
    jacobian!(S.J, S.uf, y, S.f_cache, S.alg, S.jac_config)
  end

  if DiffEqBase.has_paramjac(S.f)
    S.f(Val{:paramjac},S.pJ,y,p,t) # Calculate the parameter Jacobian into pJ
  else
    S.pf.t = t
    S.pf.u .= y
    jacobian!(S.pJ, S.pf, p, S.f_cache, S.alg, S.paramjac_config)
  end

  # Compute the parameter derivatives
  for i in eachindex(p)
    Sj = @view u[i*S.numindvar+1:(i+1)*S.numindvar]
    dp = @view du[i*S.numindvar+1:(i+1)*S.numindvar]
    A_mul_B!(dp,S.J,Sj)
    dp .+= @view S.pJ[:,i]
  end
end

mutable struct ODELocalSensitivityProblem{uType,tType,isinplace,P,F,C,MM} <: DiffEqBase.AbstractODEProblem{uType,tType,isinplace}
  f::F
  u0::uType
  tspan::Tuple{tType,tType}
  p::P
  indvars::Int
  callback::C
  mass_matrix::MM
end

function ODELocalSensitivityProblem(f,u0,tspan,p=nothing,alg=SensitivityAlg();
                                    callback=CallbackSet(),mass_matrix=I)
  isinplace = DiffEqBase.isinplace(f,4)
  indvars = length(u0)

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



  sense = ODELocalSensitvityFunction(f,uf,pf,u0,jac_config,paramjac_config,p,similar(u0),alg)
  sense_u0 = [u0;zeros(indvars*sense.numparams)]
  ODELocalSensitivityProblem{typeof(u0),typeof(tspan[1]),
                             isinplace,typeof(p),typeof(sense),typeof(callback),
                             typeof(mass_matrix)}(
                             sense,sense_u0,tspan,p,indvars,callback,mass_matrix)
end


function extract_local_sensitivities(sol)
  x = sol[1:sol.prob.indvars,:]
  x,[sol[sol.prob.indvars*j+1:sol.prob.indvars*(j+1),:] for j in 1:(length(sol.prob.p))]
end

function extract_local_sensitivities(sol,i::Integer)
  x = sol[1:sol.prob.indvars,i]
  x,[sol[sol.prob.indvars*j+1:sol.prob.indvars*(j+1),i] for j in 1:(length(sol.prob.p))]
end

function extract_local_sensitivities(sol,t)
  tmp = sol(t)
  x = tmp[1:sol.prob.indvars]
  x,[tmp[sol.prob.indvars*j+1:sol.prob.indvars*(j+1)] for j in 1:(length(sol.prob.p))]
end

function extract_local_sensitivities(tmp,sol,t)
  sol(tmp,t)
  x = @view tmp[1:sol.prob.indvars]
  x,[@view(tmp[sol.prob.indvars*j+1:sol.prob.indvars*(j+1)]) for j in 1:(length(sol.prob.p))]
end
