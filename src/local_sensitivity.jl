immutable ODELocalSensitvityFunction{F,UF,PF,JC,PJC,P,A,fc,uEltype} <: SensitivityFunction
  f::F
  uf::UF
  pf::PF
  J::Matrix{uEltype}
  pJ::Matrix{uEltype}
  jac_config::JC
  paramjac_config::PJC
  p::P
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
                             p,alg,numparams,numindvar,f_cache)
end

function (S::ODELocalSensitvityFunction)(t,u,du)
  y = @view u[1:S.numindvar] # These are the independent variables
  S.f(t,y,@view du[1:S.numindvar]) # Make the first part be the ODE

  # Now do sensitivities
  # Compute the Jacobian

  if has_jac(S.f)
    S.f(Val{:jac},t,y,S.J) # Calculate the Jacobian into J
  else
    S.uf.t = t
    jacobian!(S.J, S.uf, y, S.f_cache, S.alg, S.jac_config)
  end

  if has_paramjac(S.f)
    S.f(Val{:paramjac},t,y,S.p,S.pJ) # Calculate the parameter Jacobian into pJ
  else
    S.pf.t = t
    S.pf.u .= y
    jacobian!(S.pJ, S.pf, S.p, S.f_cache, S.alg, S.paramjac_config)
  end

  # Compute the parameter derivatives
  for i in eachindex(S.f.params)
    Sj = @view u[i*S.numindvar+1:(i+1)*S.numindvar]
    dp = @view du[i*S.numindvar+1:(i+1)*S.numindvar]
    A_mul_B!(dp,S.J,Sj)
    dp .+= @view S.pJ[:,i]
  end
end

type ODELocalSensitivityProblem{uType,tType,isinplace,F,C,MM} <: AbstractODEProblem{uType,tType,isinplace}
  f::F
  u0::uType
  tspan::Tuple{tType,tType}
  indvars::Int
  callback::C
  mass_matrix::MM
end

function ODELocalSensitivityProblem(f,u0,tspan,alg=SensitivityAlg();
                                    callback=CallbackSet(),mass_matrix=I)
  isinplace = DiffEqBase.isinplace(f)
  indvars = length(u0)
  p = param_values(f)
  uf = DiffEqDiffTools.UJacobianWrapper(f,tspan[1])
  pf = DiffEqDiffTools.ParamJacobianWrapper(f,tspan[1],copy(u0))
  if has_jac(f)
    jac_config = nothing
  else
    jac_config = build_jac_config(alg,uf,u0)
  end

  if has_paramjac(f)
    paramjac_config = nothing
  else
    paramjac_config = build_param_jac_config(alg,pf,u0,p)
  end



  sense = ODELocalSensitvityFunction(f,uf,pf,u0,jac_config,paramjac_config,p,similar(u0),alg)
  sense_u0 = [u0;zeros(indvars*sense.numparams)]
  ODELocalSensitivityProblem{typeof(u0),typeof(tspan[1]),
                             isinplace,typeof(sense),typeof(callback),
                             typeof(mass_matrix)}(
                             sense,sense_u0,tspan,indvars,callback,mass_matrix)
end


function extract_local_sensitivities(sol)
  x = sol[1:sol.prob.indvars,:]
  x,[sol[sol.prob.indvars*j+1:sol.prob.indvars*(j+1),:] for j in 1:(num_params(sol.prob.f.f))]
end

function extract_local_sensitivities(sol,i::Integer)
  x = sol[1:sol.prob.indvars,i]
  x,[sol[sol.prob.indvars*j+1:sol.prob.indvars*(j+1),i] for j in 1:(num_params(sol.prob.f.f))]
end

function extract_local_sensitivities(sol,t)
  tmp = sol(t)
  x = tmp[1:sol.prob.indvars]
  x,[tmp[sol.prob.indvars*j+1:sol.prob.indvars*(j+1)] for j in 1:(num_params(sol.prob.f.f))]
end

function extract_local_sensitivities(tmp,sol,t)
  sol(tmp,t)
  x = @view tmp[1:sol.prob.indvars]
  x,[@view(tmp[sol.prob.indvars*j+1:sol.prob.indvars*(j+1)]) for j in 1:(num_params(sol.prob.f.f))]
end
