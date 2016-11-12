module DiffEqSensitivity

using DiffEqBase
immutable ODELocalSensitvityFunction{uEltype} <: SensitivityFunction
  f::ParameterizedFunction
  J::Matrix{uEltype}
  df::Vector{uEltype}
  numparams::Int
  numindvar::Int
end

function ODELocalSensitvityFunction(f,u0)
  numparams = length(f.params)
  numindvar = length(u0)
  J = Matrix{eltype(u0)}(numindvar,numindvar)
  df = Vector{eltype(u0)}(numindvar) # number of funcs size
  ODELocalSensitvityFunction(f,J,df,numparams,numindvar)
end

function (S::ODELocalSensitvityFunction)(t,u,du)
  y = @view u[1:S.numindvar] # These are the independent variables
  S.f(t,y,@view du[1:S.numindvar]) # Make the first part be the ODE
  S.f(Val{:Jac},t,y,S.J) # Calculate the Jacobian into J
  for i in eachindex(S.f.params)
    Sj = @view u[i*S.numindvar+1:(i+1)*S.numindvar]
    S.f(t,y,getfield(S.f,S.f.params[i]),S.df,S.f.params[i],:Deriv) # Calculate the parameter derivatives into df
    du[i*S.numindvar+1:(i+1)*S.numindvar] = S.J*Sj + S.df
  end
end

type ODELocalSensitivityProblem{uType,uEltype} <: AbstractODEProblem
  f::ODELocalSensitvityFunction{uEltype}
  u0::uType
  analytic::Function
  knownanalytic::Bool
  numvars::Int
  isinplace::Bool
end

function ODELocalSensitivityProblem(f::Function,u0)
  isinplace = numparameters(f)>=3
  numvars = length(u0)
  sense = ODELocalSensitvityFunction(f,u0)
  sense_u0 = [u0;zeros(numvars*sense.numparams)]
  ODELocalSensitivityProblem{typeof(u0),eltype(u0)}(sense,sense_u0,(t,u,du)->0,false,numvars,isinplace)
end

export ODELocalSensitvityFunction, ODELocalSensitivityProblem
end # module
