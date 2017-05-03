__precompile__()

module DiffEqSensitivity

using DiffEqBase, Compat

@compat abstract type SensitivityFunction end

immutable ODELocalSensitvityFunction{F,uEltype} <: SensitivityFunction
  f::F
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
  S.f(Val{:jac},t,y,S.J) # Calculate the Jacobian into J
  for i in eachindex(S.f.params)
    Sj = @view u[i*S.numindvar+1:(i+1)*S.numindvar]
    S.f(Val{:deriv},Val{S.f.params[i]},t,y,getfield(S.f,S.f.params[i]),S.df) # Calculate the parameter derivatives into df
    du[i*S.numindvar+1:(i+1)*S.numindvar] = S.J*Sj + S.df
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

function ODELocalSensitivityProblem(f,u0,tspan;callback=CallbackSet(),mass_matrix=I)
  isinplace = numargs(f)>=3
  indvars = length(u0)
  sense = ODELocalSensitvityFunction(f,u0)
  sense_u0 = [u0;zeros(indvars*sense.numparams)]
  ODELocalSensitivityProblem{typeof(u0),typeof(tspan[1]),
                             isinplace,typeof(sense),typeof(callback),
                             typeof(mass_matrix)}(
                             sense,sense_u0,tspan,indvars,callback,mass_matrix)
end

export ODELocalSensitvityFunction, ODELocalSensitivityProblem, SensitivityFunction
end # module
