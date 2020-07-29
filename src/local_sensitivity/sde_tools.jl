# for Ito / Stratonovich conversion
struct StochasticTransformedFunction{uType,duType,pType,fType<:DiffEqBase.AbstractDiffEqFunction,gType,noiseType} <: TransformedFunction
  utmp::uType
  ducor::duType
  prob::pType
  f::fType
  g::gType
  gtmp::noiseType
  inplace::Bool
end


function StochasticTransformedFunction(sol,f,g)
  @unpack prob = sol
  utmp = copy(sol.u[end])
  ducor = copy(sol.u[end])
  if StochasticDiffEq.is_diagonal_noise(prob)
    gtmp = copy(sol.u[end])
  else
    gtmp = similar(prob.p, size(prob.noise_rate_prototype))
  end

  return StochasticTransformedFunction(utmp,ducor,prob,f,g,gtmp,DiffEqBase.isinplace(prob))
end


function (Tfunc::StochasticTransformedFunction)(du,u,p,t)
  @unpack utmp, ducor, prob, f, g = Tfunc

  copyto!(vec(utmp), u)
  fill!(ducor, zero(eltype(u)))

  vecjacobian!(ducor, utmp, p, t, Tfunc)

  if DiffEqBase.isinplace(prob)
    f(du,u,p,t)
    @. du = du - ducor
  else
    tmp1 = f(u,p,t)
    @. du = tmp1 - ducor
  end

  return nothing
end


function (Tfunc::StochasticTransformedFunction)(u,p,t)
  @unpack prob, f, g = Tfunc

  utmp = copy(u)

  ducor = vecjacobian(utmp, p, t, Tfunc)
  if DiffEqBase.isinplace(prob)
    du = zero(u)
    f(du,u,p,t)
    return (du - ducor)
  else
    tmp1 = f(u,p,t)
    du = tmp1 - ducor
    return du
  end
end
