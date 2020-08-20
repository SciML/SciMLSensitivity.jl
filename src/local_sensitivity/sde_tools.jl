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
  @unpack utmp, ducor, f, g = Tfunc

  copyto!(vec(utmp), u)
  fill!(ducor, zero(eltype(u)))

  vecjacobian!(ducor, utmp, p, t, Tfunc)

  f(du,u,p,t)
  @. du = du - ducor
  # else
  #   tmp1 = f(u,p,t)
  #   @. du = tmp1 - ducor
  # end

  return nothing
end


function (Tfunc::StochasticTransformedFunction)(u,p,t)
  @unpack f, g = Tfunc
  #ducor = vecjacobian(u, p, t, Tfunc)

  _dy, back = Zygote.pullback(u, p) do uloc, ploc
    vec(g(uloc, ploc, t))
  end
  ducor, _ = back(_dy)

  du = f(u,p,t)

  du = @. du - ducor
  return du
end
