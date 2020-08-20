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
  @unpack utmp, ducor, gtmp, f, g = Tfunc

  tape = ReverseDiff.GradientTape((u, p, [t])) do uloc,ploc,tloc
    du1 = similar(uloc, size(gtmp))
    g(du1,uloc,ploc,first(tloc))
    return vec(du1)
  end
  tu, tp, tt = ReverseDiff.input_hook(tape)

  output = ReverseDiff.output_hook(tape)

  #@show utmp  ReverseDiff.value(output)

  copyto!(vec(gtmp), ReverseDiff.value(output))#
  #@show utmp

  ReverseDiff.unseed!(tu) # clear any "leftover" derivatives from previous calls
  ReverseDiff.unseed!(tp)
  ReverseDiff.unseed!(tt)

  ReverseDiff.value!(tu, u)
  ReverseDiff.value!(tp, p)
  ReverseDiff.value!(tt, [t])

  ReverseDiff.forward_pass!(tape)
  ReverseDiff.increment_deriv!(output, vec(gtmp))
  ReverseDiff.reverse_pass!(tape)

  ReverseDiff.deriv(tu)
  ReverseDiff.pull_value!(output)
  copyto!(vec(ducor), ReverseDiff.deriv(tu))


  f(du,u,p,t)

  @. du = du - ducor

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
