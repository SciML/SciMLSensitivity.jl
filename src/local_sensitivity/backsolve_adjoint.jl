struct ODEBacksolveSensitivityFunction{C<:AdjointDiffCache,Alg<:BacksolveAdjoint,uType,pType,fType<:Union{ODEFunction, SDEFunction},CV} <: SensitivityFunction
  diffcache::C
  sensealg::Alg
  discrete::Bool
  y::uType
  prob::pType
  f::fType
  colorvec::CV
end


function ODEBacksolveSensitivityFunction(g,sensealg,discrete,sol,dg,f,colorvec)
  diffcache, y = adjointdiffcache(g,sensealg,discrete,sol,dg,f;quad=false)

  return ODEBacksolveSensitivityFunction(diffcache,sensealg,discrete,
                                         y,sol.prob,f,colorvec)
end

# u = λ'
function (S::ODEBacksolveSensitivityFunction)(du,u,p,t)
  @unpack y, prob, f, discrete = S
  idx = length(y)

  λ     = @view u[1:idx]
  grad  = @view u[idx+1:end-idx]
  _y    = @view u[end-idx+1:end]
  dλ    = @view du[1:idx]
  dgrad = @view du[idx+1:end-idx]
  dy    = @view du[end-idx+1:end]

  copyto!(vec(y), _y)

  vecjacobian!(dλ, λ, p, t, S, dgrad=dgrad, dy=dy)

  dλ .*= -one(eltype(λ))

  discrete || accumulate_dgdu!(dλ, y, p, t, S)
  return nothing
end


# g is either g(t,u,p) or discrete g(t,u,i)
@noinline function ODEAdjointProblem(sol,sensealg::BacksolveAdjoint,
                                     g,t=nothing,dg=nothing;
                                     checkpoints=sol.t,
                                     callback=CallbackSet(),kwargs...)
  @unpack f, p, u0, tspan = sol.prob
  tspan = reverse(tspan)
  discrete = t != nothing

  p === DiffEqBase.NullParameters() && error("Your model does not have parameters, and thus it is impossible to calculate the derivative of the solution with respect to the parameters. Your model must have parameters to use parameter sensitivity calculations!")
  numstates = length(u0)
  numparams = length(p)

  len = length(u0)+numparams
  λ = similar(u0, len)
  sense = ODEBacksolveSensitivityFunction(g,sensealg,discrete,sol,dg,f,f.colorvec)

  init_cb = t !== nothing && tspan[1] == t[end]
  cb = generate_callbacks(sense, g, λ, t, callback, init_cb)
  checkpoints = ischeckpointing(sensealg, sol) ? checkpoints : nothing
  if checkpoints !== nothing
    cb = backsolve_checkpoint_callbacks(sense, sol, checkpoints, cb)
  end

  z0 = [vec(zero(λ)); vec(sense.y)]
  original_mm = sol.prob.f.mass_matrix
  if original_mm === I
    mm = I
  else
    sense.diffcache.issemiexplicitdae && @warn "`BacksolveAdjoint` is likely to fail on semi-explicit DAEs, if memory is a concern, please consider using InterpolatingAdjoint(checkpoint=true) instead."
    len2 = length(z0)
    mm = zeros(len2, len2)
    idx = 1:numstates
    copyto!(@view(mm[idx, idx]), sol.prob.f.mass_matrix')
    idx = numstates+1:numstates+1+numparams
    copyto!(@view(mm[idx, idx]), I)
    idx = len+1:len2
    copyto!(@view(mm[idx, idx]), sol.prob.f.mass_matrix)
  end
  odefun = ODEFunction(sense, mass_matrix=mm)
  return ODEProblem(odefun,z0,tspan,p,callback=cb)
end



@noinline function SDEAdjointProblem(sol,sensealg::BacksolveAdjoint,
                                     g,t=nothing,dg=nothing;
                                     checkpoints=sol.t,
                                     callback=CallbackSet(),kwargs...)
  @unpack f, p, u0, tspan = sol.prob
  tspan = reverse(tspan)
  discrete = t != nothing

  p === DiffEqBase.NullParameters() && error("Your model does not have parameters, and thus it is impossible to calculate the derivative of the solution with respect to the parameters. Your model must have parameters to use parameter sensitivity calculations!")
  numstates = length(u0)
  numparams = length(p)

  len = length(u0)+numparams
  λ = similar(u0, len)

  drift_function = ODEFunction(sol.prob.f)
  sense_drift = ODEBacksolveSensitivityFunction(g,sensealg,discrete,sol,dg,drift_function,drift_function.colorvec)

  diffusion_function = ODEFunction(sol.prob.g)
  sense_diffusion = ODEBacksolveSensitivityFunction(g,sensealg,discrete,sol,dg,diffusion_function,diffusion_function.colorvec)

  init_cb = t !== nothing && tspan[1] == t[end]
  cb = generate_callbacks(sense_drift, g, λ, t, callback, init_cb)
  checkpoints = ischeckpointing(sensealg, sol) ? checkpoints : nothing
  if checkpoints !== nothing
    cb = backsolve_checkpoint_callbacks(sense_drift, sol, checkpoints, cb)
  end

  z0 = [vec(zero(λ)); vec(sense_drift.y)]
  #@show z0, vec(zero(λ)), sense_drift.y

  original_mm = sol.prob.f.mass_matrix
  if original_mm === I
    mm = I
  else
    sense_drift.diffcache.issemiexplicitdae && @warn "`BacksolveAdjoint` is likely to fail on semi-explicit DAEs, if memory is a concern, please consider using InterpolatingAdjoint(checkpoint=true) instead."
    len2 = length(z0)
    mm = zeros(len2, len2)
    idx = 1:numstates
    copyto!(@view(mm[idx, idx]), sol.prob.f.mass_matrix')
    idx = numstates+1:numstates+1+numparams
    copyto!(@view(mm[idx, idx]), I)
    idx = len+1:len2
    copyto!(@view(mm[idx, idx]), sol.prob.f.mass_matrix)
  end

  sdefun = SDEFunction(sense_drift,sense_diffusion,mass_matrix=mm)

  # replicated noise
  _sol = deepcopy(sol)
  backwardnoise = DiffEqNoiseProcess.NoiseGrid(reverse!(_sol.t),reverse!(_sol.W.W))

  return SDEProblem(sdefun,sense_diffusion,z0,tspan,p,callback=cb,
    noise=backwardnoise
    )
end



function backsolve_checkpoint_callbacks(sensefun, sol, checkpoints, callback)
  prob = sol.prob
  cur_time = Ref(length(checkpoints))
  condition = let checkpoints=checkpoints
    (u,t,integrator) ->
      checkpoints !== nothing && ((idx = searchsortedfirst(checkpoints, t)) <= length(checkpoints)) && checkpoints[idx] == t
  end
  affect! = let sol=sol, cur_time=cur_time, idx=length(prob.u0)
    function (integrator)
      _y = reshape(@view(integrator.u[end-idx+1:end]), axes(prob.u0))
      sol(_y, integrator.t)
      u_modified!(integrator,true)
      cur_time[] -= 1
      return nothing
    end
  end
  cb = DiscreteCallback(condition,affect!)

  return CallbackSet(cb,callback)
end
