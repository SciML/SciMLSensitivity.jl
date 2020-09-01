struct ODEBacksolveSensitivityFunction{C<:AdjointDiffCache,Alg<:BacksolveAdjoint,uType,pType,fType<:DiffEqBase.AbstractDiffEqFunction} <: SensitivityFunction
  diffcache::C
  sensealg::Alg
  discrete::Bool
  y::uType
  prob::pType
  f::fType
  noiseterm::Bool
end


function ODEBacksolveSensitivityFunction(g,sensealg,discrete,sol,dg,f;noiseterm=false)
  diffcache, y = adjointdiffcache(g,sensealg,discrete,sol,dg,f;quad=false,noiseterm=noiseterm)

  return ODEBacksolveSensitivityFunction(diffcache,sensealg,discrete,
                                         y,sol.prob,f,noiseterm)
end

# u = λ'
function (S::ODEBacksolveSensitivityFunction)(du,u,p,t)
  @unpack y, prob, f, discrete = S
  idx = length(y)

  λ     = @view u[1:idx]
  grad  = @view u[idx+1:end-idx]
  _y    = @view u[end-idx+1:end]

  if length(u) == length(du)
    # ODE/Drift term and scalar noise
    dλ    = @view du[1:idx]
    dgrad = @view du[idx+1:end-idx]
    dy    = @view du[end-idx+1:end]

  elseif length(u) != length(du) &&  StochasticDiffEq.is_diagonal_noise(prob) && !isnoisemixing(S.sensealg)
    # Diffusion term, diagonal noise, length(du) =  u*m
    idx1 = [length(u)*(i-1)+i for i in 1:idx] # for diagonal indices of [1:idx,1:idx]
    idx2 = [(length(u)+1)*i-idx for i in 1:idx] # for diagonal indices of [end-idx+1:end,1:idx]

    dλ    = @view du[idx1]
    dgrad = @view du[idx+1:end-idx,1:idx]
    dy    = @view du[idx2]

  elseif length(u) != length(du) &&  StochasticDiffEq.is_diagonal_noise(prob) && isnoisemixing(S.sensealg)
    # Diffusion term, diagonal noise, (as above but can handle mixing noise terms)
    idx2 = [(length(u)+1)*i-idx for i in 1:idx] # for diagonal indices of [end-idx+1:end,1:idx]

    dλ    = @view du[1:idx,1:idx]
    dgrad = @view du[idx+1:end-idx,1:idx]
    dy    = @view du[idx2]

  else
    # non-diagonal noise
    dλ    = @view du[1:idx, 1:idx]
    dgrad = @view du[idx+1:end-idx,1:idx]
    dy    = @view du[end-idx+1:end, 1:idx]
  end

  copyto!(vec(y), _y)

  if S.noiseterm
    if length(u) == length(du)
      vecjacobian!(dλ, y, λ, p, t, S, dgrad=dgrad,dy=dy)
    elseif length(u) != length(du) &&  StochasticDiffEq.is_diagonal_noise(prob) && !isnoisemixing(S.sensealg)
      vecjacobian!(dλ, y, λ, p, t, S, dy=dy)
      jacNoise!(λ, y, p, t, S, dgrad=dgrad)
    else
      jacNoise!(λ, y, p, t, S, dgrad=dgrad, dλ=dλ, dy=dy)
    end
  else
    vecjacobian!(dλ, y, λ, p, t, S, dgrad=dgrad, dy=dy)
  end

  dλ .*= -one(eltype(λ))

  discrete || accumulate_cost!(dλ, y, p, t, S, dgrad)
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

  numstates = length(u0)
  numparams = p === nothing || p === DiffEqBase.NullParameters() ? 0 : length(p)

  len = length(u0)+numparams
  λ = p === nothing || p === DiffEqBase.NullParameters() ? similar(u0) : one(eltype(u0)) .* similar(p, len)
  λ .= false
  sense = ODEBacksolveSensitivityFunction(g,sensealg,discrete,sol,dg,f)

  init_cb = t !== nothing && tspan[1] == t[end]
  cb = generate_callbacks(sense, g, λ, t, callback, init_cb)
  checkpoints = ischeckpointing(sensealg, sol) ? checkpoints : nothing
  if checkpoints !== nothing
    cb = backsolve_checkpoint_callbacks(sense, sol, checkpoints, cb)
  end

  z0 = [vec(zero(λ)); vec(sense.y)]
  original_mm = sol.prob.f.mass_matrix
  zzz(A, m, n) = fill!(similar(A, m, n), zero(eltype(original_mm)))
  if original_mm === I || original_mm === (I,I)
    mm = I
  else
    sense.diffcache.issemiexplicitdae && @warn "`BacksolveAdjoint` is likely to fail on semi-explicit DAEs, if memory is a concern, please consider using InterpolatingAdjoint(checkpoint=true) instead."
    II = Diagonal(I, numparams)
    Z1 = zzz(original_mm, numstates, numstates+numparams)
    Z2 = zzz(original_mm, numparams, numstates)
    mm = [copy(original_mm')   Z1
          Z2                   II  Z2
          Z1                       original_mm]
  end
  jac_prototype = sol.prob.f.jac_prototype
  if !sense.discrete || jac_prototype === nothing
    adjoint_jac_prototype = nothing
  else
    J = jac_prototype
    Ja = copy(J')
    II = Diagonal(I, numparams)
    Z1 = zzz(J, numstates, numstates+numparams)
    Z2 = zzz(J, numparams, numstates)
    adjoint_jac_prototype = [Ja       Z1
                             Z2       II     Z2
                             Z1              J]
  end
  odefun = ODEFunction(sense, mass_matrix=mm)
  return ODEProblem(odefun,z0,tspan,p,callback=cb)
end



@noinline function SDEAdjointProblem(sol,sensealg::BacksolveAdjoint,
                                     g,t=nothing,dg=nothing;
                                     checkpoints=sol.t,
                                     callback=CallbackSet(),
                                     diffusion_jac=nothing, diffusion_paramjac=nothing,kwargs...)
  @unpack f, p, u0, tspan = sol.prob
  tspan = reverse(tspan)
  discrete = t != nothing

  p === DiffEqBase.NullParameters() && error("Your model does not have parameters, and thus it is impossible to calculate the derivative of the solution with respect to the parameters. Your model must have parameters to use parameter sensitivity calculations!")

  numstates = length(u0)
  numparams = length(p)

  len = length(u0)+numparams
  λ = one(eltype(u0)) .* similar(p, len)

  sense_drift = ODEBacksolveSensitivityFunction(g,sensealg,discrete,sol,dg,sol.prob.f)

  diffusion_function = ODEFunction(sol.prob.g, jac=diffusion_jac, paramjac=diffusion_paramjac)
  sense_diffusion = ODEBacksolveSensitivityFunction(g,sensealg,discrete,sol,dg,diffusion_function;noiseterm=true)

  init_cb = t !== nothing && tspan[1] == t[end]
  cb = generate_callbacks(sense_drift, g, λ, t, callback, init_cb)
  checkpoints = ischeckpointing(sensealg, sol) ? checkpoints : nothing
  if checkpoints !== nothing
    cb = backsolve_checkpoint_callbacks(sense_drift, sol, checkpoints, cb)
  end

  z0 = [vec(zero(λ)); vec(sense_drift.y)]

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
  _sol.W.save_everystep = false
  backwardnoise = DiffEqNoiseProcess.NoiseWrapper(_sol.W, reverse=true)
  #backwardnoise = DiffEqNoiseProcess.NoiseGrid(reverse!(_sol.t),reverse!( _sol.W.W))

  if StochasticDiffEq.is_diagonal_noise(sol.prob) && typeof(sol.W[end])<:Number
    # scalar noise case
    noise_matrix = nothing
  else
    noise_matrix = similar(z0,length(z0),numstates)
    noise_matrix .= false
  end

  return SDEProblem(sdefun,sense_diffusion,z0,tspan,p,
    callback=cb,
    noise=backwardnoise,
    noise_rate_prototype = noise_matrix
    )
end



function backsolve_checkpoint_callbacks(sensefun, sol, checkpoints, callback)
  prob = sol.prob
  cur_time = Ref(length(checkpoints))
  affect! = let sol=sol, cur_time=cur_time, idx=length(prob.u0)
    function (integrator)
      _y = reshape(@view(integrator.u[end-idx+1:end]), axes(prob.u0))
      sol(_y, integrator.t)
      u_modified!(integrator,true)
      cur_time[] -= 1
      return nothing
    end
  end

  cb = PresetTimeCallback(checkpoints,affect!)

  return CallbackSet(cb,callback)
end
