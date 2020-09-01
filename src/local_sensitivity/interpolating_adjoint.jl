struct ODEInterpolatingAdjointSensitivityFunction{C<:AdjointDiffCache,Alg<:InterpolatingAdjoint,
                                                  uType,SType,CPS,pType,fType<:DiffEqBase.AbstractDiffEqFunction} <: SensitivityFunction
  diffcache::C
  sensealg::Alg
  discrete::Bool
  y::uType
  sol::SType
  checkpoint_sol::CPS
  prob::pType
  f::fType
  noiseterm::Bool
end

mutable struct CheckpointSolution{S,I,T}
  cpsol::S # solution in a checkpoint interval
  intervals::I # checkpoint intervals
  cursor::Int # sol.prob.tspan = intervals[cursor]
  tols::T
end

function ODEInterpolatingAdjointSensitivityFunction(g,sensealg,discrete,sol,dg,f,checkpoints,tols;noiseterm=false)
  tspan = reverse(sol.prob.tspan)
  checkpointing = ischeckpointing(sensealg, sol)
  (checkpointing && checkpoints === nothing) && error("checkpoints must be passed when checkpointing is enabled.")

  checkpoint_sol = if checkpointing
    intervals = map(tuple, @view(checkpoints[1:end-1]), @view(checkpoints[2:end]))
    interval_end = intervals[end][end]
    tspan[1] > interval_end && push!(intervals, (interval_end, tspan[1]))
    cursor = lastindex(intervals)
    interval = intervals[cursor]

    if typeof(sol.prob) <: SDEProblem
      # replicated noise
      _sol = deepcopy(sol)
      sol.W.save_everystep = false
      _sol.W.save_everystep = false
      idx1 = searchsortedfirst(_sol.t, interval[1]-1000eps(interval[1]))
      #idx2 = searchsortedfirst(_sol.t, interval[2])
      #forwardnoise = DiffEqNoiseProcess.NoiseGrid(_sol.t[idx1:idx2], _sol.W.W[idx1:idx2])
      forwardnoise = DiffEqNoiseProcess.NoiseWrapper(_sol.W, indx=idx1)
      dt = abs(_sol.W.dt)
      if dt < 1000eps(_sol.t[end])
        dt = interval[2] - interval[1]
      end
      cpsol = solve(remake(sol.prob, tspan=interval, u0=sol(interval[1]), noise=forwardnoise), sol.alg, save_noise=false; dt=dt, tstops=_sol.t[idx1:end] ,tols...)
    else
      cpsol = solve(remake(sol.prob, tspan=interval, u0=sol(interval[1])), sol.alg; tols...)
    end
    CheckpointSolution(cpsol, intervals, cursor, tols)
  else
    nothing
  end

  diffcache, y = adjointdiffcache(g,sensealg,discrete,sol,dg,f;quad=false,noiseterm=noiseterm)

  return ODEInterpolatingAdjointSensitivityFunction(diffcache,sensealg,
                                                    discrete,y,sol,
                                                    checkpoint_sol,sol.prob,f,noiseterm)
end

function findcursor(intervals, t)
  # equivalent with `findfirst(x->x[1] <= t <= x[2], intervals)`
  lt(x, t) = <(x[2], t)
  return searchsortedfirst(intervals, t, lt=lt)
end

# u = λ'
# add tstop on all the checkpoints
function (S::ODEInterpolatingAdjointSensitivityFunction)(du,u,p,t)
  @unpack sol, y, checkpoint_sol, discrete, prob, f = S
  idx = length(y)

  if checkpoint_sol === nothing
    if typeof(t) <: ForwardDiff.Dual && eltype(S.y) <: AbstractFloat
      y = sol(t)
    else
      sol(y,t)
    end
  else
    intervals = checkpoint_sol.intervals
    interval = intervals[checkpoint_sol.cursor]
    if !(interval[1] <= t <= interval[2])
      cursor′ = findcursor(intervals, t)
      interval = intervals[cursor′]
      cpsol_t = checkpoint_sol.cpsol.t
      if typeof(t) <: ForwardDiff.Dual && eltype(S.y) <: AbstractFloat
        y = sol(interval[1])
      else
        sol(y, interval[1])
      end
      if typeof(sol.prob) <: SDEProblem
        #idx1 = searchsortedfirst(sol.t, interval[1])
        _sol = deepcopy(sol)
        _sol.W.save_everystep = false
        idx1 = searchsortedfirst(_sol.t, interval[1]-100eps(interval[1]))
        idx2 = searchsortedfirst(_sol.t, interval[2])
        #forwardnoise = DiffEqNoiseProcess.NoiseGrid(_sol.t[idx1:idx2], _sol.W.W[idx1:idx2])
        forwardnoise = DiffEqNoiseProcess.NoiseWrapper(_sol.W, indx=idx1)
        prob′ = remake(prob, tspan=intervals[cursor′], u0=y, noise=forwardnoise)
        dt = abs(cpsol_t[end]-cpsol_t[end-1])
        if dt < 10000eps(cpsol_t[end])
          dt = interval[2] - interval[1]
        end
        cpsol′ = solve(prob′, sol.alg, noise=forwardnoise, save_noise=false; dt=dt, tstops=_sol.t[idx1:idx2], checkpoint_sol.tols...)
      else
        prob′ = remake(prob, tspan=intervals[cursor′], u0=y)
        cpsol′ = solve(prob′, sol.alg; dt=abs(cpsol_t[end] - cpsol_t[end-1]), checkpoint_sol.tols...)
      end
      checkpoint_sol.cpsol = cpsol′
      checkpoint_sol.cursor = cursor′
    end
    checkpoint_sol.cpsol(y, t)
  end

  λ     = @view u[1:idx]
  grad  = @view u[idx+1:end]

  if length(u) == length(du)
    dλ    = @view du[1:idx]
    dgrad = @view du[idx+1:end]

  elseif length(u) != length(du) &&  StochasticDiffEq.is_diagonal_noise(prob) && !isnoisemixing(S.sensealg)
    idx1 = [length(u)*(i-1)+i for i in 1:idx] # for diagonal indices of [1:idx,1:idx]

    dλ    = @view du[idx1]
    dgrad = @view du[idx+1:end,1:idx]

  else
    # non-diagonal noise and noise mixing case
    dλ    = @view du[1:idx,1:idx]
    dgrad = @view du[idx+1:end,1:idx]
  end

  if S.noiseterm
    if length(u) == length(du)
      vecjacobian!(dλ, y, λ, p, t, S, dgrad=dgrad)
    elseif length(u) != length(du) &&  StochasticDiffEq.is_diagonal_noise(prob) && !isnoisemixing(S.sensealg)
      vecjacobian!(dλ, y, λ, p, t, S)
      jacNoise!(λ, y, p, t, S, dgrad=dgrad)
    else
      jacNoise!(λ, y, p, t, S, dgrad=dgrad, dλ=dλ)
    end
  else
    vecjacobian!(dλ, y, λ, p, t, S, dgrad=dgrad)
  end

  dλ .*= -one(eltype(λ))

  discrete || accumulate_cost!(dλ, y, p, t, S, dgrad)
  return nothing
end

# g is either g(t,u,p) or discrete g(t,u,i)
@noinline function ODEAdjointProblem(sol,sensealg::InterpolatingAdjoint,
                                     g,t=nothing,dg=nothing;
                                     checkpoints=sol.t,
                                     callback=CallbackSet(),
                                     reltol=nothing, abstol=nothing,
                                     kwargs...)
  @unpack f, p, u0, tspan = sol.prob
  tspan = reverse(tspan)
  discrete = t != nothing

  numstates = length(u0)
  numparams = p === nothing || p === DiffEqBase.NullParameters() ? 0 : length(p)

  len = numstates+numparams

  λ = p === nothing || p === DiffEqBase.NullParameters() ? similar(u0) : one(eltype(u0)) .* similar(p, len)
  λ .= false

  sense = ODEInterpolatingAdjointSensitivityFunction(g,sensealg,discrete,sol,dg,f,
                                                     checkpoints,
                                                     (reltol=reltol,abstol=abstol))

  init_cb = t !== nothing && tspan[1] == t[end]
  cb = generate_callbacks(sense, g, λ, t, callback, init_cb)
  z0 = vec(zero(λ))
  original_mm = sol.prob.f.mass_matrix
  if original_mm === I || original_mm === (I,I)
    mm = I
  else
    adjmm = copy(sol.prob.f.mass_matrix')
    zzz = similar(adjmm, numstates, numparams)
    fill!(zzz, zero(eltype(zzz)))
    # using concrate I is slightly more efficient
    II = Diagonal(I, numparams)
    mm = [adjmm       zzz
          copy(zzz')   II]
  end

  jac_prototype = sol.prob.f.jac_prototype
  if !sense.discrete || jac_prototype === nothing
    adjoint_jac_prototype = nothing
  else
    _adjoint_jac_prototype = copy(jac_prototype')
    zzz = similar(_adjoint_jac_prototype, numstates, numparams)
    fill!(zzz, zero(eltype(zzz)))
    II = Diagonal(I, numparams)
    adjoint_jac_prototype = [_adjoint_jac_prototype zzz
                             copy(zzz')             II]
  end

  odefun = ODEFunction(sense, mass_matrix=mm, jac_prototype=adjoint_jac_prototype)
  return ODEProblem(odefun,z0,tspan,p,callback=cb)
end


@noinline function SDEAdjointProblem(sol,sensealg::InterpolatingAdjoint,
                                     g,t=nothing,dg=nothing;
                                     checkpoints=sol.t,
                                     callback=CallbackSet(),
                                     reltol=nothing, abstol=nothing,
                                     diffusion_jac=nothing, diffusion_paramjac=nothing,
                                     kwargs...)
  @unpack f, p, u0, tspan = sol.prob
  tspan = reverse(tspan)
  discrete = t != nothing

  if length(unique(round.(checkpoints, digits=13))) != length(checkpoints)
    @warn "The given checkpoints are not unique. To avoid issues in the interpolation the checkpoints were redefined. You may want to check sol.t if default checkpoints were used."
    checkpoints = unique(round.(checkpoints, digits=13))
  end


  p === DiffEqBase.NullParameters() && error("Your model does not have parameters, and thus it is impossible to calculate the derivative of the solution with respect to the parameters. Your model must have parameters to use parameter sensitivity calculations!")
  numstates = length(u0)
  numparams = length(p)

  len = numstates+numparams

  λ = one(eltype(u0)) .* similar(p, len)
  λ .= false

  sense_drift = ODEInterpolatingAdjointSensitivityFunction(g,sensealg,discrete,sol,dg,sol.prob.f,
                                                     checkpoints,(reltol=reltol,abstol=abstol))

  diffusion_function = ODEFunction(sol.prob.g, jac=diffusion_jac, paramjac=diffusion_paramjac)
  sense_diffusion = ODEInterpolatingAdjointSensitivityFunction(g,sensealg,discrete,sol,dg,diffusion_function,
                                                     checkpoints,(reltol=reltol,abstol=abstol);noiseterm=true)

  init_cb = t !== nothing && tspan[1] == t[end]
  cb = generate_callbacks(sense_drift, g, λ, t, callback, init_cb)
  z0 = vec(zero(λ))
  original_mm = sol.prob.f.mass_matrix
  if original_mm === I || original_mm === (I,I)
    mm = I
  else
    adjmm = copy(sol.prob.f.mass_matrix')
    zzz = similar(adjmm, numstates, numparams)
    fill!(zzz, zero(eltype(zzz)))
    # using concrate I is slightly more efficient
    II = Diagonal(I, numparams)
    mm = [adjmm       zzz
          copy(zzz')   II]
  end

  jac_prototype = sol.prob.f.jac_prototype
  if !sense_drift.discrete || jac_prototype === nothing
    adjoint_jac_prototype = nothing
  else
    _adjoint_jac_prototype = copy(jac_prototype')
    zzz = similar(_adjoint_jac_prototype, numstates, numparams)
    fill!(zzz, zero(eltype(zzz)))
    II = Diagonal(I, numparams)
    adjoint_jac_prototype = [_adjoint_jac_prototype zzz
                             copy(zzz')             II]
  end

  sdefun = SDEFunction(sense_drift,sense_diffusion,mass_matrix=mm,jac_prototype=adjoint_jac_prototype)

  # replicated noise
  _sol = deepcopy(sol)
  backwardnoise = DiffEqNoiseProcess.NoiseWrapper(_sol.W, reverse=true)

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
