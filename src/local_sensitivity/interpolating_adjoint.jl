struct ODEInterpolatingAdjointSensitivityFunction{C<:AdjointDiffCache,Alg<:InterpolatingAdjoint,
                                                  uType,SType,CPS,fType<:DiffEqBase.AbstractDiffEqFunction} <: SensitivityFunction
  diffcache::C
  sensealg::Alg
  discrete::Bool
  y::uType
  sol::SType
  checkpoint_sol::CPS
  f::fType
end

mutable struct CheckpointSolution{S,I,T}
  cpsol::S # solution in a checkpoint interval
  intervals::I # checkpoint intervals
  cursor::Int # sol.prob.tspan = intervals[cursor]
  tols::T
end

function ODEInterpolatingAdjointSensitivityFunction(g,sensealg,discrete,sol,dg,checkpoints,tols)
  tspan = reverse(sol.prob.tspan)
  checkpointing = ischeckpointing(sensealg, sol)
  (checkpointing && checkpoints === nothing) && error("checkpoints must be passed when checkpointing is enabled.")

  checkpoint_sol = if checkpointing
    intervals = map(tuple, @view(checkpoints[1:end-1]), @view(checkpoints[2:end]))
    interval_end = intervals[end][end]
    tspan[1] > interval_end && push!(intervals, (interval_end, tspan[1]))
    cursor = lastindex(intervals)
    interval = intervals[cursor]
    cpsol = solve(remake(sol.prob, tspan=interval, u0=sol(interval[1])), sol.alg; tols...)
    CheckpointSolution(cpsol, intervals, cursor, tols)
  else
    nothing
  end

  diffcache, y = adjointdiffcache(g,sensealg,discrete,sol,dg,sol.prob.f)

  return ODEInterpolatingAdjointSensitivityFunction(diffcache,sensealg,
                                                    discrete,y,sol,
                                                    checkpoint_sol,sol.prob.f,)
end

function findcursor(intervals, t)
  # equivalent with `findfirst(x->x[1] <= t <= x[2], intervals)`
  lt(x, t) = <(x[2], t)
  return searchsortedfirst(intervals, t, lt=lt)
end

# u = λ'
# add tstop on all the checkpoints
function (S::ODEInterpolatingAdjointSensitivityFunction)(du,u,p,t)
  @unpack sol, y, checkpoint_sol, discrete = S
  idx = length(y)
  f = sol.prob.f

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
      prob′ = remake(sol.prob, tspan=intervals[cursor′], u0=y)
      cpsol′ = solve(prob′, sol.alg; dt=abs(cpsol_t[end] - cpsol_t[end-1]), checkpoint_sol.tols...)
      checkpoint_sol.cpsol = cpsol′
      checkpoint_sol.cursor = cursor′
    end
    checkpoint_sol.cpsol(y, t)
  end

  λ     = @view u[1:idx]
  grad  = @view u[idx+1:end]
  dλ    = @view du[1:idx]
  dgrad = @view du[idx+1:end]

  vecjacobian!(dλ, y, λ, p, t, S, dgrad=dgrad)

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

  p === DiffEqBase.NullParameters() && error("Your model does not have parameters, and thus it is impossible to calculate the derivative of the solution with respect to the parameters. Your model must have parameters to use parameter sensitivity calculations!")
  numstates = length(u0)
  numparams = length(p)

  len = numstates+numparams

  λ = similar(p, len)
  λ .= false
  sense = ODEInterpolatingAdjointSensitivityFunction(g,sensealg,discrete,sol,dg,
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
