struct ODEInterpolatingAdjointSensitivityFunction{C<:AdjointDiffCache,Alg<:InterpolatingAdjoint,
                                                  uType,SType,CPS,CV} <: SensitivityFunction
  diffcache::C
  sensealg::Alg
  discrete::Bool
  y::uType
  sol::SType
  checkpoint_sol::CPS
  colorvec::CV
end

mutable struct CheckpointSolution{S,I,T}
  cpsol::S # solution in a checkpoint interval
  intervals::I # checkpoint intervals
  cursor::Int # sol.prob.tspan = intervals[cursor]
  tols::T
end

function ODEInterpolatingAdjointSensitivityFunction(g,sensealg,discrete,sol,dg,checkpoints,colorvec,tols)
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

  diffcache, y = adjointdiffcache(g,sensealg,discrete,sol,dg)

  return ODEInterpolatingAdjointSensitivityFunction(diffcache,sensealg,
                                                    discrete,y,sol,
                                                    checkpoint_sol,colorvec)
end

function findcursor(intervals, t)
  # equivalent with `findfirst(x->x[1] <= t <= x[2], intervals)`
  lt(x, t) = <(x[2], t)
  return searchsortedfirst(intervals, t, lt=lt)
end

# u = λ'
# add tstop on all the checkpoints
function (S::ODEInterpolatingAdjointSensitivityFunction)(du,u,p,t)
  @unpack y, sol, checkpoint_sol, discrete = S
  idx = length(y)
  f = sol.prob.f

  if checkpoint_sol === nothing
    sol(y,t)
  else
    intervals = checkpoint_sol.intervals
    interval = intervals[checkpoint_sol.cursor]
    if !(interval[1] <= t <= interval[2])
      cursor′ = findcursor(intervals, t)
      interval = intervals[cursor′]
      cpsol_t = checkpoint_sol.cpsol.t
      sol(y, interval[1])
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

  vecjacobian!(dλ, λ, p, t, S, dgrad=dgrad)

  dλ .*= -one(eltype(λ))

  discrete || accumulate_dgdu!(dλ, y, p, t, S)
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
  λ = similar(u0, len)
  sense = ODEInterpolatingAdjointSensitivityFunction(g,sensealg,discrete,sol,dg,
                                                     checkpoints,f.colorvec,
                                                     (reltol=reltol,abstol=abstol))

  init_cb = t !== nothing && tspan[1] == t[end]
  cb = generate_callbacks(sense, g, λ, t, callback, init_cb)
  z0 = vec(zero(λ))
  original_mm = sol.prob.f.mass_matrix
  if original_mm === I
    mm = I
  else
    mm = zeros(len, len)
    copyto!(@view(mm[1:numstates, 1:numstates]), sol.prob.f.mass_matrix')
    copyto!(@view(mm[numstates+1:end, numstates+1:end]), I)
  end
  odefun = ODEFunction(sense, mass_matrix=mm)
  return ODEProblem(odefun,z0,tspan,p,callback=cb)
end
