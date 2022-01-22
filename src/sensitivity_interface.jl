## Direct calls

function adjoint_sensitivities(sol,args...;
                                  sensealg=InterpolatingAdjoint(),
                                  kwargs...)

  if DiffEqBase.isinplace(sol.prob) && sensealg.autojacvec isa ZygoteVJP
    error("In-place differential equations are incompatible with ZygoteVJP since that requires mutation. Choose a different VJP.")
  end

  _adjoint_sensitivities(sol,sensealg,args...;kwargs...)
end

function _adjoint_sensitivities(sol,sensealg,alg,g,t=nothing,dg=nothing;
                                   abstol=1e-6,reltol=1e-3,
                                   checkpoints=sol.t,
                                   corfunc_analytical=nothing,
                                   callback = nothing,
                                   kwargs...)

  if sol.prob isa ODEProblem
    adj_prob = ODEAdjointProblem(sol,sensealg,g,t,dg,checkpoints=checkpoints,
                                 callback = callback,
                                 abstol=abstol,reltol=reltol)

  elseif sol.prob isa SDEProblem
    adj_prob = SDEAdjointProblem(sol,sensealg,g,t,dg,checkpoints=checkpoints,
                                 callback = callback,
                                 abstol=abstol,reltol=reltol,
                                 corfunc_analytical=corfunc_analytical)
  elseif sol.prob isa RODEProblem
    adj_prob = RODEAdjointProblem(sol,sensealg,g,t,dg,checkpoints=checkpoints,
                                callback = callback,
                                abstol=abstol,reltol=reltol,
                                corfunc_analytical=corfunc_analytical)
  else
    error("Continuous adjoint sensitivities are only supported for ODE/SDE/RODE problems.")
  end

  tstops = ischeckpointing(sensealg, sol) ? checkpoints : similar(sol.t, 0)
  adj_sol = solve(adj_prob,alg;
                  save_everystep=false,save_start=false,saveat=eltype(sol[1])[],
                  tstops=tstops,abstol=abstol,reltol=reltol,kwargs...)

  p = sol.prob.p
  l = p === nothing || p === DiffEqBase.NullParameters() ? 0 : length(sol.prob.p)
  du0 = -adj_sol[end][1:length(sol.prob.u0)]

  if eltype(sol.prob.p) <: real(eltype(adj_sol[end]))
    dp = real.(adj_sol[end][(1:l) .+ length(sol.prob.u0)])'
  elseif p === nothing || p === DiffEqBase.NullParameters()
    dp = nothing
  else
    dp = adj_sol[end][(1:l) .+ length(sol.prob.u0)]'
  end

  du0,dp
end

function _adjoint_sensitivities(sol,sensealg::SteadyStateAdjoint,alg,g,dg=nothing;
                                   abstol=1e-6,reltol=1e-3,
                                   kwargs...)
  SteadyStateAdjointProblem(sol,sensealg,g,dg;kwargs...)
end

function _adjoint_sensitivities(sol,sensealg::SteadyStateAdjoint,alg;
                                   g=nothing,dg=nothing,
                                   abstol=1e-6,reltol=1e-3,
                                   kwargs...)
  SteadyStateAdjointProblem(sol,sensealg,g,dg;kwargs...)
end

function second_order_sensitivities(loss,prob,alg,args...;
                                    sensealg=ForwardDiffOverAdjoint(InterpolatingAdjoint(autojacvec=ReverseDiffVJP())),
                                    kwargs...)
  _second_order_sensitivities(loss,prob,alg,sensealg,args...;kwargs...)
end

function second_order_sensitivity_product(loss,v,prob,alg,args...;
                                          sensealg=ForwardDiffOverAdjoint(InterpolatingAdjoint(autojacvec=ReverseDiffVJP())),
                                          kwargs...)
  _second_order_sensitivity_product(loss,v,prob,alg,sensealg,args...;kwargs...)
end
