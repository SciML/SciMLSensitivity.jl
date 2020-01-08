## Direct calls

function adjoint_sensitivities_u0(sol,args...;
                                  sensealg=InterpolatingAdjoint(),
                                  kwargs...)
  _adjoint_sensitivities_u0(sol,sensealg,args...;kwargs...)
end

function _adjoint_sensitivities_u0(sol,sensealg,alg,g,t=nothing,dg=nothing;
                                   abstol=1e-6,reltol=1e-3,
                                   checkpoints=sol.t,
                                   kwargs...)
  adj_prob = ODEAdjointProblem(sol,sensealg,g,t,dg,checkpoints=checkpoints,
                               abstol=abstol,reltol=reltol)
  tstops = ischeckpointing(sensealg, sol) ? checkpoints : similar(sol.t, 0)
  adj_sol = solve(adj_prob,alg;
                  save_everystep=false,save_start=false,saveat=eltype(sol[1])[],
                  tstops=tstops,abstol=abstol,reltol=reltol,kwargs...)

  l = length(sol.prob.p)
  -adj_sol[end][1:length(sol.prob.u0)],
    adj_sol[end][(1:l) .+ length(sol.prob.u0)]'
end

function adjoint_sensitivities(sol,args...;
                               sensealg=InterpolatingAdjoint(),
                               kwargs...)
  _adjoint_sensitivities(sol,sensealg,args...;
                         kwargs...)
end

function _adjoint_sensitivities(sol,sensealg,alg,g,t=nothing,dg=nothing;
                               abstol=1e-6,reltol=1e-3,
                               iabstol=abstol, ireltol=reltol,
                               checkpoints=sol.t,
                               kwargs...)
  adj_prob = ODEAdjointProblem(sol,sensealg,g,t,dg,checkpoints=checkpoints,
                               abstol=abstol,reltol=reltol)
  tstops = ischeckpointing(sensealg, sol) ? checkpoints : similar(sol.t, 0)
  adj_sol = solve(adj_prob,alg;abstol=abstol,reltol=reltol,
                  tstops=tstops,save_everystep=false,save_start=false,kwargs...)
  l = length(sol.prob.p) 
  adj_sol[end][(1:l) .+ length(sol.prob.u0)]'
end
