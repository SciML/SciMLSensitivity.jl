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
  tstops = checkpoints === nothing ? similar(sol.t, 0) : checkpoints
  adj_sol = solve(adj_prob,alg;
                  save_everystep=false,save_start=false,saveat=eltype(sol[1])[],
                  tstops=tstops,abstol=abstol,reltol=reltol,kwargs...)

  l = sol.prob.p isa Zygote.Params ? sum(length.(sol.prob.p)) : length(sol.prob.p)
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
  tstops = checkpoints === nothing ? similar(sol.t, 0) : checkpoints
  adj_sol = solve(adj_prob,alg;abstol=abstol,reltol=reltol,
                  tstops=tstops,save_everystep=false,save_start=false,kwargs...)
  l = sol.prob.p isa Zygote.Params ? sum(length.(sol.prob.p)) : length(sol.prob.p)
  adj_sol[end][(1:l) .+ length(sol.prob.u0)]'
end

### Common utils

function generate_callbacks(sensefun, g, λ, t, callback, init_cb)
  if sensefun.discrete
    @unpack sensealg, y, sol = sensefun
    prob = sol.prob
    cur_time = Ref(length(t))
    function time_choice(integrator)
      cur_time[] > 0 ? t[cur_time[]] : nothing
    end
    affect! = let isq = (sensealg isa QuadratureAdjoint), λ=λ, t=t, y=y, cur_time=cur_time, idx=length(prob.u0)
      function (integrator)
        p, u = integrator.p, integrator.u
        λ  = isq ? λ : @view(λ[1:idx])
        g(λ,y,p,t[cur_time[]],cur_time[])
        if isq
          u .+= λ
        else
          u = @view u[1:idx]
          u .= λ .+ @view integrator.u[1:idx]
        end
        u_modified!(integrator,true)
        cur_time[] -= 1
      end
    end
    cb = IterativeCallback(time_choice,affect!,eltype(prob.tspan);initial_affect=init_cb)

    _cb = CallbackSet(cb,callback)
  else
    _cb = callback
  end
  return _cb
end
