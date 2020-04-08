struct ODEQuadratureAdjointSensitivityFunction{C<:AdjointDiffCache,Alg<:QuadratureAdjoint,
                                               uType,SType,CV} <: SensitivityFunction
  diffcache::C
  sensealg::Alg
  discrete::Bool
  y::uType
  sol::SType
  colorvec::CV
end

function ODEQuadratureAdjointSensitivityFunction(g,sensealg,discrete,sol,dg,colorvec)
  diffcache, y = adjointdiffcache(g,sensealg,discrete,sol,dg;quad=true)

  return ODEQuadratureAdjointSensitivityFunction(diffcache,sensealg,discrete,
                                                 y,sol,colorvec)
end

# u = λ'
function (S::ODEQuadratureAdjointSensitivityFunction)(du,u,p,t)
  @unpack y, sol, discrete = S
  f = sol.prob.f
  sol(y,t)
  λ  = u
  dλ = du

  vecjacobian!(dλ, λ, p, t, S)
  dλ .*= -one(eltype(λ))

  discrete || accumulate_dgdu!(dλ, y, p, t, S)
  return nothing
end

# g is either g(t,u,p) or discrete g(t,u,i)
@noinline function ODEAdjointProblem(sol,sensealg::QuadratureAdjoint,g,
                                     t=nothing,dg=nothing,
                                     callback=CallbackSet())
  @unpack f, p, u0, tspan = sol.prob
  tspan = reverse(tspan)
  discrete = t != nothing

  p === DiffEqBase.NullParameters() && error("Your model does not have parameters, and thus it is impossible to calculate the derivative of the solution with respect to the parameters. Your model must have parameters to use parameter sensitivity calculations!")

  len = length(u0)
  λ = similar(u0, len)
  sense = ODEQuadratureAdjointSensitivityFunction(g,sensealg,discrete,sol,dg,f.colorvec)

  init_cb = t !== nothing && tspan[1] == t[end]
  z0 = vec(zero(λ))
  cb = generate_callbacks(sense, g, λ, t, callback, init_cb)
  odefun = ODEFunction(sense, mass_matrix=sol.prob.f.mass_matrix')
  return ODEProblem(odefun,z0,tspan,p,callback=cb)
end

struct AdjointSensitivityIntegrand{pType,uType,rateType,S,AS,PF,PJC,PJT}
  sol::S
  adj_sol::AS
  p::pType
  y::uType
  λ::uType
  pf::PF
  f_cache::rateType
  pJ::PJT
  paramjac_config::PJC
  sensealg::QuadratureAdjoint
end

function AdjointSensitivityIntegrand(sol,adj_sol,sensealg)
  prob = sol.prob
  @unpack f, p, tspan, u0 = prob
  numparams = length(p)
  y = similar(sol.prob.u0)
  λ = similar(adj_sol.prob.u0)
  # we need to alias `y`
  pf = DiffEqBase.ParamJacobianWrapper(f,tspan[1],y)
  f_cache = similar(y)
  isautojacvec = get_jacvec(sensealg)
  pJ = isautojacvec ? nothing : similar(u0,length(u0),numparams)

  if DiffEqBase.has_paramjac(f) || isautojacvec
    paramjac_config = nothing
  else
    paramjac_config = build_param_jac_config(sensealg,pf,y,p)
  end
  AdjointSensitivityIntegrand(sol,adj_sol,p,y,λ,pf,f_cache,pJ,paramjac_config,sensealg)
end

function (S::AdjointSensitivityIntegrand)(out,t)
  @unpack y, λ, pJ, pf, p, f_cache, paramjac_config, sensealg, sol, adj_sol = S
  f = sol.prob.f
  sol(y,t)
  adj_sol(λ,t)
  λ .*= -one(eltype(λ))
  isautojacvec = get_jacvec(sensealg)
  # y is aliased

  if !isautojacvec
    if DiffEqBase.has_paramjac(f)
      f.paramjac(pJ,y,p,t) # Calculate the parameter Jacobian into pJ
    else
      pf.t = t
      jacobian!(pJ, pf, p, f_cache, sensealg, paramjac_config)
    end
    mul!(out',λ',pJ)
  else
    if DiffEqBase.isinplace(sol.prob)
      _, back = Tracker.forward(y,p) do u,p
        out_ = map(zero, u)
        f(out_, y, p, t)
        Tracker.collect(out_)
      end
      out[:] = vec(Tracker.data(back(λ)[2]))
    else
      _, back = Zygote.pullback(p) do p
        vec(f(y, p, t))
      end
      out[:] = vec(back(λ)[1])
    end
  end
  out'
end

function (S::AdjointSensitivityIntegrand)(t)
  out = similar(S.p)
  S(out,t)
end

function _adjoint_sensitivities(sol,sensealg::QuadratureAdjoint,alg,g,
                                t=nothing,dg=nothing;
                                abstol=1e-6,reltol=1e-3,
                                kwargs...)
  adj_prob = ODEAdjointProblem(sol,sensealg,g,t,dg)
  adj_sol = solve(adj_prob,alg;abstol=abstol,reltol=reltol,
                               save_everystep=true,save_start=true,kwargs...)
  integrand = AdjointSensitivityIntegrand(sol,adj_sol,sensealg)

  if t === nothing
    res,err = quadgk(integrand,sol.prob.tspan[1],sol.prob.tspan[2],
                     atol=sensealg.abstol,rtol=sensealg.reltol)
  else
    res = zero(integrand.p)'
    for i in 1:length(t)-1
      res .+= quadgk(integrand,t[i],t[i+1],
                     atol=sensealg.abstol,rtol=sensealg.reltol)[1]
    end
    if t[1] != sol.prob.tspan[1]
      res .+= quadgk(integrand,sol.prob.tspan[1],t[1],
                     atol=sensealg.abstol,rtol=sensealg.reltol)[1]
    end
  end
  -adj_sol[end], res
end
