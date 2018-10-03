struct ODEAdjointSensitvityFunction{F,AN,J,PJ,UF,G,JC,GC,A,fc,SType,DG,uEltype,MM} <: SensitivityFunction
  f::F
  analytic::AN
  jac::J
  paramjac::PJ
  uf::UF
  g::G
  J::Matrix{uEltype}
  dg_val::Adjoint{uEltype,Vector{uEltype}}
  jac_config::JC
  g_grad_config::GC
  alg::A
  numparams::Int
  numindvar::Int
  f_cache::fc
  discrete::Bool
  y::Vector{uEltype}
  sol::SType
  dg::DG
  mass_matrix::MM
end

function ODEAdjointSensitvityFunction(f,analytic,jac,paramjac,uf,g,u0,
                                      jac_config,g_grad_config,
                                      p,f_cache,alg,discrete,y,sol,dg,mm)
  numparams = length(p)
  numindvar = length(u0)
  J = Matrix{eltype(u0)}(undef,numindvar,numindvar)
  dg_val = Vector{eltype(u0)}(undef,numindvar)' # number of funcs size
  ODEAdjointSensitvityFunction(f,analytic,jac,paramjac,uf,g,J,dg_val,
                               jac_config,g_grad_config,
                               alg,numparams,numindvar,f_cache,
                               discrete,y,sol,dg,mm)
end

# u = λ'
function (S::ODEAdjointSensitvityFunction)(du,u,p,t)
  y = S.y
  S.sol(y,t)

  if DiffEqBase.has_jac(S.f)
    S.f.jac(S.J,y,p,t) # Calculate the Jacobian into J
  else
    S.uf.t = t
    jacobian!(S.J, S.uf, y, S.f_cache, S.alg, S.jac_config)
  end

  mul!(du,u,S.J); du .*= -one(eltype(u))

  if !S.discrete
    if S.dg != nothing
      S.dg(S.dg_val',y,p,t)
    else
      S.g.t = t
      gradient!(S.dg_val, S.g, y, S.f_cache, S.alg, S.g_gradient_config)
    end
    du .+= S.dg_val
  end

end

# g is either g(t,u,p) or discrete g(t,u,i)
function ODEAdjointProblem(sol,g,t=nothing,dg=nothing,
                                      alg=SensitivityAlg();
                                      callback=CallbackSet(),mass_matrix=I)

  f = sol.prob.f
  tspan = (sol.prob.tspan[2],sol.prob.tspan[1])
  discrete = t != nothing

  isinplace = DiffEqBase.isinplace(sol.prob)
  p = sol.prob.p
  uf = DiffEqDiffTools.UJacobianWrapper(f,tspan[2],p)
  pg = DiffEqDiffTools.UJacobianWrapper(g,tspan[2],p)

  u0 = zero(sol.prob.u0)'

  if DiffEqBase.has_jac(f)
    jac_config = nothing
  else
    jac_config = build_jac_config(alg,uf,u0)
  end

  if !discrete
    if dg != nothing
      pg_config = nothing
    else
      pg_config = build_grad_config(alg,pg,u0,p)
    end
  else
    pg_config = nothing
  end

  y = copy(sol(tspan[1])) # TODO: Has to start at interpolation value!
  λ = similar(u0)
  sense = ODEAdjointSensitvityFunction(f,nothing,f.jac,f.paramjac,
                                       uf,pg,u0,jac_config,pg_config,
                                       λ,deepcopy(u0),alg,discrete,
                                       y,sol,dg,mass_matrix)

  if discrete
    cur_time = Ref(length(t)-1)
    function time_choice(integrator)
      cur_time[]-=1
      t[cur_time[]+1]
    end
    function affect!(integrator)
      g(λ',y,p,t[cur_time[]],cur_time[])
      integrator.u .+= λ
      u_modified!(integrator,true)
    end
    cb = IterativeCallback(time_choice,affect!,eltype(tspan);initial_affect=true)

    _cb = CallbackSet(cb,callback)
  else
    _cb = callback
  end

  ODEProblem(sense,u0,tspan,p,callback=_cb)
end

struct AdjointSensitivityIntegrand{S,AS,F,PF,PJC,uEltype,A}
  sol::S
  adj_sol::AS
  f::F
  p::Vector{uEltype}
  y::Vector{uEltype}
  λ::Adjoint{uEltype,Vector{uEltype}}
  pf::PF
  f_cache::Vector{uEltype}
  pJ::Matrix{uEltype}
  paramjac_config::PJC
  alg::A
end

function AdjointSensitivityIntegrand(sol,adj_sol,alg=SensitivityAlg())
  f = sol.prob.f
  tspan = sol.prob.tspan
  p = sol.prob.p
  y = similar(sol.prob.u0)
  λ = similar(adj_sol.prob.u0)
  pf = DiffEqDiffTools.ParamJacobianWrapper(f,tspan[1],copy(y))
  f_cache = similar(y)
  pJ = Matrix{eltype(sol.prob.u0)}(undef,length(sol.prob.u0),length(p))

  if DiffEqBase.has_paramjac(f)
    paramjac_config = nothing
  else
    paramjac_config = build_param_jac_config(alg,pf,y,p)
  end
  AdjointSensitivityIntegrand(sol,adj_sol,f,p,y,λ,pf,f_cache,pJ,paramjac_config,alg)
end

function (S::AdjointSensitivityIntegrand)(out,t)
  y = S.y
  S.sol(y,t)
  λ = S.λ
  S.adj_sol(λ,t)
  λ .*= -one(eltype(λ))

  if DiffEqBase.has_paramjac(S.f)
    S.f.paramjac(S.pJ,y,S.p,t) # Calculate the parameter Jacobian into pJ
  else
    S.pf.t = t
    S.pf.u .= y
    jacobian!(S.pJ, S.pf, S.p, S.f_cache, S.alg, S.paramjac_config)
  end
  mul!(out',λ,S.pJ)
end

function (S::AdjointSensitivityIntegrand)(t)
  out = similar(S.p)
  S(out,t)
end


function adjoint_sensitivities(sol,alg,g,t=nothing,dg=nothing;
                               abstol=1e-6,reltol=1e-3,
                               iabstol = abstol, ireltol=reltol,
                               kwargs...)

  adj_prob = ODEAdjointProblem(sol,g,t,dg)
  adj_sol = solve(adj_prob,alg;abstol=abstol,reltol=reltol,kwargs...)
  integrand = AdjointSensitivityIntegrand(sol,adj_sol)

  if t== nothing
    res,err = quadgk(integrand,sol.prob.tspan[1],sol.prob.tspan[2],
                   atol=iabstol,rtol=ireltol)
  else
    res = zero(integrand.p)'
    for i in 1:length(t)-1
      res .+= quadgk(integrand,t[i],t[i+1],
                     atol=iabstol,rtol=ireltol)[1]
    end
  end
  res
end
