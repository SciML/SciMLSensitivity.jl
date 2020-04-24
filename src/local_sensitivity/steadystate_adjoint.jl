struct SteadyStateAdjointSensitivityFunction{Alg<:SteadyStateAdjoint,UF,PF,G,TJ,PJT,GU,JC,GC,PJC,DG,uType,tmpT,LS} <: SensitivityFunction
  sensealg::Alg
  uf::UF
  pf::PF
  g::G
  J::TJ
  pJ::PJT
  gu::GU
  jac_config::JC
  g_grad_config::GC
  paramjac_config::PJC
  dg::DG
  u::uType
  linsolve_tmp::tmpT
  linsolve::LS
end

function SteadyStateAdjointSensitivityFunction(g,sensealg,sol,dg)
  func = sol.prob.f
  u = sol.u
  p = sol.prob.p
  u0 = sol.prob.u0
  numparams = length(p)
  numindvar = length(u0)

  uf = DiffEqBase.UJacobianWrapper(func,nothing,p)
  pf = DiffEqBase.ParamJacobianWrapper(func,nothing,copy(u))
  f_cache = DiffEqBase.isinplace(sol.prob) ? deepcopy(u0) : nothing

  J = Matrix{eltype(u0)}(undef,numindvar,numindvar) #df/du
  pJ = Matrix{eltype(u0)}(undef,numindvar,numparams) #df/dp
  gu = Matrix{eltype(u0)}(undef,1,numindvar) # dg/du

  if dg != nothing
    pg = nothing
    pg_config = nothing
    dg(vec(gu),u,p,nothing,nothing)
  else
    if g != nothing
      pg = UGradientWrapper(g,nothing,p)
      pg_config = build_grad_config(sensealg,pg,u0,p)
      gradient!(vec(gu),pg,u,sensealg,pg_config)
      #@show gu
    end
  end

  if DiffEqBase.has_jac(func)
    jac_config = nothing
    func.jac(J,u,p,nothing)
  else
    if DiffEqBase.isinplace(sol.prob)
      jac_config = build_jac_config(sensealg,uf,u0)
    else
      jac_config = nothing
    end
    jacobian!(J, uf, u, f_cache, sensealg, jac_config)
    #@show J
  end

  if DiffEqBase.has_paramjac(func)
    paramjac_config = nothing
    func.paramjac(pJ,u,p,nothing)
  else
    paramjac_config = build_param_jac_config(sensealg,pf,u0,p)
    jacobian!(pJ, pf, p, f_cache, sensealg, paramjac_config)
    #@show pJ
  end

  linsolve_tmp = zero(u0)
  linsolve = sensealg.linsolve(Val{:init},uf,u)

  SteadyStateAdjointSensitivityFunction(sensealg,uf,pf,g,J,pJ,gu,jac_config,pg_config,paramjac_config,dg,u,linsolve_tmp,linsolve)
end
