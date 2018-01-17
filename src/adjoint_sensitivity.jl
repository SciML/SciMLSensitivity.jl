struct ODEAdjointSensitvityFunction{F,UF,G,JC,GC,P,A,fc,SType,DG,uEltype} <: SensitivityFunction
  f::F
  uf::UF
  g::G
  J::Matrix{uEltype}
  dg_val::RowVector{uEltype,Vector{uEltype}}
  jac_config::JC
  g_grad_config::GC
  p::P
  alg::A
  numparams::Int
  numindvar::Int
  f_cache::fc
  discrete::Bool
  y::Vector{uEltype}
  sol::SType
  dg::DG
end

function ODEAdjointSensitvityFunction(f,uf,g,u0,jac_config,g_grad_config,p,
                                      f_cache,alg,discrete,y,sol,dg)
  numparams = length(p)
  numindvar = length(u0)
  J = Matrix{eltype(u0)}(numindvar,numindvar)
  dg_val = RowVector{eltype(u0)}(numindvar) # number of funcs size
  ODEAdjointSensitvityFunction(f,uf,g,J,dg_val,jac_config,g_grad_config,
                             p,alg,numparams,numindvar,f_cache,discrete,y,sol,dg)
end

# u = λ'
function (S::ODEAdjointSensitvityFunction)(t,u,du)
  y = S.y
  S.sol(y,t)

  if has_jac(S.f)
    S.f(Val{:jac},t,y,S.J) # Calculate the Jacobian into J
  else
    S.uf.t = t
    jacobian!(S.J, S.uf, y, S.f_cache, S.alg, S.jac_config)
  end

  A_mul_B!(du,u,S.J); du .*= -one(eltype(u))

  if !S.discrete
    if S.dg != nothing
      S.dg(S.dg_val',t,y,S.p)
    else
      S.g.t = t
      gradient!(S.dg_val, S.g, y, S.f_cache, S.alg, S.g_gradient_config)
    end
    du .+= S.dg_val
  end

end

type ODEAdjointProblem{uType,tType,isinplace,F,C,MM} <: AbstractODEProblem{uType,tType,isinplace}
  f::F
  u0::uType
  tspan::Tuple{tType,tType}
  indvars::Int
  callback::C
  mass_matrix::MM
end

# g is either g(t,u,p) or discrete g(t,u,i)
function ODEAdjointProblem(sol,g,t=nothing,dg=nothing,
                                      alg=SensitivityAlg();
                                      callback=CallbackSet(),mass_matrix=I)

  f = sol.prob.f
  tspan = (sol.prob.tspan[2],sol.prob.tspan[1])
  discrete = t != nothing

  isinplace = DiffEqBase.isinplace(sol.prob)
  indvars = length(sol.prob.u0)
  p = param_values(f)
  uf = DiffEqDiffTools.UJacobianWrapper(f,tspan[2])
  pg = DiffEqDiffTools.UJacobianWrapper(ParameterizedFunction(g,p),tspan[2])

  if has_jac(f)
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

  u0 = zeros(sol.prob.u0)'
  y = sol(tspan[1]) # TODO: Has to start at interpolation value!
  λ = similar(u0)
  sense = ODEAdjointSensitvityFunction(f,uf,pg,u0,jac_config,pg_config,p,
                                       λ,alg,discrete,
                                       y,sol,dg)

  if discrete
    cur_time = Ref(length(t)-1)
    function time_choice(integrator)
      cur_time[]-=1
      t[cur_time[]+1]
    end
    function affect!(integrator)
      g(λ',y,cur_time[])
      integrator.u .+= λ
      u_modified!(integrator,true)
    end
    cb = IterativeCallback(time_choice,affect!,eltype(tspan);initial_affect=true)

    _cb = CallbackSet(cb,callback)
  else
    _cb = callback
  end

  ODEAdjointProblem{typeof(u0),typeof(tspan[1]),
                             isinplace,typeof(sense),typeof(_cb),
                             typeof(mass_matrix)}(
                             sense,u0,tspan,indvars,_cb,mass_matrix)
end

struct AdjointSensitivityIntegrand{S,AS,F,PF,PJC,uEltype,A}
  sol::S
  adj_sol::AS
  f::F
  p::Vector{uEltype}
  y::Vector{uEltype}
  λ::RowVector{uEltype,Vector{uEltype}}
  pf::PF
  f_cache::Vector{uEltype}
  pJ::Matrix{uEltype}
  paramjac_config::PJC
  alg::A
end

function AdjointSensitivityIntegrand(sol,adj_sol,alg=SensitivityAlg())
  f = sol.prob.f
  tspan = sol.prob.tspan
  p = param_values(f)
  y = similar(sol.prob.u0)
  λ = similar(adj_sol.prob.u0)
  pf = DiffEqDiffTools.ParamJacobianWrapper(f,tspan[1],copy(y))
  f_cache = similar(y)
  pJ = Matrix{eltype(sol.prob.u0)}(length(sol.prob.u0),length(p))

  if has_paramjac(f)
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

  if has_paramjac(S.f)
    S.f(Val{:paramjac},t,y,S.p,S.pJ) # Calculate the parameter Jacobian into pJ
  else
    S.pf.t = t
    S.pf.u .= y
    jacobian!(S.pJ, S.pf, S.p, S.f_cache, S.alg, S.paramjac_config)
  end
  A_mul_B!(out',λ,S.pJ)
end

function (S::AdjointSensitivityIntegrand)(t)
  out = similar(S.p)
  S(out,t)
end
