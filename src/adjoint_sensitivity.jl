struct ODEAdjointSensitvityFunction{F,AN,J,PJ,UF,PF,G,JC,GC,A,fc,SType,DG,uEltype,MM,TJ,PJT,PJC} <: SensitivityFunction
  f::F
  analytic::AN
  jac::J
  paramjac::PJ
  uf::UF
  pf::PF
  g::G
  J::TJ
  pJ::PJT
  dg_val::Adjoint{uEltype,Vector{uEltype}}
  jac_config::JC
  g_grad_config::GC
  paramjac_config::PJC
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

function ODEAdjointSensitvityFunction(f,analytic,jac,paramjac,uf,pf,g,u0,
                                      jac_config,g_grad_config,paramjac_config,
                                      p,f_cache,alg,discrete,y,sol,dg,mm)
  numparams = length(p)
  numindvar = length(u0)
  # if there is an analytical Jacobian provided, we are not going to do automatic `jac*vec`
  isautojacvec = DiffEqBase.has_jac(f) ? false : get_jacvec(alg)
  J = isautojacvec ? nothing : Matrix{eltype(u0)}(undef,numindvar,numindvar)
  pJ = if !isquad(alg)
    isautojacvec ? nothing : Matrix{eltype(sol.prob.u0)}(undef,length(sol.prob.u0),length(p))
  else
    nothing
  end
  dg_val = Vector{eltype(u0)}(undef,numindvar)' # number of funcs size
  ODEAdjointSensitvityFunction(f,analytic,jac,paramjac,uf,pf,g,J,pJ,dg_val,
                               jac_config,g_grad_config,paramjac_config,
                               alg,numparams,numindvar,f_cache,
                               discrete,y,sol,dg,mm)
end

# u = λ'
function (S::ODEAdjointSensitvityFunction)(du,u,p,t)
  idx = length(S.y)
  y = S.y
  if isbcksol(S.alg)
    λ     = Transpose(@view u[1:idx])
    dλ    = Transpose(@view du[1:idx])
    grad  = Transpose(@view u[idx+1:end-idx])
    dgrad = Transpose(@view du[idx+1:end-idx])
    _y    = @view u[end-idx+1:end]
    dy    = @view du[end-idx+1:end]
    S.sol.prob.f(dy, _y, p, t)
    copyto!(y, _y)
  else
    S.sol(y,t)
    if isquad(S.alg)
      λ     = u
      dλ    = du
    else
      λ     = Transpose(@view u[1:idx])
      dλ    = Transpose(@view du[1:idx])
      grad  = Transpose(@view u[idx+1:end])
      dgrad = Transpose(@view du[idx+1:end])
    end
  end

  isautojacvec = DiffEqBase.has_jac(S.f) ? false : get_jacvec(S.alg)
  if !isautojacvec
    if DiffEqBase.has_jac(S.f)
      S.f.jac(S.J,y,p,t) # Calculate the Jacobian into J
    else
      S.uf.t = t
      jacobian!(S.J, S.uf, y, S.f_cache, S.alg, S.jac_config)
    end
    mul!(dλ,λ,S.J)
  else
    tape = S.jac_config
    vecjacobian!(dλ, λ, tape, y)
  end

  dλ .*= -one(eltype(λ))

  if !S.discrete
    if S.dg != nothing
      S.dg(S.dg_val',y,p,t)
    else
      S.g.t = t
      gradient!(S.dg_val, S.g, y, S.f_cache, S.alg, S.g_gradient_config)
    end
    dλ .+= S.dg_val
  end

  if !isquad(S.alg)
    if !isautojacvec
      if DiffEqBase.has_paramjac(S.f)
        S.f.paramjac(S.pJ,y,S.sol.prob.p,t) # Calculate the parameter Jacobian into pJ
      else
        jacobian!(S.pJ, S.pf, S.sol.prob.p, S.f_cache, S.alg, S.paramjac_config)
      end
      mul!(dgrad,λ,S.pJ)
    else
      tape = S.paramjac_config
      vecjacobian!(dgrad, λ, tape, y, S.sol.prob.p)
    end
  end
  nothing
end

# g is either g(t,u,p) or discrete g(t,u,i)
function ODEAdjointProblem(sol,g,t=nothing,dg=nothing,
                           alg=SensitivityAlg();
                           callback=CallbackSet(),mass_matrix=I)

  f = sol.prob.f
  tspan = (sol.prob.tspan[2],sol.prob.tspan[1])
  t != nothing && (tspan = (t[end],t[1]))
  discrete = t != nothing

  isinplace = DiffEqBase.isinplace(sol.prob)
  p = sol.prob.p
  # if there is an analytical Jacobian provided, we are not going to do automatic `jac*vec`
  isautojacvec = DiffEqBase.has_jac(f) ? false : get_jacvec(alg)
  p == nothing && error("You must have parameters to use parameter sensitivity calculations!")
  uf = DiffEqDiffTools.UJacobianWrapper(f,tspan[2],p)
  pg = DiffEqDiffTools.UJacobianWrapper(g,tspan[2],p)

  u0 = zero(sol.prob.u0)'

  if DiffEqBase.has_jac(f)
    jac_config = nothing
  elseif isautojacvec
    jac_config = ReverseDiff.compile(ReverseDiff.GradientTape(uf, sol.prob.u0))
  else
    jac_config = build_jac_config(alg,uf,u0)
  end

  if !discrete
    if dg != nothing
      pg_config = nothing
    elseif isautojacvec
      pg_config = ReverseDiff.compile(ReverseDiff.GradientTape(pg, p))
    else
      pg_config = build_grad_config(alg,pg,u0,p)
    end
  else
    pg_config = nothing
  end

  y = copy(sol(tspan[1])) # TODO: Has to start at interpolation value!
  paramjac_config = nothing
  pf = nothing
  if !isquad(alg)
    if DiffEqBase.has_paramjac(f)
      paramjac_config = nothing
    elseif isautojacvec
      pf′ = VJacobianWrapper(f, tspan[1])
      paramjac_config = ReverseDiff.compile(ReverseDiff.GradientTape(pf′, (sol.prob.u0, p)))
    else
      pf = DiffEqDiffTools.ParamJacobianWrapper(f,tspan[1],y)
      paramjac_config = build_param_jac_config(alg,pf,y,p)
    end
  end

  len = isquad(alg) ? length(u0) : length(u0)+length(p)
  λ = similar(u0, len)'
  sense = ODEAdjointSensitvityFunction(f,nothing,f.jac,f.paramjac,
                                       uf,pf,pg,u0,jac_config,pg_config,paramjac_config,
                                       p,deepcopy(u0),alg,discrete,
                                       y,sol,dg,mass_matrix)

  if discrete
    cur_time = Ref(length(t))
    function time_choice(integrator)
      cur_time[] > 0 ? t[cur_time[]] : nothing
    end
    affect! = let isq = isquad(alg), λ=λ, t=t, y=y, cur_time=cur_time, idx=length(u0)
      function (integrator)
        p, u = integrator.p, integrator.u
        λ  = isq ? λ : Transpose(@view(λ[1:idx]))
        g(λ',y,p,t[cur_time[]],cur_time[])
        if isq
          u .+= λ
        else
          u = Transpose(@view u[1:idx])
          u .= λ .+ Transpose(@view integrator.u[1:idx])
        end
        u_modified!(integrator,true)
        cur_time[] -= 1
      end
    end
    cb = IterativeCallback(time_choice,affect!,eltype(tspan);initial_affect=true)

    _cb = CallbackSet(cb,callback)
  else
    _cb = callback
  end

  z0 = isbcksol(alg) ? [zero(λ'); conj(y)]' : zero(λ)
  ODEProblem(sense,z0,tspan,p,callback=_cb)
end

struct AdjointSensitivityIntegrand{S,AS,F,PF,PJC,uEltype,A,PJT}
  sol::S
  adj_sol::AS
  f::F
  p::Vector{uEltype}
  y::Vector{uEltype}
  λ::Adjoint{uEltype,Vector{uEltype}}
  pf::PF
  f_cache::Vector{uEltype}
  pJ::PJT
  paramjac_config::PJC
  alg::A
end

function AdjointSensitivityIntegrand(sol,adj_sol,alg=SensitivityAlg())
  f = sol.prob.f
  tspan = sol.prob.tspan
  p = sol.prob.p
  # we need to copy here, because later, we will call
  # `ReverseDiff.compile(ReverseDiff.GradientTape(pf′, (y, p)))`
  y = similar(sol.prob.u0)
  λ = similar(adj_sol.prob.u0)
  # we need to alias `y`
  pf = DiffEqDiffTools.ParamJacobianWrapper(f,tspan[1],y)
  f_cache = similar(y)
  isautojacvec = DiffEqBase.has_paramjac(f) ? false : get_jacvec(alg)
  pJ = isautojacvec ? nothing : Matrix{eltype(sol.prob.u0)}(undef,length(sol.prob.u0),length(p))

  if DiffEqBase.has_paramjac(f)
    paramjac_config = nothing
  elseif isautojacvec
    pf′ = VJacobianWrapper(f, tspan[1])
    paramjac_config = ReverseDiff.compile(ReverseDiff.GradientTape(pf′, (sol.prob.u0, p)))
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
  isautojacvec = DiffEqBase.has_paramjac(S.f) ? false : get_jacvec(S.alg)
  # y is aliased
  S.pf.t = t

  if !isautojacvec
    if DiffEqBase.has_paramjac(S.f)
      S.f.paramjac(S.pJ,y,S.p,t) # Calculate the parameter Jacobian into pJ
    else
      jacobian!(S.pJ, S.pf, S.p, S.f_cache, S.alg, S.paramjac_config)
    end
    mul!(out',λ,S.pJ)
  else
    tape = S.paramjac_config
    vecjacobian!(out, λ, tape, y, S.p)
  end
  out'
end

function (S::AdjointSensitivityIntegrand)(t)
  out = similar(S.p)
  S(out,t)
end


function adjoint_sensitivities(sol,alg,g,t=nothing,dg=nothing;
                               abstol=1e-6,reltol=1e-3,
                               iabstol=abstol, ireltol=reltol,sensealg=SensitivityAlg(),
                               kwargs...)

  adj_prob = ODEAdjointProblem(sol,g,t,dg,sensealg)
  isq = isquad(sensealg)
  adj_sol = solve(adj_prob,alg;abstol=abstol,reltol=reltol,save_everystep=isq,kwargs...)
  !isq && return adj_sol[end][(1:length(sol.prob.p)) .+ length(sol.prob.u0)]'
  integrand = AdjointSensitivityIntegrand(sol,adj_sol)

  if t == nothing
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
