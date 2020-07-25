struct ODEQuadratureAdjointSensitivityFunction{C<:AdjointDiffCache,Alg<:QuadratureAdjoint,
                                               uType,SType,fType<:DiffEqBase.AbstractDiffEqFunction} <: SensitivityFunction
  diffcache::C
  sensealg::Alg
  discrete::Bool
  y::uType
  sol::SType
  f::fType
end

function ODEQuadratureAdjointSensitivityFunction(g,sensealg,discrete,sol,dg)
  diffcache, y = adjointdiffcache(g,sensealg,discrete,sol,dg,sol.prob.f;quad=true)
  return ODEQuadratureAdjointSensitivityFunction(diffcache,sensealg,discrete,
                                                 y,sol,sol.prob.f)
end

# u = λ'
function (S::ODEQuadratureAdjointSensitivityFunction)(du,u,p,t)
  @unpack y, sol, discrete = S
  f = sol.prob.f
  if typeof(t) <: ForwardDiff.Dual && eltype(y) <: AbstractFloat
    y = sol(t)
  else
    sol(y,t)
  end
  λ  = u
  dλ = du

  vecjacobian!(dλ, y, λ, p, t, S)
  dλ .*= -one(eltype(λ))

  discrete || accumulate_cost!(dλ, y, p, t, S)
  return nothing
end

# g is either g(t,u,p) or discrete g(t,u,i)
@noinline function ODEAdjointProblem(sol,sensealg::QuadratureAdjoint,g,
                                     t=nothing,dg=nothing,
                                     callback=CallbackSet())
  @unpack f, p, u0, tspan = sol.prob
  tspan = reverse(tspan)
  discrete = t != nothing

  len = length(u0)
  λ = similar(u0, len)
  λ .= false
  sense = ODEQuadratureAdjointSensitivityFunction(g,sensealg,discrete,sol,dg)

  init_cb = t !== nothing && tspan[1] == t[end]
  z0 = vec(zero(λ))
  cb = generate_callbacks(sense, g, λ, t, callback, init_cb)

  jac_prototype = sol.prob.f.jac_prototype
  adjoint_jac_prototype = !sense.discrete || jac_prototype === nothing ? nothing : copy(jac_prototype')

  original_mm = sol.prob.f.mass_matrix
  if original_mm === I || original_mm === (I,I)
    odefun = ODEFunction(sense, jac_prototype=adjoint_jac_prototype)
  else
    odefun = ODEFunction(sense, mass_matrix=sol.prob.f.mass_matrix', jac_prototype=adjoint_jac_prototype)
  end
  return ODEProblem(odefun,z0,tspan,p,callback=cb)
end

struct AdjointSensitivityIntegrand{pType,uType,lType,rateType,S,AS,PF,PJC,PJT,DGP,G}
  sol::S
  adj_sol::AS
  p::pType
  y::uType
  λ::lType
  pf::PF
  f_cache::rateType
  pJ::PJT
  paramjac_config::PJC
  sensealg::QuadratureAdjoint
  dgdp_cache::DGP
  dgdp::G
end

function AdjointSensitivityIntegrand(sol,adj_sol,sensealg,dgdp=nothing)
  prob = sol.prob
  @unpack f, p, tspan, u0 = prob
  numparams = length(p)
  y = zero(sol.prob.u0)
  λ = zero(adj_sol.prob.u0)
  # we need to alias `y`
  pf = DiffEqBase.ParamJacobianWrapper(f,tspan[1],y)
  f_cache = zero(y)
  f_cache .= false
  isautojacvec = get_jacvec(sensealg)
  pJ = isautojacvec ? nothing : similar(u0,length(u0),numparams)
  dgdp_cache = dgdp === nothing ? nothing : zero(p)

  if DiffEqBase.has_paramjac(f) || sensealg.autojacvec isa ReverseDiffVJP || (sensealg.autojacvec isa Bool && sensealg.autojacvec)
    tape = if DiffEqBase.isinplace(prob)
      ReverseDiff.GradientTape((y, prob.p, [tspan[2]])) do u,p,t
        du1 = similar(p, size(u))
        du1 .= false
        f(du1,u,p,first(t))
        return vec(du1)
      end
    else
      ReverseDiff.GradientTape((y, prob.p, [tspan[2]])) do u,p,t
        vec(f(u,p,first(t)))
      end
    end
    if compile_tape(sensealg)
      paramjac_config = ReverseDiff.compile(tape)
    else
      paramjac_config = tape
    end
  elseif isautojacvec
    paramjac_config = nothing
  else
    paramjac_config = build_param_jac_config(sensealg,pf,y,p)
  end
  AdjointSensitivityIntegrand(sol,adj_sol,p,y,λ,pf,f_cache,pJ,paramjac_config,sensealg,dgdp_cache,dgdp)
end

function (S::AdjointSensitivityIntegrand)(out,t)
  @unpack y, λ, pJ, pf, p, f_cache, dgdp_cache, paramjac_config, sensealg, sol, adj_sol = S
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
  elseif sensealg.autojacvec isa Bool || sensealg.autojacvec isa ReverseDiffVJP
    tape = paramjac_config
    tu, tp, tt = ReverseDiff.input_hook(tape)
    output = ReverseDiff.output_hook(tape)
    ReverseDiff.unseed!(tu) # clear any "leftover" derivatives from previous calls
    ReverseDiff.unseed!(tp)
    ReverseDiff.unseed!(tt)
    ReverseDiff.value!(tu, y)
    ReverseDiff.value!(tp, p)
    ReverseDiff.value!(tt, [t])
    ReverseDiff.forward_pass!(tape)
    ReverseDiff.increment_deriv!(output, λ)
    ReverseDiff.reverse_pass!(tape)
    copyto!(vec(out), ReverseDiff.deriv(tp))
  elseif sensealg.autojacvec isa ZygoteVJP
    _dy, back = Zygote.pullback(p) do p
      vec(f(y, p, t))
    end
    tmp = back(λ)
    out[:] .= vec(tmp[1])
  end

  # TODO: Add tracker?

  if S.dgdp !== nothing
    S.dgdp(dgdp_cache, y, p, t)
    out .+= dgdp_cache
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
  dgdu, dgdp = dg isa Tuple ? dg : (dg, nothing)
  adj_prob = ODEAdjointProblem(sol,sensealg,g,t,dgdu)
  adj_sol = solve(adj_prob,alg;abstol=abstol,reltol=reltol,
                               save_everystep=true,save_start=true,kwargs...)

  p = sol.prob.p
  if p === nothing || p === DiffEqBase.NullParameters()
    return -adj_sol[end],nothing
  else
    integrand = AdjointSensitivityIntegrand(sol,adj_sol,sensealg,dgdp)

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
    return -adj_sol[end], res
  end
end
