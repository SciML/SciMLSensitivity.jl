struct NILSSSensitivityFunction{iip,F,A,J,JP,S,PJ,UF,PF,JC,PJC,Alg,fc,JM,pJM,MM,CV,
     PGPU,PGPP,CONFU,CONGP,DG} <: DiffEqBase.AbstractODEFunction{iip}
  f::F
  analytic::A
  jac::J
  jac_prototype::JP
  sparsity::S
  paramjac::PJ
  uf::UF
  pf::PF
  J::JM
  pJ::pJM
  jac_config::JC
  paramjac_config::PJC
  alg::Alg
  numparams::Int
  numindvar::Int
  f_cache::fc
  mass_matrix::MM
  colorvec::CV
  pgpu::PGPU
  pgpp::PGPP
  pgpu_config::CONFU
  pgpp_config::CONGP
  dg_val::DG
end

function NILSSSensitivityFunction(sensealg,f,analytic,jac,jac_prototype,sparsity,paramjac,u0,
                                    alg,p,f_cache,mm,
                                    colorvec,tspan,g,dg)

  uf = DiffEqBase.UJacobianWrapper(f,tspan[1],p)
  pf = DiffEqBase.ParamJacobianWrapper(f,tspan[1],copy(u0))

  if DiffEqBase.has_jac(f)
    jac_config = nothing
  else
    jac_config = build_jac_config(sensealg,uf,u0)
  end

  if DiffEqBase.has_paramjac(f)
    paramjac_config = nothing
  else
    paramjac_config = build_param_jac_config(sensealg,pf,u0,p)
  end
  numparams = length(p)
  numindvar = length(u0)
  J = Matrix{eltype(u0)}(undef,numindvar,numindvar)
  pJ = Matrix{eltype(u0)}(undef,numindvar,numparams) # number of funcs size

  # compute gradients of objective
  if dg != nothing
    pgpu = nothing
    pgpp = nothing
    pgpu_config = nothing
    pgpp_config = nothing
    if dg isa Tuple && length(dg) == 2
      dg_val = (similar(u0, numindvar),similar(u0, numparams))
      dg_val[1] .= false
      dg_val[2] .= false
    else
      dg_val = similar(u0, numindvar) # number of funcs size
      dg_val .= false
    end
  else
    pgpu = UGradientWrapper(g,tspan[1],p) # ∂g∂u
    pgpp = ParamGradientWrapper(g,tspan[1],u0) #∂g∂p
    pgpu_config = build_grad_config(sensealg,pgpu,u0,tspan[1])
    pgpp_config = build_grad_config(sensealg,pgpp,p,tspan[1])
    dg_val = (similar(u0, numindvar),similar(u0, numparams))
    dg_val[1] .= false
    dg_val[2] .= false
  end

  NILSSSensitivityFunction{isinplace(f),typeof(f),typeof(analytic),
                             typeof(jac),typeof(jac_prototype),typeof(sparsity),
                             typeof(paramjac),
                             typeof(uf),
                             typeof(pf),typeof(jac_config),
                             typeof(paramjac_config),typeof(alg),
                             typeof(f_cache),
                             typeof(J),typeof(pJ),typeof(mm),typeof(f.colorvec),
                             typeof(pgpu),typeof(pgpp),typeof(pgpu_config),typeof(pgpp_config),typeof(dg_val)}(
                             f,analytic,jac,jac_prototype,sparsity,paramjac,uf,pf,J,pJ,
                             jac_config,paramjac_config,alg,
                             numparams,numindvar,f_cache,mm,colorvec,
                             pgpu,pgpp,pgpu_config,pgpp_config,dg_val)
end


struct NILSSProblem{A,C,solType,dtType,umidType,dudtType,SType,Ftype,bType,ηType,wType,vType,windowType,
    ΔtType,G0,G,DG,resType}
  sensealg::A
  diffcache::C
  sol::solType
  dt::dtType
  umid::umidType
  dudt::dudtType
  S::SType
  F::Ftype
  b::bType
  η::ηType
  w::wType
  v::vType
  window::windowType
  Δt::ΔtType
  Nt::Int
  g0::G0
  g::G
  dg::DG
  res::resType
end


function NILSSProblem(sol, sensealg::ForwardLSS, g, dg = nothing;
                            kwargs...)

  @unpack f, p, u0, tspan = sol.prob
  isinplace = DiffEqBase.isinplace(f)

  p == nothing && error("You must have parameters to use parameter sensitivity calculations!")
  !(sol.u isa AbstractVector) && error("`u` has to be an AbstractVector.")


  sense = NILSSSensitivityFunction(sensealg,f,f.analytic,f.jac,
                                     f.jac_prototype,f.sparsity,f.paramjac,
                                     u0,sensealg,
                                     p,similar(u0),f.mass_matrix,
                                     f.colorvec,
                                     tspan,g,dg)

  @unpack numparams, numindvar = sense
  Nt = length(sol.t)
  Ndt = Nt-one(Nt)

  # pre-allocate variables
  dt = similar(sol.t, Ndt)
  umid = Matrix{eltype(u0)}(undef,numindvar,Ndt)
  dudt = Matrix{eltype(u0)}(undef,numindvar,Ndt)
  # compute their values
  discretize_ref_trajectory!(dt, umid, dudt, sol, Ndt)

  S = LSSSchur(dt,u0,numindvar,Nt,Ndt,sensealg.alpha)

  if sensealg.alpha isa Number
    η = similar(dt,Ndt)
    window = nothing
    g0 = g(u0,p,tspan[1])
  else
    η = nothing
    window = similar(dt,Nt)
    g0 = nothing
  end

  b = Matrix{eltype(u0)}(undef,numindvar*Ndt,numparams)
  w = similar(dt,numindvar*Ndt)
  v = similar(dt,numindvar*Nt)

  Δt = tspan[2] - tspan[1]
  wB!(S,Δt,Nt,numindvar,dt)
  wE!(S,Δt,dt,sensealg.alpha)

  B!(S,dt,umid,sense,sensealg)
  E!(S,dudt,sensealg.alpha)
  F = SchurLU(S)

  res = similar(u0, numparams)

  NILSSProblem{typeof(sensealg),typeof(sense),typeof(sol),typeof(dt),
    typeof(umid),typeof(dudt),
    typeof(S),typeof(F),typeof(b),typeof(η),typeof(w),typeof(v),typeof(window),typeof(Δt),
    typeof(g0),typeof(g),typeof(dg),typeof(res)}(sensealg,sense,sol,dt,umid,dudt,S,F,b,η,w,v,
    window,Δt,Nt,g0,g,dg,res)
end


function shadow_forward(prob::NILSSProblem; t0skip=zero(prob.Δt), t1skip=zero(prob.Δt))
  shadow_forward(prob,prob.sensealg,prob.sensealg.alpha,t0skip,t1skip)
end

function shadow_forward(prob::NILSSProblem,sensealg::ForwardLSS,alpha::Number,t0skip,t1skip)
  @unpack sol, S, F, window, Δt, diffcache, b, w, v, η, res, g, g0, dg, umid = prob
  @unpack wBinv, wEinv, B, E = S
  @unpack dg_val, pgpu, pgpu_config, pgpp, pgpp_config, numparams, numindvar, uf = diffcache


  return res
end
