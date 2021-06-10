struct LSSSchur{wBType,wEType,BType,EType,SType}
  wBinv::wBType
  wEinv::wEType
  B::BType
  E::EType
  Smat::SType
end

struct LSSSensitivityFunction{iip,F,A,J,JP,S,PJ,UF,PF,JC,PJC,Alg,fc,JM,pJM,MM,CV,
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

function LSSSensitivityFunction(sensealg,f,analytic,jac,jac_prototype,sparsity,paramjac,u0,
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



  LSSSensitivityFunction{isinplace(f),typeof(f),typeof(analytic),
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


struct ForwardLSSProblem{A,C,solType,dtType,umidType,dudtType,SType,bType,ηType,wType,vType,windowType,
    ΔtType,G0,G,resType}
  sensealg::A
  diffcache::C
  sol::solType
  dt::dtType
  umid::umidType
  dudt::dudtType
  S::SType
  b::bType
  η::ηType
  w::wType
  v::vType
  window::windowType
  Δt::ΔtType
  Nt::Int
  g0::G0
  g::G
  res::resType
end


function ForwardLSSProblem(sol, sensealg::ForwardLSS, g, dg = nothing;
                            kwargs...)

  @unpack f, p, u0, tspan = sol.prob
  isinplace = DiffEqBase.isinplace(f)

  p == nothing && error("You must have parameters to use parameter sensitivity calculations!")


  sense = LSSSensitivityFunction(sensealg,f,f.analytic,f.jac,
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
  S!(S)

  res = similar(u0, numparams)

  ForwardLSSProblem{typeof(sensealg),typeof(sense),typeof(sol),typeof(dt),
    typeof(umid),typeof(dudt),
    typeof(S),typeof(b),typeof(η),typeof(w),typeof(v),typeof(window),typeof(Δt),
    typeof(g),typeof(g0),typeof(res)}(sensealg,sense,sol,dt,umid,dudt,S,b,η,w,v,window,Δt,Nt,g,g0,
    res)
end

function LSSSchur(dt,u0,numindvar,Nt,Ndt,alpha)
  wBinv = similar(dt,numindvar*Nt)
  if alpha isa Number
    wEinv = similar(dt,Ndt)
    E = Matrix{eltype(u0)}(undef,numindvar*Ndt,Ndt)
  else
    wEinv = nothing
    E = nothing
  end
  B = Matrix{eltype(u0)}(undef,numindvar*Ndt,numindvar*Nt)
  Smat = Matrix{eltype(u0)}(undef,numindvar*Ndt,numindvar*Ndt)

  LSSSchur(wBinv,wEinv,B,E,Smat)
end


# compute discretized reference trajectory
function discretize_ref_trajectory!(dt, umid, dudt, sol, Ndt)
  for i=1:Ndt
    tr = sol.t[i+1]
    tl = sol.t[i]
    ur = sol.u[i+1]
    ul = sol.u[i]
    dt[i] = tr-tl
    copyto!((@view umid[:,i]), (ur + ul)/2)
    copyto!((@view dudt[:,i]), (ur - ul)/dt[i])
  end
  return nothing
end

function wB!(S::LSSSchur,Δt,Nt,numindvar,dt)
  @unpack wBinv = S
  fill!(wBinv, one(Δt))
  dim = numindvar * Nt
  tmp = @view wBinv[1:numindvar]
  tmp ./= dt[1]
  tmp = @view wBinv[dim-2:end]
  tmp ./= dt[end]
  for indx = 2:Nt-1
    tmp = @view wBinv[(indx-1)*numindvar+1:indx*numindvar]
    tmp ./= (dt[indx]+dt[indx-1])
  end

  wBinv .*= 2*Δt
  return nothing
end

wE!(S::LSSSchur,Δt,dt,alpha::Union{CosWindowing,Cos2Windowing}) = nothing

function wE!(S::LSSSchur,Δt,dt,alpha)
  @unpack wEinv = S
  @. wEinv = Δt/(alpha^2*dt)
  return nothing
end

function B!(S::LSSSchur,dt,umid,sense,sensealg)
  @unpack B = S
  @unpack f,J,uf,numindvar,f_cache,jac_config = sense

  fill!(B, zero(eltype(J)))

  for (i,u) in enumerate(eachcol(umid))
    if DiffEqBase.has_jac(f)
      f.jac(J,u,uf.p,uf.t) # Calculate the Jacobian into J
    else
      jacobian!(J, uf, u, f_cache, sensealg, jac_config)
    end
    B0 = @view B[(i-1)*numindvar+1:i*numindvar,i*numindvar+1:(i+1)*numindvar]
    B1 =  @view B[(i-1)*numindvar+1:i*numindvar,(i-1)*numindvar+1:i*numindvar]
    B0 .+= I/dt[i] - J/2
    B1 .+= -I/dt[i] -J/2
  end
  return nothing
end

E!(S::LSSSchur,dudt,alpha::Union{CosWindowing,Cos2Windowing}) = nothing

function E!(S::LSSSchur,dudt,alpha)
  @unpack E = S
  numindvar, Ndt = size(dudt)
  for i=1:Ndt
    tmp = @view E[(i-1)*numindvar+1:i*numindvar,i]
    copyto!(tmp, (@view dudt[:,i]))
  end
  return nothing
end

# compute Schur
function S!(S::LSSSchur)
  @unpack B, E, wBinv, wEinv, Smat = S
  Smat .= B*Diagonal(wBinv)*B'
  (wEinv !== nothing) && (Smat .+= E*Diagonal(wEinv)*E')
  return nothing
end

function b!(b, prob::ForwardLSSProblem)
  @unpack diffcache, umid, sensealg = prob
  @unpack f, f_cache, pJ, pf, paramjac_config, uf, numindvar = diffcache

  for (i,u) in enumerate(eachcol(umid))
    if DiffEqBase.has_paramjac(f)
      f.paramjac(pJ, u, uf.p, pf.t)
    else
      pf.u = u
      jacobian!(pJ, pf, uf.p, f_cache, sensealg, paramjac_config)
    end
    tmp = @view b[(i-1)*numindvar+1:i*numindvar,:]
    copyto!(tmp, pJ)
  end
  return nothing
end

function solve(prob::ForwardLSSProblem; t0skip=zero(prob.Δt), t1skip=zero(prob.Δt))
  _solve(prob,prob.sensealg,prob.sensealg.alpha,t0skip,t1skip)
end

function _solve(prob::ForwardLSSProblem,sensealg::ForwardLSS,alpha::Number,t0skip,t1skip)
  @unpack sol, S, window, Δt, diffcache, b, w, v, η, res, g, g0 = prob
  @unpack wBinv, wEinv, B, E, Smat = S
  @unpack dg_val, pgpu, pgpu_config, pgpp, pgpp_config, numparams, numindvar, uf = diffcache


  n0 = searchsortedfirst(sol.t, sol.t[1]+t0skip)
  n1 = searchsortedfirst(sol.t, sol.t[end]-t1skip)

  b!(b,prob)

  ures = @view sol.u[n0:n1]
  umidres = @view umid[n0:n1-1]

  # reset
  res .*=false
  g0 *= false #running average

  for i=1:numparams
    bpar = @view b[:,i]
    w .= Smat\bpar
    v .= Diagonal(wBinv)*(B'*w)
    η .= Diagonal(we)*(E'*w)

    vres = @view v[n0:n1]
    ηres =  @view η[n0:n1-1]

    for (j,u) in enumerate(umidres)
      vtmp = @view vres[(j-1)*numindvar+1:j*numindvar]
      #  final gradient result for ith parameter
      if dg_val isa Tuple
        DiffEqSensitivity.gradient!(dg_val[1], pgpu, u, sensealg,pgpu_config)
        DiffEqSensitivity.gradient!(dg_val[2], pgpp, uf.p, sensealg,pgpp_config)
        res[i] += dot(dg_val[1],vtmp)
        res[i] += dg_val[2][i]
      else
        DiffEqSensitivity.gradient!(dg_val, pgpu, u, sensealg,pgpu_config)
        res[i] += dot(dg_val,vtmp)
      end

      # compute objective
      gtmp = g(u,uf.p,nothing)
      g0 += gtmp

      res[i] -= ηres[j]*gtmp
    end
    res[i] = res[i]/(n1-n0)
    res[i] += sum(ηres)*g0/(n1-n0)
  end
  return res
end

function _solve(prob::ForwardLSSProblem,sensealg::ForwardLSS,alpha::CosWindowing,t0skip,t1skip)
  @unpack sol, S, window, Δt, diffcache, b, w, v, res = prob
  @unpack wBinv, B, Smat = S
  @unpack dg_val, pgpu, pgpu_config, pgpp, pgpp_config, numparams, numindvar, uf = diffcache

  b!(b,prob)

  # windowing (cos)
  @. window = (sol.t-sol.t[1])*convert(eltype(Δt),2*pi/Δt)
  @. window = one(eltype(window)) - cos(window)
  window ./= sum(window)

  res .*= false

  for i=1:numparams
    bpar = @view b[:,i]
    w .= Smat\bpar
    v .= Diagonal(wBinv)*(B'*w)
    for (j,u) in enumerate(sol.u)
      vtmp = @view v[(j-1)*numindvar+1:j*numindvar]
      #  final gradient result for ith parameter
      if dg_val isa Tuple
        DiffEqSensitivity.gradient!(dg_val[1], pgpu, u, sensealg,pgpu_config)
        DiffEqSensitivity.gradient!(dg_val[2], pgpp, uf.p, sensealg,pgpp_config)
        res[i] += dot(dg_val[1],vtmp)*window[j]
        res[i] += dg_val[2][i]*window[j]
      else
        DiffEqSensitivity.gradient!(dg_val, pgpu, u, sensealg,pgpu_config)
        res[i] += dot(dg_val,vtmp)*window[j]
      end
    end
  end
  return res
end

function _solve(prob::ForwardLSSProblem,sensealg::ForwardLSS,alpha::Cos2Windowing,t0skip,t1skip)
    @unpack sol, S, window, Δt, diffcache, b, w, v, res = prob
    @unpack wBinv, B,  Smat = S
    @unpack dg_val, pgpu, pgpu_config, pgpp, pgpp_config, numparams, numindvar, uf = diffcache

    b!(b,prob)

    res .*= false

    # windowing cos2
    @. window = (sol.t-sol.t[1])*convert(eltype(Δt),2*pi/Δt)
    @. window = (one(eltype(window)) - cos(window))^2
    window ./= sum(window)

    for i=1:numparams
      bpar = @view b[:,i]
      w .= Smat\bpar
      v .= Diagonal(wBinv)*(B'*w)
      for (j,u) in enumerate(sol.u)
        vtmp = @view v[(j-1)*numindvar+1:j*numindvar]
        #  final gradient result for ith parameter
        if dg_val isa Tuple
          DiffEqSensitivity.gradient!(dg_val[1], pgpu, u, sensealg,pgpu_config)
          DiffEqSensitivity.gradient!(dg_val[2], pgpp, uf.p, sensealg,pgpp_config)
          res[i] += dot(dg_val[1],vtmp)*window[j]
          res[i] += dg_val[2][i]*window[j]
        else
          DiffEqSensitivity.gradient!(dg_val, pgpu, u, sensealg,pgpu_config)
          res[i] += dot(dg_val,vtmp)*window[j]
        end
      end
    end
    return res
end
