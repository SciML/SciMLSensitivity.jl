struct LSSSchur{wBType,wEType,BType,EType}
  wBinv::wBType
  wEinv::wEType
  B::BType
  E::EType
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
  if dg !== nothing
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


struct ForwardLSSProblem{A,C,solType,dtType,umidType,dudtType,SType,Ftype,bType,ηType,wType,vType,windowType,
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


function ForwardLSSProblem(sol, sensealg::ForwardLSS, t=nothing, dg = nothing;
                            kwargs...)

  @unpack f, p, u0, tspan = sol.prob
  @unpack g = sensealg

  isinplace = DiffEqBase.isinplace(f)

  # some shadowing sensealgs require knowledge of g
  check_for_g(sensealg,g)

  p === nothing && error("You must have parameters to use parameter sensitivity calculations!")
  !(sol.u isa AbstractVector) && error("`u` has to be an AbstractVector.")

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

  # assert that all ts are hit if concrete solve interface/discrete costs are used
  if t !== nothing
    @assert sol.t == t
  end

  S = LSSSchur(dt,u0,numindvar,Nt,Ndt,sensealg.LSSregularizer)

  if sensealg.LSSregularizer isa TimeDilation
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
  wE!(S,Δt,dt,sensealg.LSSregularizer)
  B!(S,dt,umid,sense,sensealg)
  E!(S,dudt,sensealg.LSSregularizer)

  F = SchurLU(S)

  res = similar(u0, numparams)

  ForwardLSSProblem{typeof(sensealg),typeof(sense),typeof(sol),typeof(dt),
    typeof(umid),typeof(dudt),
    typeof(S),typeof(F),typeof(b),typeof(η),typeof(w),typeof(v),typeof(window),typeof(Δt),
    typeof(g0),typeof(g),typeof(dg),typeof(res)}(sensealg,sense,sol,dt,umid,dudt,S,F,b,η,w,v,
    window,Δt,Nt,g0,g,dg,res)
end

function LSSSchur(dt,u0,numindvar,Nt,Ndt,LSSregularizer::TimeDilation)
  wBinv = similar(dt,numindvar*Nt)
  wEinv = similar(dt,Ndt)
  E = Matrix{eltype(u0)}(undef,numindvar*Ndt,Ndt)
  B = Matrix{eltype(u0)}(undef,numindvar*Ndt,numindvar*Nt)

  LSSSchur(wBinv,wEinv,B,E)
end

function LSSSchur(dt,u0,numindvar,Nt,Ndt,LSSregularizer::AbstractCosWindowing)
  wBinv = similar(dt,numindvar*Nt)
  wEinv = nothing
  E = nothing
  B = Matrix{eltype(u0)}(undef,numindvar*Ndt,numindvar*Nt)

  LSSSchur(wBinv,wEinv,B,E)
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

wE!(S::LSSSchur,Δt,dt,LSSregularizer::AbstractCosWindowing) = nothing

function wE!(S::LSSSchur,Δt,dt,LSSregularizer::TimeDilation)
  @unpack wEinv = S
  @unpack alpha = LSSregularizer
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
    B1 = @view B[(i-1)*numindvar+1:i*numindvar,(i-1)*numindvar+1:i*numindvar]
    B0 .+= I/dt[i] - J/2
    B1 .+= -I/dt[i] -J/2
  end
  return nothing
end

E!(S::LSSSchur,dudt,LSSregularizer::AbstractCosWindowing) = nothing

function E!(S::LSSSchur,dudt,LSSregularizer::TimeDilation)
  @unpack E = S
  numindvar, Ndt = size(dudt)
  for i=1:Ndt
    tmp = @view E[(i-1)*numindvar+1:i*numindvar,i]
    copyto!(tmp, (@view dudt[:,i]))
  end
  return nothing
end

# compute Schur
function SchurLU(S::LSSSchur)
  @unpack B, E, wBinv, wEinv = S
  Smat = B*Diagonal(wBinv)*B'
  (wEinv !== nothing) && (Smat .+= E*Diagonal(wEinv)*E')
  F = lu!(Smat)
  return F
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

function shadow_forward(prob::ForwardLSSProblem; sensealg=prob.sensealg)
  shadow_forward(prob,sensealg,sensealg.LSSregularizer)
end

function shadow_forward(prob::ForwardLSSProblem,sensealg::ForwardLSS,LSSregularizer::TimeDilation)
  @unpack sol, S, F, window, Δt, diffcache, b, w, v, η, res, g, g0, dg, umid = prob
  @unpack wBinv, wEinv, B, E = S
  @unpack dg_val, numparams, numindvar, uf = diffcache
  @unpack t0skip, t1skip = LSSregularizer

  n0 = searchsortedfirst(sol.t, sol.t[1]+t0skip)
  n1 = searchsortedfirst(sol.t, sol.t[end]-t1skip)

  b!(b,prob)

  ures = @view sol.u[n0:n1]
  umidres = @view umid[:,n0:n1-1]

  # reset
  res .*=false

  for i=1:numparams
    #running average
    g0 *= false
    bpar = @view b[:,i]
    w .= F\bpar
    v .= Diagonal(wBinv)*(B'*w)
    η .= Diagonal(wEinv)*(E'*w)

    ηres = @view η[n0:n1-1]

    for (j, u) in enumerate(ures)
      vtmp = @view v[(n0+j-2)*numindvar+1:(n0+j-1)*numindvar]
      #  final gradient result for ith parameter
      accumulate_cost!(dg, u, uf.p, uf.t, sensealg, diffcache, n0+j-1)

      if dg_val isa Tuple
        res[i] += dot(dg_val[1], vtmp)
        res[i] += dg_val[2][i]
      else
        res[i] += dot(dg_val, vtmp)
      end

    end
    # mean value
    res[i] = res[i]/(n1-n0+1)

    for (j,u) in enumerate(eachcol(umidres))
      # compute objective
      gtmp = g(u,uf.p,nothing)
      g0 += gtmp
      res[i] -= ηres[j]*gtmp/(n1-n0)
    end
    res[i] = res[i] + sum(ηres)*g0/(n1-n0)^2

  end
  return res
end

function shadow_forward(prob::ForwardLSSProblem,sensealg::ForwardLSS,LSSregularizer::CosWindowing)
  @unpack sol, S, F, window, Δt, diffcache, b, w, v, dg, res = prob
  @unpack wBinv, B = S
  @unpack dg_val, numparams, numindvar, uf = diffcache

  b!(b,prob)

  # windowing (cos)
  @. window = (sol.t-sol.t[1])*convert(eltype(Δt),2*pi/Δt)
  @. window = one(eltype(window)) - cos(window)
  window ./= sum(window)

  res .*= false

  for i=1:numparams
    bpar = @view b[:,i]
    w .= F\bpar
    v .= Diagonal(wBinv)*(B'*w)
    for (j,u) in enumerate(sol.u)
      vtmp = @view v[(j-1)*numindvar+1:j*numindvar]
      #  final gradient result for ith parameter
      accumulate_cost!(dg, u, uf.p, uf.t, sensealg, diffcache, j)
      if dg_val isa Tuple
        res[i] += dot(dg_val[1], vtmp) * window[j]
        res[i] += dg_val[2][i] * window[j]
      else
        res[i] += dot(dg_val, vtmp) * window[j]
      end
    end
  end
  return res
end

function shadow_forward(prob::ForwardLSSProblem,sensealg::ForwardLSS,LSSregularizer::Cos2Windowing)
    @unpack sol, S, F, window, Δt, diffcache, b, w, v, dg, res = prob
    @unpack wBinv, B = S
    @unpack dg_val, numparams, numindvar, uf = diffcache

    b!(b,prob)

    res .*= false

    # windowing cos2
    @. window = (sol.t-sol.t[1])*convert(eltype(Δt),2*pi/Δt)
    @. window = (one(eltype(window)) - cos(window))^2
    window ./= sum(window)

    for i=1:numparams
      bpar = @view b[:,i]
      w .= F\bpar
      v .= Diagonal(wBinv)*(B'*w)
      for (j, u) in enumerate(sol.u)
        vtmp = @view v[(j-1)*numindvar+1:j*numindvar]
        #  final gradient result for ith parameter
        accumulate_cost!(dg, u, uf.p, uf.t, sensealg, diffcache, j)
        if dg_val isa Tuple
          res[i] += dot(dg_val[1], vtmp) * window[j]
          res[i] += dg_val[2][i] * window[j]
        else
          res[i] += dot(dg_val, vtmp) * window[j]
        end
      end
    end
    return res
end

function accumulate_cost!(dg, u, p, t, sensealg::ForwardLSS, diffcache, indx)
  @unpack dg_val, pgpu, pgpu_config, pgpp, pgpp_config, uf = diffcache

  if dg === nothing
    if dg_val isa Tuple
      DiffEqSensitivity.gradient!(dg_val[1], pgpu, u, sensealg, pgpu_config)
      DiffEqSensitivity.gradient!(dg_val[2], pgpp, p, sensealg, pgpp_config)
    else
      DiffEqSensitivity.gradient!(dg_val, pgpu, u, sensealg, pgpu_config)
    end
  else
    if dg_val isa Tuple
      dg[1](dg_val[1], u, p, nothing, indx) # indx = n0 + j - 1 for LSSregularizer and j for windowing
      dg[2](dg_val[2], u, p, nothing, indx)
    else
      dg(dg_val, u, p, nothing, indx)
    end
  end

  return nothing
end
struct AdjointLSSProblem{A,C,solType,dtType,umidType,dudtType,SType,FType,hType,bType,wType,
    ΔtType,G0,G,DG,resType}
  sensealg::A
  diffcache::C
  sol::solType
  dt::dtType
  umid::umidType
  dudt::dudtType
  S::SType
  F::FType
  h::hType
  b::bType
  wa::wType
  Δt::ΔtType
  Nt::Int
  g0::G0
  g::G
  dg::DG
  res::resType
end


function AdjointLSSProblem(sol, sensealg::AdjointLSS, t=nothing, dg = nothing;
                            kwargs...)

  @unpack f, p, u0, tspan = sol.prob
  @unpack g = sensealg

  isinplace = DiffEqBase.isinplace(f)

  # some shadowing sensealgs require knowledge of g
  check_for_g(sensealg,g)

  p === nothing && error("You must have parameters to use parameter sensitivity calculations!")
  !(sol.u isa AbstractVector) && error("`u` has to be an AbstractVector.")

  # assert that all ts are hit if concrete solve interface/discrete costs are used
  if t !== nothing
    @assert sol.t == t
  end

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

  S = LSSSchur(dt,u0,numindvar,Nt,Ndt,sensealg.LSSregularizer)

  if sensealg.LSSregularizer isa TimeDilation
    g0 = g(u0,p,tspan[1])
  else
    g0 = nothing
  end

  b = Vector{eltype(u0)}(undef,numindvar*Ndt)
  h = Vector{eltype(u0)}(undef,Ndt)
  wa = similar(dt,numindvar*Ndt)

  Δt = tspan[2] - tspan[1]
  wB!(S,Δt,Nt,numindvar,dt)
  wE!(S,Δt,dt,sensealg.LSSregularizer)

  B!(S,dt,umid,sense,sensealg)
  E!(S,dudt,sensealg.LSSregularizer)
  F = SchurLU(S)
  wBcorrect!(S,sol,g,Nt,sense,sensealg,dg)

  h!(h,g0,g,umid,p,S.wEinv)

  res = similar(u0, numparams)

  AdjointLSSProblem{typeof(sensealg),typeof(sense),typeof(sol),typeof(dt),
    typeof(umid),typeof(dudt),
    typeof(S),typeof(F),typeof(h),typeof(b),typeof(wa),typeof(Δt),
    typeof(g0),typeof(g),typeof(dg),typeof(res)}(sensealg,sense,sol,dt,umid,dudt,S,F,h,b,wa,
    Δt,Nt,g0,g,dg,res)
end

function h!(h,g0,g,u,p,wEinv)

  for (j,uj) in enumerate(eachcol(u))
    # compute objective
    h[j] = g(uj,p,nothing)
  end
  h .= -(h .- mean(h)) / (size(u)[2])

  @. h = wEinv*h

  return nothing
end

function wBcorrect!(S,sol,g,Nt,sense,sensealg,dg)
  @unpack dg_val, pgpu, pgpu_config, numparams, numindvar, uf = sense
  @unpack wBinv = S

  for (i,u) in enumerate(sol.u)
    _wBinv = @view wBinv[(i-1)*numindvar+1:i*numindvar]
    if dg === nothing
      if dg_val isa Tuple
        DiffEqSensitivity.gradient!(dg_val[1], pgpu, u, sensealg, pgpu_config)
        @. _wBinv = _wBinv*dg_val[1]/Nt
      else
        DiffEqSensitivity.gradient!(dg_val, pgpu, u, sensealg, pgpu_config)
        @. _wBinv = _wBinv*dg_val/Nt
      end
    else
      if dg_val isa Tuple
        dg[1](dg_val[1],u,uf.p,nothing,i)
        @. _wBinv = _wBinv*dg_val[1]/Nt
      else
        dg(dg_val,u,uf.p,nothing,i)
        @. _wBinv = _wBinv*dg_val/Nt
      end
    end
  end
  return nothing
end

function shadow_adjoint(prob::AdjointLSSProblem; sensealg=prob.sensealg)
  shadow_adjoint(prob,sensealg,sensealg.LSSregularizer)
end

function shadow_adjoint(prob::AdjointLSSProblem,sensealg::AdjointLSS,LSSregularizer::TimeDilation)
  @unpack sol, S, F, Δt, diffcache, h, b, wa, res, g, g0, dg, umid = prob
  @unpack wBinv, B, E = S
  @unpack dg_val, pgpp, pgpp_config, numparams, numindvar, uf, f, f_cache, pJ, pf, paramjac_config = diffcache
  @unpack t0skip, t1skip = LSSregularizer

  b .= E*h + B*wBinv
  wa .= F\b

  n0 = searchsortedfirst(sol.t, sol.t[1]+t0skip)
  n1 = searchsortedfirst(sol.t, sol.t[end]-t1skip)

  umidres = @view umid[:,n0:n1-1]
  wares = @view wa[(n0-1)*numindvar+1:(n1-1)*numindvar]

  # reset
  res .*= false

  if dg_val isa Tuple
    for (j,u) in enumerate(eachcol(umidres))
      if dg === nothing
        DiffEqSensitivity.gradient!(dg_val[2], pgpp, uf.p, sensealg, pgpp_config)
        @. res += dg_val[2]
      else
        dg[2](dg_val[2],u,uf.p,nothing,n0+j-1)
        @. res += dg_val[2]
      end
    end
    res ./= (size(umidres)[2])
  end

  for (j,u) in enumerate(eachcol(umidres))
    _wares = @view wares[(j-1)*numindvar+1:j*numindvar]
    if DiffEqBase.has_paramjac(f)
      f.paramjac(pJ, u, uf.p, pf.t)
    else
      pf.u = u
      jacobian!(pJ, pf, uf.p, f_cache, sensealg, paramjac_config)
    end

    res .+= pJ'*_wares
  end

  return res
end

check_for_g(sensealg::Union{ForwardLSS,AdjointLSS},g)=((sensealg.LSSregularizer isa TimeDilation && g===nothing) && error("Time dilation needs explicit knowledge of g. Either pass `g` as a kwarg to `ForwardLSS(g=g)` or `AdjointLSS(g=g)` or use ForwardLSS/AdjointLSS with windowing."))
