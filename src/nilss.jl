struct NILSSSensitivityFunction{iip,F,Alg,
     PGPU,PGPP,CONFU,CONGP,DG} <: DiffEqBase.AbstractODEFunction{iip}
  f::F
  alg::Alg
  numparams::Int
  numindvar::Int
  pgpu::PGPU
  pgpp::PGPP
  pgpu_config::CONFU
  pgpp_config::CONGP
  dg_val::DG
end

function NILSSSensitivityFunction(sensealg,f,u0,alg,p,tspan,g,dg)

  numparams = length(p)
  numindvar = length(u0)

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

  NILSSSensitivityFunction{isinplace(f),typeof(f),typeof(alg),
                             typeof(pgpu),typeof(pgpp),typeof(pgpu_config),typeof(pgpp_config),typeof(dg_val)}(
                             f,alg,numparams,numindvar,pgpu,pgpp,pgpu_config,pgpp_config,dg_val)
end


struct NILSSProblem{A,CacheType,FSV,FSW,probType,TType,dtType,gType,yType,vstarType,
    wType,RType,bType,weightType,CType,dType,BType,aType,vType,ksiType,
    G,DG,resType}
  sensealg::A
  diffcache::CacheType
  forward_prob_v::FSV
  forward_prob_w::FSW
  prob::probType
  nus::Int
  T_seg::TType
  dtsave::dtType
  gsave::gType
  y::yType
  dudt::yType
  dgdu::yType
  vstar::vstarType
  vstar_perp::vstarType
  w::wType
  w_perp::wType
  R::RType
  b::bType
  weight::weightType
  Cinv::CType
  d::dType
  B::BType
  a::aType
  v::vType
  v_perp::vType
  ξ::ksiType
  g::G
  dg::DG
  res::resType
end


function NILSSProblem(prob, sensealg::NILSS, g, dg = nothing; nus = nothing,
                            kwargs...)

  @unpack f, p, u0, tspan = prob
  @unpack nseg, nstep, rng = sensealg  #number of segments on time interval, number of steps saved on each segment

  numindvar = length(u0)
  numparams = length(p)

  # integer dimension of the unstable subspace
  if nus === nothing
    nus = numindvar - one(numindvar)
  end
  (nus >= numindvar) && error("`nus` must be smaller than `numindvar`.")

  isinplace = DiffEqBase.isinplace(f)

  p == nothing && error("You must have parameters to use parameter sensitivity calculations!")
  !(u0 isa AbstractVector) && error("`u` has to be an AbstractVector.")

  # segmentation: determine length of segmentation and spacing between saved points
  T_seg = (tspan[2]-tspan[1])/nseg # length of each segment
  dtsave = T_seg/(nstep-1)

  # inhomogenous forward sensitivity problem
  chunk_size = determine_chunksize(numparams,sensealg)
  autodiff = alg_autodiff(sensealg)
  difftype = diff_type(sensealg)
  autojacvec = sensealg.autojacvec
  forward_prob_v = ODEForwardSensitivityProblem(f,u0,tspan,p,ForwardSensitivity(chunk_size=chunk_size,autodiff=autodiff,
                                                diff_type=difftype,autojacvec=autojacvec);kwargs...)
  # homogenous forward sensitivity problem
  forward_prob_w = ODEForwardSensitivityProblem(f,u0,tspan,p,ForwardSensitivity(chunk_size=chunk_size,autodiff=autodiff,
                                                diff_type=difftype,autojacvec=autojacvec);homogenous=true, kwargs...)

  sense = NILSSSensitivityFunction(sensealg,f,u0,sensealg,p,tspan,g,dg)

  # pre-allocate variables
  gsave = Matrix{eltype(u0)}(undef, nstep, nseg)
  y = Array{eltype(u0)}(undef, numindvar, nstep, nseg)
  dudt = similar(y)
  dgdu = similar(y)
  vstar = Array{eltype(u0)}(undef, numparams, numindvar, nstep, nseg) # generalization for several parameters numindvar*numparams
  vstar_perp = Array{eltype(u0)}(undef, numparams, numindvar, nstep, nseg)
  w = Array{eltype(u0)}(undef, numparams, numindvar, nstep, nseg, nus)
  w_perp = similar(w)

  # assign initial values to y, v*, w
  y[:,1,1] .= u0
  for i=1:numparams
    _vstar = @view vstar[i,:,1,1]
    copyto!(_vstar, zero(u0))
    for ius=1:nus
      _w = @view w[i,:,1,1,ius]
      rand!(rng,_w)
      normalize!(_w)
    end
  end

  R = Array{eltype(u0)}(undef, numparams, nseg, nstep-1, nus, nus)
  b = Array{eltype(u0)}(undef, numparams, nseg*(nstep-1)*nus)

  # a weight matrix for integration, 0.5 at interfaces
  weight = ones(nstep)
  weight[1] /= 2
  weight[end] /= 2

  # Construct Schur complement of the Lagrange multiplier method of the NILSS problem.
  # See the paper on FD-NILSS
  # find C^-1
  Cinv = Matrix{eltype(u0)}(undef, nseg*nus, nseg*nus)
  d = Vector{eltype(u0)}(undef, nseg*nus)
  B = Matrix{eltype(u0)}(undef, (nseg-1)*nus, nseg*nus)

  a = Vector{eltype(u0)}(undef, nseg*nus)
  v = Array{eltype(u0)}(undef, numindvar, nstep, nseg)
  v_perp = similar(v)

  # only need to use last step in each segment
  ξ = Matrix{eltype(u0)}(undef, nseg, 2)

  res = similar(u0, numparams)

  NILSSProblem{typeof(sensealg),typeof(sense),typeof(forward_prob_v),typeof(forward_prob_w),typeof(prob),
    typeof(T_seg),typeof(dtsave),typeof(gsave),typeof(y),typeof(vstar),typeof(w),typeof(R),
    typeof(b),typeof(weight),typeof(Cinv),typeof(d),typeof(B),typeof(a),typeof(v),typeof(ξ),
    typeof(g),typeof(dg),typeof(res)}(sensealg,sense,forward_prob_v,forward_prob_w,prob,
    nus,T_seg,dtsave,gsave,y,dudt,dgdu,vstar,vstar_perp,w,w_perp,R,b,weight,Cinv,d,
    B,a,v,v_perp,ξ,g,dg,res)
end


function (NS::NILSSForwardSensitivityFunction)(du,u,p,t)
  @unpack S = NS
  y = @view u[1:S.numindvar] # These are the independent variables
  dy = @view du[1:S.numindvar]
  S.f(dy,y,p,t) # Make the first part be the ODE

  # Now do sensitivities
  # Compute the Jacobian

  if !S.isautojacvec
    if DiffEqBase.has_jac(S.f)
      S.jac(S.J,y,p,t) # Calculate the Jacobian into J
    else
      S.uf.t = t
      jacobian!(S.J, S.uf, y, S.f_cache, S.alg, S.jac_config)
    end
  end


  if DiffEqBase.has_paramjac(S.f)
    S.paramjac(S.pJ,y,p,t) # Calculate the parameter Jacobian into pJ
  else
    S.pf.t = t
    S.pf.u .= y
    jacobian!(S.pJ, S.pf, p, S.f_cache, S.alg, S.paramjac_config)
  end


  # Compute the parameter derivatives
  for j=1:nus+1
    for i in eachindex(p)
      Sj = @view u[i*S.numindvar+1:(i+1)*S.numindvar]
      dp = @view du[i*S.numindvar+1:(i+1)*S.numindvar]
      if !S.isautojacvec
        mul!(dp,S.J,Sj)
      else
        jacobianvec!(dp, S.uf, y, Sj, S.alg, S.jac_config)
      end
      dp .+= @view S.pJ[:,i]
    end
  end
end

function forward_sense_W(prob::NILSSProblem,sensealg::NILSS,alg)
  #TODO determine a good dtsave (ΔT in paper, see Sec.4.2)
  @unpack nus, T_seg, dtsave, w, w_perp, R, y, dudt, gsave, forward_prob_w = prob
  @unpack nseg, nstep = sensealg

  # push forward
  t0 = forward_prob_w.tspan[1]
  _prob = remake(forward_prob_w, tspan=(t0, t0+T_seg))
  for iseg=1:nseg
    # compute y, w, dudt, gsave, dJdu
    solve(_prob, alg, saveat=dtsave)
    # calculate w_perp, store dudt and objective for vstar and later
    for istep=1:nstep
      f[iseg, istep], J[iseg, istep], dJdu[iseg, istep] = fJJu(u[iseg, istep], s)
      for ius=1:nus
        w_perp[iseg, i, ius] = w[iseg, i,ius] - np.dot(w[iseg, i, ius], f[iseg, i]) / np.dot(f[iseg, i], f[iseg, i]) * f[iseg, i]
      end
    end

    # renormalize at interfaces
    Q_temp, R_temp = qr(w_perp[iseg,-1].T)
    Rs.append(R_temp)

    if iseg < nseg-1:
      u[iseg+1, 0] = u[iseg, -1]
      w[iseg+1, 0] = Q_temp.T
      vstar[iseg+1,0] = p_temp
end

function shadow_forward(prob::NILSSProblem,alg)
  shadow_forward(prob,prob.sensealg,alg)
end

function shadow_forward(prob::NILSSProblem,sensealg::NILSS,alg)
  @unpack res = sensealg

  # compute vstar, w
  forward_sense_W(prob,sensealg,alg)

  return res
end
