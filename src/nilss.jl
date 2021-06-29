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


struct NILSSProblem{A,CacheType,FSprob,probType,u0Type,vstar0Type,w0Type,
    TType,dtType,gType,yType,vstarType,
    wType,RType,bType,weightType,CType,dType,BType,aType,vType,ksiType,
    G,DG,resType}
  sensealg::A
  diffcache::CacheType
  forward_prob::FSprob
  prob::probType
  u0::u0Type
  vstar0::vstar0Type
  w0::w0Type
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
  # homogenous + inhomogenous forward sensitivity problems
  forward_prob = ODEForwardSensitivityProblem(f,u0,tspan,p,ForwardSensitivity(chunk_size=chunk_size,autodiff=autodiff,
                                                diff_type=difftype,autojacvec=autojacvec);nus=nus, kwargs...)

  sense = NILSSSensitivityFunction(sensealg,f,u0,sensealg,p,tspan,g,dg)

  # pre-allocate variables
  gsave = Matrix{eltype(u0)}(undef, nstep, nseg)
  y = Array{eltype(u0)}(undef, numindvar, nstep, nseg)
  dudt = similar(y)
  dgdu = similar(y)
  vstar = Array{eltype(u0)}(undef, numparams, numindvar, nstep, nseg) # generalization for several parameters numindvar*numparams
  vstar_perp = similar(vstar)
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
  vstar0 = zeros(eltype(u0), numindvar*numparams)
  w0 = vec(w[:,:,1,1,:])

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

  NILSSProblem{typeof(sensealg),typeof(sense),typeof(forward_prob),typeof(prob),
    typeof(u0), typeof(vstar0), typeof(w0),
    typeof(T_seg),typeof(dtsave),typeof(gsave),typeof(y),typeof(vstar),typeof(w),typeof(R),
    typeof(b),typeof(weight),typeof(Cinv),typeof(d),typeof(B),typeof(a),typeof(v),typeof(ξ),
    typeof(g),typeof(dg),typeof(res)}(sensealg,sense,forward_prob,prob,u0,vstar0,w0,
    nus,T_seg,dtsave,gsave,y,dudt,dgdu,vstar,vstar_perp,w,w_perp,R,b,weight,Cinv,d,
    B,a,v,v_perp,ξ,g,dg,res)
end


function (NS::NILSSForwardSensitivityFunction)(du,u,p,t)
  @unpack S, nus = NS
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
      indx1 = (j-1)*S.numindvar*S.numparams + i*S.numindvar+1
      indx2 = (j-1)*S.numindvar*S.numparams + (i+1)*S.numindvar
      Sj = @view u[indx1:indx2]
      dp = @view du[indx1:indx2]
      if !S.isautojacvec
        mul!(dp,S.J,Sj)
      else
        jacobianvec!(dp, S.uf, y, Sj, S.alg, S.jac_config)
      end
      if j == nus+1
        # inhomogenous (otherwise homogenous tangent solution)
        dp .+= @view S.pJ[:,i]
      end
    end
  end
  return nothing
end

function forward_sense(prob::NILSSProblem,nilss::NILSS,alg)
  #TODO determine a good dtsave (ΔT in paper, see Sec.4.2)
  @unpack nus, T_seg, dtsave, vstar, vstar_perp, w, w_perp, R, y, dudt, gsave, forward_prob, u0, vstar0, w0 = prob
  @unpack p, f = forward_prob
  @unpack S, sensealg = f
  @unpack nseg, nstep = nilss

  # push forward
  t1 = forward_prob.tspan[1]
  t2 = forward_prob.tspan[1]+T_seg
  _prob = ODEForwardSensitivityProblem(S.f,u0,(t1,t2),p,sensealg;nus=nus,w0=w0,v0=vstar0)

  for iseg=1:nseg
    # compute y, w, vstar
    _sol = Array(solve(_prob, alg, saveat=dtsave, dt=dtsave)(_prob.tspan[1]:dtsave:_prob.tspan[2]))
    @show size(_sol), size(y), size(w), size(vstar)
    error()
    store_y_w_vstar!(y, w, vstar, _sol, iseg)

    # store dudt, objective g (gsave), and its derivative wrt. to u (dgdu)
    dudt_g_dgdu!(dudt, gsave, dgdu, u, forward_prob.p, iseg)

    # calculate w_perp, vstar_perp
    _vstar_perp = @view vstar_perp[:, :, :, iseg]
    _vstar = @view vstar[:, :, :, iseg]
    perp!(_vstar_perp, _vstar, dudt)
    for ius=1:nus
      _w_perp = @view _w_perp[:, :, :, iseg, ius]
      _w = @view w[]
      perp!(_w_perp, _w, dudt)
    end

    # renormalize at interfaces
    Q_temp, R_temp = qr(w_perp[iseg,end].T)

    if iseg < nseg
      set_new_initial_values!(u0,w0,vstar0, y[iseg, -1], Q_temp.T, p_temp)
      t1 = forward_prob.tspan[1]+j*T_seg
      t2 = forward_prob.tspan[1]+(j+1)*T_seg
      _prob = ODEForwardSensitivityProblem(S,u0,(t1,t2),p,sensealg;nus=nus,w0=w0,v0=vstar0,kwargs...)
    end

  end
end

function store_y_w_vstar!(y, w, vstar, _sol, iseg)
  return nothing
end

function dudt_g_dgdu!(dudt, gsave, dgdu, u, p, iseg)
  return nothing
end

function perp!(v_perp, v, dudt)
  #w_perp[iseg, i, ius] = w[iseg, i,ius] - dot(w[iseg, i, ius], f[iseg, i]) / dot(f[iseg, i], f[iseg, i]) * f[iseg, i]
  return nothing
end

function set_new_initial_values!(u0,w0,vstar0, yend, Qtransp, ptemp)
  u0 .= yend
  w0 .= Qtransp
  vstar0 .= ptemp
  return nothing
end

function Cinv()
  # compute C^-1
  # Cinvs = []
  #   for iseg in range(nseg):
  #       C_iseg = np.zeros([nus, nus])
  #       for i in range(nus):
  #           for j in range(nus):
  #               C_iseg[i,j] = np.sum(w_perp[iseg, :, i, :] * w_perp[iseg, :, j, :] * weight[:, np.newaxis])
  #       Cinvs.append(np.linalg.inv(C_iseg))
  #   Cinv = block_diag(*Cinvs)
end

function compute_d()
  # construct d
    # ds = []
    # for iseg in range(nseg):
    #     d_iseg = np.zeros(nus)
    #     for i in range(nus):
    #         d_iseg[i] = np.sum(w_perp[iseg, :, i, :] * vstar_perp[iseg] * weight[:, np.newaxis])
    #     ds.append(d_iseg)
    # d = np.ravel(ds)
end

function shadow_forward(prob::NILSSProblem,alg)
  shadow_forward(prob,prob.sensealg,alg)
end

function shadow_forward(prob::NILSSProblem,sensealg::NILSS,alg)
  @unpack res = prob

  # compute vstar, w
  forward_sense(prob,sensealg,alg)

  # compute Javg
  #gavg = nsum(J*weight[np.newaxis,:]) / (nstep-1) / nseg


  return res
end
