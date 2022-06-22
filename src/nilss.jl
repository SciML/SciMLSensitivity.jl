struct NILSSSensitivityFunction{iip,F,Alg,
     PGPU,PGPP,CONFU,CONGP,DGVAL,DG,jType,RefType} <: DiffEqBase.AbstractODEFunction{iip}
  f::F
  alg::Alg
  numparams::Int
  numindvar::Int
  pgpu::PGPU
  pgpp::PGPP
  pgpu_config::CONFU
  pgpp_config::CONGP
  dg_val::DGVAL
  dg::DG
  jevery::jType # if concrete_solve interface for discrete cost functions is used
  cur_time::RefType
end

function NILSSSensitivityFunction(sensealg,f,u0,p,tspan,g,dg,jevery=nothing,cur_time=nothing)

  numparams = length(p)
  numindvar = length(u0)

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
    pgpp_config = build_grad_config(sensealg,pgpp,u0,tspan[1])
    dg_val = (similar(u0, numindvar),similar(u0, numparams))
    dg_val[1] .= false
    dg_val[2] .= false
  end

  NILSSSensitivityFunction{isinplace(f),typeof(f),typeof(sensealg),
                             typeof(pgpu),typeof(pgpp),typeof(pgpu_config),typeof(pgpp_config),typeof(dg_val),typeof(dg),typeof(jevery),typeof(cur_time)}(
                             f,sensealg,numparams,numindvar,pgpu,pgpp,pgpu_config,pgpp_config,dg_val,dg,jevery,cur_time)
end


struct NILSSProblem{A,CacheType,FSprob,probType,u0Type,vstar0Type,w0Type,
    TType,dtType,gType,yType,vstarType,
    wType,RType,bType,weightType,CType,dType,BType,aType,vType,xiType,
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
  ξ::xiType
  g::G
  dg::DG
  res::resType
end


function NILSSProblem(prob, sensealg::NILSS, t=nothing, dg = nothing;
                            kwargs...)

  @unpack f, p, u0, tspan = prob
  @unpack nseg, nstep, nus, rng, g = sensealg  #number of segments on time interval, number of steps saved on each segment

  numindvar = length(u0)
  numparams = length(p)

  # some shadowing sensealgs require knowledge of g
  check_for_g(sensealg,g)

  # integer dimension of the unstable subspace
  if nus === nothing
    nus = numindvar - one(numindvar)
  end
  (nus >= numindvar) && error("`nus` must be smaller than `numindvar`.")

  isinplace = DiffEqBase.isinplace(f)

  p === nothing && error("You must have parameters to use parameter sensitivity calculations!")
  !(u0 isa AbstractVector) && error("`u` has to be an AbstractVector.")

  # segmentation: determine length of segmentation and spacing between saved points
  T_seg = (tspan[2]-tspan[1])/nseg # length of each segment
  dtsave = T_seg/(nstep-1)

  # assert that dtsave is chosen such that all ts are hit if concrete solve interface/discrete costs are used
  if t!==nothing
    @assert t isa StepRangeLen
    dt_ts = step(t)
    @assert dt_ts >= dtsave
    @assert T_seg >= dt_ts
    jevery = Int(dt_ts/dtsave) # will throw an inexact error if dt_ts is not a multiple of dtsave. (could be more sophisticated)
    cur_time = Ref(1)
  else
    jevery = nothing
    cur_time = nothing
  end

  # inhomogenous forward sensitivity problem
  chunk_size = determine_chunksize(numparams,sensealg)
  autodiff = alg_autodiff(sensealg)
  difftype = diff_type(sensealg)
  autojacvec = sensealg.autojacvec
  # homogenous + inhomogenous forward sensitivity problems
  forward_prob = ODEForwardSensitivityProblem(f,u0,tspan,p,ForwardSensitivity(chunk_size=chunk_size,autodiff=autodiff,
                                                diff_type=difftype,autojacvec=autojacvec);nus=nus, kwargs...)

  sense = NILSSSensitivityFunction(sensealg,f,u0,p,tspan,g,dg,jevery,cur_time)

  # pre-allocate variables
  gsave = Matrix{eltype(u0)}(undef, nstep, nseg)
  y = Array{eltype(u0)}(undef, numindvar, nstep, nseg)
  dudt = similar(y)
  dgdu = similar(y)
  vstar = Array{eltype(u0)}(undef, numparams, numindvar, nstep, nseg) # generalization for several parameters numindvar*numparams
  vstar_perp = similar(vstar)
  w = Array{eltype(u0)}(undef, numindvar, nstep, nseg, nus)
  w_perp = similar(w)

  # assign initial values to y, v*, w
  y[:,1,1] .= u0
  for i=1:numparams
    _vstar = @view vstar[i,:,1,1]
    copyto!(_vstar, zero(u0))
  end
  for ius=1:nus
    _w = @view w[:,1,1,ius]
    rand!(rng,_w)
    normalize!(_w)
  end

  vstar0 = zeros(eltype(u0), numindvar*numparams)
  w0 = vec(w[:,1,1,:])

  R = Array{eltype(u0)}(undef, numparams, nseg-1, nus, nus)
  b = Array{eltype(u0)}(undef, numparams, (nseg-1)*nus)

  # a weight matrix for integration, 0.5 at interfaces
  weight = ones(1,nstep)
  weight[1] /= 2
  weight[end] /= 2

  # Construct Schur complement of the Lagrange multiplier method of the NILSS problem.
  # See the paper on FD-NILSS
  # find C^-1
  Cinv = Matrix{eltype(u0)}(undef, nseg*nus, nseg*nus)
  Cinv .*= false
  d = Vector{eltype(u0)}(undef, nseg*nus)
  B = Matrix{eltype(u0)}(undef, (nseg-1)*nus, nseg*nus)
  B .*= false

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
    if has_original_jac(S.f)
      S.original_jac(S.J,y,p,t) # Calculate the Jacobian into J
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
      indx1 = (j-1)*S.numindvar*1 + i*S.numindvar+1
      indx2 = (j-1)*S.numindvar*1 + (i+1)*S.numindvar
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
  @unpack nus, T_seg, dtsave, vstar, vstar_perp, w, w_perp, R, b, y, dudt, gsave, dgdu, forward_prob, u0, vstar0, w0 = prob
  @unpack p, f = forward_prob
  @unpack S, sensealg = f
  @unpack nseg, nstep = nilss
  @unpack numindvar, numparams = S

  # push forward
  t1 = forward_prob.tspan[1]
  t2 = forward_prob.tspan[1]+T_seg
  _prob = ODEForwardSensitivityProblem(S.f,u0,(t1,t2),p,sensealg;nus=nus,w0=w0,v0=vstar0)

  for iseg=1:nseg
    # compute y, w, vstar
    # _sol is a numindvar*(1+nus+1) x nstep matrix

    dt = (t2 - t1) / (nstep-1)
    _sol = Array(solve(_prob, alg, saveat=t1:dt:t2))

    store_y_w_vstar!(y, w, vstar, _sol, nus, numindvar, numparams, iseg)

    # store dudt, objective g (gsave), and its derivative wrt. to u (dgdu)
    dudt_g_dgdu!(dudt, gsave, dgdu, prob, y, forward_prob.p, iseg)

    # calculate w_perp, vstar_perp
    perp!(w_perp, vstar_perp, w, vstar, dudt, iseg, numparams, nstep, nus)

    # update sense problem
    if iseg < nseg
      # renormalize at interfaces
      renormalize!(R,b,w_perp,vstar_perp,y,vstar,w,iseg,numparams,nus)
      t1 = forward_prob.tspan[1]+iseg*T_seg
      t2 = forward_prob.tspan[1]+(iseg+1)*T_seg
      _prob = ODEForwardSensitivityProblem(S.f,y[:,1,iseg+1],(t1,t2),p,sensealg; nus=nus,
                                 w0=vec(w[:,1,iseg+1,:]),v0=vec(vstar[:,:,1,iseg+1]))
    end

  end
end

function store_y_w_vstar!(y, w, vstar, sol, nus, numindvar, numparams, iseg)
  # fill y
  _y = @view y[:,:,iseg]
  copyto!(_y, (@view sol[1:numindvar,:]))

  # fill w
  # only calculate w one time, w can be reused for each parameter
  for j=1:nus
   indx1 = (j-1)*numindvar*1 + numindvar+1
   indx2 = (j-1)*numindvar*1 + 2*numindvar

   _w = @view w[:,:,iseg, j]
   copyto!(_w, (@view sol[indx1:indx2,:]))
  end

  # fill vstar
  for i=1:numparams
    indx1 = nus*numindvar*1 + i*numindvar+1
    indx2 = nus*numindvar*1 + (i+1)*numindvar
    _vstar = @view vstar[i,:,:,iseg]
    copyto!(_vstar, (@view sol[indx1:indx2,:]))
  end

  return nothing
end

function dudt_g_dgdu!(dudt, gsave, dgdu, nilssprob::NILSSProblem, y, p, iseg)
  @unpack sensealg, diffcache, dg, g, prob = nilssprob
  @unpack prob = nilssprob
  @unpack jevery, cur_time = diffcache # akin to ``discrete"

  _y = @view y[:,:,iseg]

  for (j,u) in enumerate(eachcol(_y))
    _dgdu = @view dgdu[:,j,iseg]
    _dudt = @view dudt[:,j,iseg]

    # compute dudt
    if isinplace(prob)
      prob.f(_dudt,u,p,nothing)
    else
      copyto!(_dudt,prob.f(u,p,nothing))
    end

    # compute objective
    gsave[j,iseg] = g(u,p,nothing)

    #  compute gradient of objective wrt. state
    if jevery!==nothing
      # only bump on every jevery entry
      # corresponds to (iseg-1)* value of dg
      if (j-1) % jevery == 0
        accumulate_cost!(_dgdu, dg, u, p, nothing, sensealg, diffcache, cur_time[])
        cur_time[] += one(jevery)
      end
    else
      # continuous cost function
      accumulate_cost!(_dgdu, dg, u, p, nothing, sensealg, diffcache, j)
    end
  end
  jevery !== nothing && (cur_time[] -= one(jevery)) # interface between segments gets two bumps
  return nothing
end

function perp!(w_perp, vstar_perp, w, vstar, dudt, iseg, numparams, nsteps, nus)
  for indx_steps=1:nsteps
    _dudt = @view dudt[:,indx_steps,iseg]
    for indx_nus=1:nus
      _w_perp = @view w_perp[:,indx_steps,iseg,indx_nus]
      _w = @view w[:,indx_steps,iseg,indx_nus]
      perp!(_w_perp, _w, _dudt)
    end
    for indx_params=1:numparams
      _vstar_perp = @view vstar_perp[indx_params,:,indx_steps,iseg]
      _vstar = @view vstar[indx_params,:,indx_steps,iseg]
      perp!(_vstar_perp, _vstar, _dudt)
    end
  end

  return nothing
end

function perp!(v1, v2, v3)
  v1 .= v2 - dot(v2, v3)/dot(v3, v3) * v3
end

function renormalize!(R,b,w_perp,vstar_perp,y,vstar,w,iseg,numparams,nus)
  for i=1:numparams
    _b = @view b[i,(iseg-1)*nus+1:iseg*nus]
    _R = @view R[i,iseg,:,:]
    _w_perp = @view w_perp[:,end,iseg,:]
    _vstar_perp = @view vstar_perp[i,:,end,iseg]
    _w = @view w[:,1,iseg+1,:]
    _vstar = @view vstar[i,:,1,iseg+1]

    Q_temp, R_temp = qr(_w_perp)
    b_tmp = @view (Q_temp'*_vstar_perp)[1:nus]

    copyto!(_b, b_tmp)
    copyto!(_R, R_temp)
    # set new initial values
    copyto!(_w, (@view Q_temp[:,1:nus]))
    copyto!(_vstar, _vstar_perp - Q_temp*b_tmp)
  end
  _yend = @view y[:,end,iseg]
  _ystart = @view y[:,1,iseg+1]
  copyto!(_ystart, _yend)

  return nothing
end


function compute_Cinv!(Cinv,w_perp,weight,nseg,nus,indxp)
  # construct Schur complement of Lagrange multiplier
  _weight = @view weight[1,:]
  for iseg=1:nseg
    _C = @view Cinv[(iseg-1)*nus+1:iseg*nus, (iseg-1)*nus+1:iseg*nus]
    for i=1:nus
      wi = @view w_perp[:,:,iseg,i]
      for j =1:nus
        wj = @view w_perp[:,:,iseg,j]
        _C[i,j] = sum(wi .* wj * _weight)
      end
    end
    invC = inv(_C)
    copyto!(_C, invC)
  end
  return nothing
end

function compute_d!(d,w_perp,vstar_perp,weight,nseg,nus,indxp)
  # construct d
  _weight = @view weight[1,:]
  for iseg=1:nseg
    _d = @view d[(iseg-1)*nus+1:iseg*nus]
    vi = @view vstar_perp[indxp,:,:,iseg]
    for i=1:nus
      wi = @view w_perp[:,:,iseg,i]
      _d[i] = sum(wi .* vi * _weight)
    end
  end
  return nothing
end

function compute_B!(B,R,nseg,nus,indxp)
  for iseg=1:nseg-1
    _B = @view B[(iseg-1)*nus+1:iseg*nus, (iseg-1)*nus+1:iseg*nus]
    _R = @view R[indxp,iseg,:,:]
    copyto!(_B, -_R)
    # off diagonal one
    for i=1:nus
      B[(iseg-1)*nus+i, iseg*nus+i] = one(eltype(R))
    end
  end
  return nothing
end

function compute_a!(a,B,Cinv,b,d,indxp)
  _b = @view b[indxp,:]

  lbd = (-B*Cinv*B') \ (B*Cinv*d + _b)
  a .= -Cinv*(B'*lbd + d)
  return nothing
end

function compute_v!(v,v_perp,vstar,vstar_perp,w,w_perp,a,nseg,nus,indxp)
  _vstar = @view vstar[indxp,:,:,:]
  _vstar_perp = @view vstar_perp[indxp,:,:,:]

  copyto!(v, _vstar)
  copyto!(v_perp, _vstar_perp)

  for iseg=1:nseg
    vi = @view v[:,:,iseg]
    vpi = @view v_perp[:,:,iseg]
    for i=1:nus
      wi = @view w[:,:,iseg,i]
      wpi = @view w_perp[:,:,iseg,i]

      vi .+= a[(iseg-1)*nus+i]*wi
      vpi .+= a[(iseg-1)*nus+i]*wpi
    end
  end

  return nothing
end

function compute_xi(ξ,v,dudt,nseg)
  for iseg=1:nseg
    _v = @view v[:,1,iseg]
    _dudt = @view dudt[:,1,iseg]
    ξ[iseg,1] = dot(_v,_dudt)/dot(_dudt,_dudt)

    _v = @view v[:,end,iseg]
    _dudt = @view dudt[:,end,iseg]
    ξ[iseg,2] = dot(_v,_dudt)/dot(_dudt,_dudt)
  end
  # check if segmentation is chosen correctly
  _ξ = ξ[:,1]
  all(_ξ.<1e-4) || @warn "Detected a large value of ξ at the beginning of a segment."
  return nothing
end

function accumulate_cost!(_dgdu, dg, u, p, t, sensealg::NILSS, diffcache::NILSSSensitivityFunction, j)
  @unpack dg_val, pgpu, pgpu_config, pgpp, pgpp_config = diffcache

  if dg===nothing
    if dg_val isa Tuple
      DiffEqSensitivity.gradient!(dg_val[1], pgpu, u, sensealg, pgpu_config)
      copyto!(_dgdu, dg_val[1])
    else
      DiffEqSensitivity.gradient!(dg_val, pgpu, u, sensealg, pgpu_config)
      copyto!(_dgdu, dg_val)
    end
  else
    if dg_val isa Tuple
      dg[1](dg_val[1],u,p,nothing,j)
      @. _dgdu = dg_val[1]
    else
      dg(dg_val,u,p,nothing,j)
      @. _dgdu = dg_val
    end
  end

  return nothing
end

function shadow_forward(prob::NILSSProblem,alg; sensealg=prob.sensealg)
  shadow_forward(prob,sensealg,alg)
end

function shadow_forward(prob::NILSSProblem,sensealg::NILSS,alg)
  @unpack nseg, nstep = sensealg
  @unpack res, nus, dtsave, vstar, vstar_perp, w, w_perp, R, b, dudt,
          gsave, dgdu, forward_prob, weight, Cinv, d, B, a, v, v_perp, ξ = prob
  @unpack numindvar, numparams = forward_prob.f.S

  # reset dg pointer
  @unpack jevery, cur_time = prob.diffcache
  jevery !== nothing && (cur_time[] = one(jevery))

  # compute vstar, w
  forward_sense(prob,sensealg,alg)

  # compute avg objective
  gavg = sum(prob.weight*gsave)/((nstep-1)*nseg)

  # reset gradient
  res .*= false

  # loop over parameters
  for i=1:numparams
    compute_Cinv!(Cinv,w_perp,weight,nseg,nus,i)
    compute_d!(d,w_perp,vstar_perp,weight,nseg,nus,i)
    compute_B!(B,R,nseg,nus,i)
    compute_a!(a,B,Cinv,b,d,i)
    compute_v!(v,v_perp,vstar,vstar_perp,w,w_perp,a,nseg,nus,i)
    compute_xi(ξ,v,dudt,nseg)


    _weight = @view weight[1,:]

    for iseg=1:nseg
      _dgdu = @view dgdu[:,:,iseg]
      _v = @view v[:,:,iseg]
      res[i] += sum((_v.*_dgdu)*_weight)/((nstep-1)*nseg)
      res[i] += ξ[iseg,end]*(gavg-gsave[end,iseg])/(dtsave*(nstep-1)*nseg)
    end
  end

  return res
end

check_for_g(sensealg::NILSS,g) = (g===nothing && error("To use NILSS, g must be passed as a kwarg to `NILSS(g=g)`."))
