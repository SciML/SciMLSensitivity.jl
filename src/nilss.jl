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


struct NILSSProblem{A,FSV,FSW,solType,TType,dtType,gType,yType,dudtType,vstarType,
    wType,RType,bType,weightType,CType,dType,BType,aType,vType,ksiType,
    G,DG,resType}
  sensealg::A
  forward_prob_v::FSV
  forward_prob_w::FSW
  sol::solType
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


function NILSSProblem(sol, sensealg::NILSS, g, dg = nothing; nus = nothing,
                            kwargs...)

  @unpack f, p, u0, tspan = sol.prob
  @unpack nseg, nstep, rng = sensealg  #number of segments on time interval, number of steps saved on each segment

  numindvar = len(u0)
  numparams = length(p)

  # integer dimension of the unstable subspace
  if nus === nothing
    nus = numindvar - one(numindvar)
  end
  (nus >= numindvar) && error("`nus` must be smaller than `numindvar`.")

  isinplace = DiffEqBase.isinplace(f)

  p == nothing && error("You must have parameters to use parameter sensitivity calculations!")
  !(sol.u isa AbstractVector) && error("`u` has to be an AbstractVector.")

  # segmentation: determine length of segmentation and spacing between saved points
  T_seg = (tspan[2]-tspan[1])/nseg # length of each segment
  dtsave = Tseg/(nstep-1)

  # inhomogenous forward sensitivity problem
  forward_prob_v = ODEForwardSensitivityProblem(f,u0,tspan,p,ForwardSensitivity();kwargs...)
  # homogenous forward sensitivity problem
  forward_prob_w = ODEForwardSensitivityProblem(f,u0,tspan,p,ForwardSensitivity();homogenous=true, kwargs...)

  sense = NILSSSensitivityFunction(sensealg,f,u0,sensealg,p,similar(u0),tspan,g,dg)

  # pre-allocate variables
  gsave = Matrix{eltype(u0)}(undef, nseg, nstep)
  y = Array{eltype(u0)}(undef, nseg, nstep, numindvar)
  dudt = similar(y)
  dgdu = similar(y)
  vstar = Array{eltype(u0)}(undef, nseg, nstep, numindvar, numparams) # generalization for several parameters numindvar*numparams
  vstar_perp = Array{eltype(u0)}(undef, nseg, nstep, numindvar, numparams)
  w = Array{eltype(u0)}(undef, nseg, nstep, nus, numindvar, numparams)
  w_perp = similar(w)

  R = Array{eltype(u0)}(undef, nseg, nstep-1, nus, nus, numparams)
  b = Array{eltype(u0)}(undef, nseg*(nstep-1)*nus, numparams)

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
  v = Array{eltype(u0)}(undef, nseg, nstep, numindvar)
  v_perp = similar(v)

  # only need to use last step in each segment
  ξ = Matrix{eltype(u0)}(undef, nseg, 2)

  res = similar(u0, numparams)

  NILSSProblem{typeof(sensealg),typeof(sense),typeof(sol),typeof(dt),
    typeof(umid),typeof(dudt),
    typeof(S),typeof(F),typeof(b),typeof(η),typeof(w),typeof(v),typeof(window),typeof(Δt),
    typeof(g0),typeof(g),typeof(dg),typeof(res)}(sensealg,sense,sol,dt,umid,dudt,S,F,b,η,w,v,
    window,Δt,Nt,g0,g,dg,res)
end


function shadow_forward(prob::NILSSProblem)
  shadow_forward(prob,prob.sensealg)
end

function shadow_forward(prob::NILSSProblem,sensealg::NILSS)
  @unpack res

  return res
end
