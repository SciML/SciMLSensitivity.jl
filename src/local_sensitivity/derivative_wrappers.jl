# Not in FiniteDiff because `u` -> scalar isn't used anywhere else,
# but could be upstreamed.
mutable struct UGradientWrapper{fType,tType,P} <: Function
  f::fType
  t::tType
  p::P
end

(ff::UGradientWrapper)(uprev) = ff.f(uprev,ff.p,ff.t)

mutable struct ParamGradientWrapper{fType,tType,uType} <: Function
  f::fType
  t::tType
  u::uType
end

(ff::ParamGradientWrapper)(p) = ff.f(ff.u,p,ff.t)

# the next four definitions are only needed in case of non-diagonal SDEs

mutable struct ParamNonDiagNoiseGradientWrapper{fType,tType,uType} <: Function
  f::fType
  t::tType
  u::uType
end

(ff::ParamNonDiagNoiseGradientWrapper)(p) = vec(ff.f(ff.u,p,ff.t))

mutable struct ParamNonDiagNoiseJacobianWrapper{fType,tType,uType,duType} <: Function
  f::fType
  t::tType
  u::uType
  du::duType
end

function (ff::ParamNonDiagNoiseJacobianWrapper)(p)
  du1 = similar(p, size(ff.du))
  ff.f(du1,ff.u,p,ff.t)
  return vec(du1)
end

function (ff::ParamNonDiagNoiseJacobianWrapper)(du1,p)
  ff.f(du1,ff.u,p,ff.t)
  return vec(du1)
end

mutable struct UNonDiagNoiseGradientWrapper{fType,tType,P} <: Function
  f::fType
  t::tType
  p::P
end

(ff::UNonDiagNoiseGradientWrapper)(uprev) = vec(ff.f(uprev,ff.p,ff.t))

mutable struct UNonDiagNoiseJacobianWrapper{fType,tType,P,duType} <: Function
  f::fType
  t::tType
  p::P
  du::duType
end

(ff::UNonDiagNoiseJacobianWrapper)(uprev) = (du1 = similar(ff.du); ff.f(du1,uprev,ff.p,ff.t); vec(du1))

function (ff::UNonDiagNoiseJacobianWrapper)(du1,uprev)
  ff.f(du1,uprev,ff.p,ff.t)
  return vec(du1)
end


Base.@pure function determine_chunksize(u,alg::DiffEqBase.AbstractSensitivityAlgorithm)
  determine_chunksize(u,get_chunksize(alg))
end

Base.@pure function determine_chunksize(u,CS)
  if CS != 0
    return CS
  else
    return ForwardDiff.pickchunksize(length(u))
  end
end

function jacobian(f, x::AbstractArray{<:Number}, alg::DiffEqBase.AbstractSensitivityAlgorithm)
  if alg_autodiff(alg)
    J = ForwardDiff.jacobian(f, x)
  else
    J = FiniteDiff.finite_difference_jacobian(f, x)
  end
  return J
end


function jacobian!(J::AbstractMatrix{<:Number}, f, x::AbstractArray{<:Number},
                   fx::Union{Nothing,AbstractArray{<:Number}}, alg::DiffEqBase.AbstractSensitivityAlgorithm, jac_config)
  if alg_autodiff(alg)
    if fx === nothing
      ForwardDiff.jacobian!(J, f, x)
    else
      ForwardDiff.jacobian!(J, f, fx, x, jac_config)
    end
  else
    FiniteDiff.finite_difference_jacobian!(J, f, x, jac_config)
  end
  nothing
end

function gradient!(df::AbstractArray{<:Number}, f,
                   x::Union{Number,AbstractArray{<:Number}},
                   alg::DiffEqBase.AbstractSensitivityAlgorithm, grad_config)
    if alg_autodiff(alg)
        ForwardDiff.gradient!(df, f, x, grad_config)
    else
        FiniteDiff.finite_difference_gradient!(df, f, x, grad_config)
    end
    nothing
end

"""
  jacobianvec!(Jv, f, x, v, alg, (buffer, seed)) -> nothing

``Jv <- J(f(x))v``
"""
function jacobianvec!(Jv::AbstractArray{<:Number}, f, x::AbstractArray{<:Number},
                      v, alg::DiffEqBase.AbstractSensitivityAlgorithm, config)
  if alg_autodiff(alg)
    buffer, seed = config
    TD = typeof(first(seed))
    T  = typeof(first(seed).partials)
    @. seed = TD(x, T(tuple(v)))
    f(buffer, seed)
    Jv .= ForwardDiff.partials.(buffer, 1)
  else
    buffer1, buffer2 = config
    f(buffer1,x)
    T = eltype(x)
    # Should it be min? max? mean?
    ϵ = sqrt(eps(real(T))) * max(one(real(T)), abs(norm(x)))
    @. x += ϵ*v
    f(buffer2,x)
    @. x -= ϵ*v
    @. du = (buffer2 - buffer1)/ϵ
  end
  nothing
end

function vecjacobian!(dλ, y, λ, p, t, S::SensitivityFunction;
                      dgrad=nothing, dy=nothing)
  _vecjacobian!(dλ, y, λ, p, t, S, S.sensealg.autojacvec, dgrad, dy)
  return
end

function _vecjacobian!(dλ, y, λ, p, t, S::SensitivityFunction, isautojacvec::Bool, dgrad, dy)
  @unpack sensealg, f = S
  prob = getprob(S)
  if isautojacvec isa Bool && !isautojacvec
    @unpack J, uf, f_cache, jac_config = S.diffcache
    if !(prob isa DiffEqBase.SteadyStateProblem)
      if DiffEqBase.has_jac(f)
        f.jac(J,y,p,t) # Calculate the Jacobian into J
      else
        uf.t = t
        uf.p = p
        jacobian!(J, uf, y, f_cache, sensealg, jac_config)
      end
      mul!(dλ',λ',J)
    end
    if dgrad !== nothing
      @unpack pJ, pf, paramjac_config = S.diffcache
      if DiffEqBase.has_paramjac(f)
        # Calculate the parameter Jacobian into pJ
        f.paramjac(pJ,y,prob.p,t)
      else
        pf.t = t
        pf.u = y
        if DiffEqBase.isinplace(prob)
          jacobian!(pJ, pf, prob.p, f_cache, sensealg, paramjac_config)
        else
          temp = jacobian(pf, prob.p, sensealg)
          pJ .= temp
        end
      end
      mul!(dgrad',λ',pJ)
    end
    if dy !== nothing
      if DiffEqBase.isinplace(prob)
         f(dy, y, p, t)
      else
        dy[:] .= vec(f(y, p, t))
      end
    end
  elseif DiffEqBase.isinplace(prob)
    _vecjacobian!(dλ, y, λ, p, t, S, ReverseDiffVJP(), dgrad, dy)
  else
    _vecjacobian!(dλ, y, λ, p, t, S, ZygoteVJP(), dgrad, dy)
  end
  return
end

function _vecjacobian!(dλ, y, λ, p, t, S::SensitivityFunction, isautojacvec::TrackerVJP, dgrad, dy)
  @unpack sensealg, f = S
  prob = getprob(S)
  isautojacvec = get_jacvec(sensealg)
  if DiffEqBase.isinplace(prob)
    _dy, back = Tracker.forward(y, prob.p) do u, p
      out_ = map(zero, u)
      f(out_, u, p, t)
      Tracker.collect(out_)
    end
    tmp1, tmp2 = Tracker.data.(back(λ))
    dλ[:] .= vec(tmp1)
    dgrad !== nothing && (dgrad[:] .= vec(tmp2))
    dy !== nothing && (dy[:] .= vec(Tracker.data(_dy)))
  else
    _dy, back = Tracker.forward(y, prob.p) do u, p
      Tracker.collect(f(u, p, t))
    end
    tmp1, tmp2 = Tracker.data.(back(λ))
    dλ[:] .= vec(tmp1)
    dgrad !== nothing && (dgrad[:] .= vec(tmp2))
    dy !== nothing && (dy[:] .= vec(Tracker.data(_dy)))
  end
  return
end

function _vecjacobian!(dλ, y, λ, p, t, S::SensitivityFunction, isautojacvec::ReverseDiffVJP, dgrad, dy)
  @unpack sensealg, f = S
  prob = getprob(S)
  isautojacvec = get_jacvec(sensealg)

  if typeof(prob) <: SteadyStateProblem || (eltype(λ) <: eltype(prob.u0) && typeof(t) <: eltype(prob.u0) && compile_tape(sensealg.autojacvec))
    tape = S.diffcache.paramjac_config

  ## These other cases happen due to autodiff in stiff ODE solvers
  elseif DiffEqBase.isinplace(prob)
    _y = eltype(y) === eltype(λ) ? y : convert.(promote_type(eltype(y),eltype(λ)),y)
    tape = ReverseDiff.GradientTape((_y, prob.p, [t])) do u,p,t
      du1 = similar(u, size(u))
      f(du1,u,p,first(t))
      return vec(du1)
     end
  else
    _y = eltype(y) === eltype(λ) ? y : convert.(promote_type(eltype(y),eltype(λ)),y)
    tape = ReverseDiff.GradientTape((_y, prob.p, [t])) do u,p,t
      vec(f(u,p,first(t)))
    end
  end

  if prob isa DiffEqBase.SteadyStateProblem
    tu, tp = ReverseDiff.input_hook(tape)
  else
    tu, tp, tt = ReverseDiff.input_hook(tape)
  end
  output = ReverseDiff.output_hook(tape)
  ReverseDiff.unseed!(tu) # clear any "leftover" derivatives from previous calls
  ReverseDiff.unseed!(tp)
  if !(prob isa DiffEqBase.SteadyStateProblem)
    ReverseDiff.unseed!(tt)
  end
  ReverseDiff.value!(tu, y)
  ReverseDiff.value!(tp, prob.p)
  if !(prob isa DiffEqBase.SteadyStateProblem)
    ReverseDiff.value!(tt, [t])
  end
  ReverseDiff.forward_pass!(tape)
  ReverseDiff.increment_deriv!(output, λ)
  ReverseDiff.reverse_pass!(tape)
  copyto!(vec(dλ), ReverseDiff.deriv(tu))
  dgrad !== nothing && copyto!(vec(dgrad), ReverseDiff.deriv(tp))
  ReverseDiff.pull_value!(output)
  dy !== nothing && copyto!(vec(dy), ReverseDiff.value(output))
  return
end

function _vecjacobian!(dλ, y, λ, p, t, S::SensitivityFunction, isautojacvec::ZygoteVJP, dgrad, dy)
  @unpack sensealg, f = S
  prob = getprob(S)

  isautojacvec = get_jacvec(sensealg)
  if DiffEqBase.isinplace(prob)
    _dy, back = Zygote.pullback(y, prob.p) do u, p
      out_ = Zygote.Buffer(similar(u))
      f(out_, u, p, t)
      vec(copy(out_))
    end
    tmp1,tmp2 = back(λ)
    dλ[:] .= vec(tmp1)
    dgrad !== nothing && tmp2 != nothing && (dgrad[:] .= vec(tmp2))
    dy !== nothing && (dy[:] .= vec(_dy))
  else
    _dy, back = Zygote.pullback(y, prob.p) do u, p
      vec(f(u, p, t))
    end
    tmp1,tmp2 = back(λ)
    dλ[:] .= vec(tmp1)

    dy !== nothing && (dy[:] .= vec(_dy))
    dgrad !== nothing && tmp2 != nothing && (dgrad[:] .= vec(tmp2))
  end
  return
end


function jacNoise!(λ, y, p, t, S::SensitivityFunction;
                      dgrad=nothing, dλ=nothing, dy=nothing)
  _jacNoise!(λ, y, p, t, S, S.sensealg.noise, dgrad, dλ, dy)
  return
end

function _jacNoise!(λ, y, p, t, S::SensitivityFunction, isnoise::Bool, dgrad, dλ, dy)
  @unpack sensealg, f = S
  prob = getprob(S)
  if isnoise isa Bool && !isnoise
    if dgrad !== nothing
      @unpack pJ, pf, f_cache, paramjac_noise_config = S.diffcache
      if DiffEqBase.has_paramjac(f)
        # Calculate the parameter Jacobian into pJ
        f.paramjac(pJ,y,prob.p,t)
      else
        pf.t = t
        pf.u = y
        if DiffEqBase.isinplace(prob)
          jacobian!(pJ, pf, prob.p, nothing, sensealg, nothing)
          #jacobian!(pJ, pf, prob.p, f_cache, sensealg, paramjac_noise_config)
        else
          temp = jacobian(pf, prob.p, sensealg)
          pJ .= temp
        end
      end

      if StochasticDiffEq.is_diagonal_noise(prob)
        pJt = transpose(λ).*transpose(pJ)
        dgrad[:] .= vec(pJt)
      else
        m = size(prob.noise_rate_prototype)[2]
        for i in 1:m
          tmp = λ'*pJ[(i-1)*m+1:i*m,:]
          dgrad[:,i] .= vec(tmp)
        end
      end
    end

    if dλ !== nothing && (isnoisemixing(sensealg) || !StochasticDiffEq.is_diagonal_noise(prob))
      @unpack J, uf, f_cache, jac_noise_config = S.diffcache
      if dy!=nothing
        if DiffEqBase.isinplace(prob)
          f(dy, y, p, t)
        else
          dy .= f(y, p, t)
        end
      end

      if DiffEqBase.has_jac(f)
        f.jac(J,y,p,t) # Calculate the Jacobian into J
      else
        if DiffEqBase.isinplace(prob)
          if dy != nothing
            ForwardDiff.jacobian!(J,uf,dy,y)
          else
            if StochasticDiffEq.is_diagonal_noise(prob)
              dy = similar(y)
            else
              dy = similar(prob.noise_rate_prototype)
              f(dy, y, p, t)
              ForwardDiff.jacobian!(J,uf,dy,y)
            end
            f(dy, y, p, t)
            ForwardDiff.jacobian!(J,uf,dy,y)
          end
        else
          tmp = ForwardDiff.jacobian(uf,y)
          J .= tmp
        end
      #  uf.t = t
      #  uf.p = p
      #  jacobian!(J, uf, y, nothing, sensealg, nothing)
      end

      if StochasticDiffEq.is_diagonal_noise(prob)
        Jt = transpose(λ).*transpose(J)
        dλ[:] .= vec(Jt)
      else
        for i in 1:m
          tmp = λ'*J[(i-1)*m+1:i*m,:]
          dλ[:,i] .= vec(tmp)
        end
      end
    end
  elseif DiffEqBase.isinplace(prob)
    _jacNoise!(λ, y, p, t, S, ReverseDiffNoise(), dgrad, dλ, dy)
  else
    _jacNoise!(λ, y, p, t, S, ZygoteNoise(), dgrad, dλ, dy)
  end
  return
end

function _jacNoise!(λ, y, p, t, S::SensitivityFunction, isnoise::ReverseDiffNoise, dgrad, dλ, dy)
  @unpack sensealg, f = S
  prob = getprob(S)

  for (i, λi) in enumerate(λ)
    tapei = S.diffcache.paramjac_noise_config[i]
    tu, tp, tt = ReverseDiff.input_hook(tapei)
    output = ReverseDiff.output_hook(tapei)
    ReverseDiff.unseed!(tu) # clear any "leftover" derivatives from previous calls
    ReverseDiff.unseed!(tp)
    ReverseDiff.unseed!(tt)
    ReverseDiff.value!(tu, y)
    ReverseDiff.value!(tp, p)
    ReverseDiff.value!(tt, [t])
    ReverseDiff.forward_pass!(tapei)
    if StochasticDiffEq.is_diagonal_noise(prob)
      ReverseDiff.increment_deriv!(output, λi)
    else
      ReverseDiff.increment_deriv!(output, λ)
    end
    ReverseDiff.reverse_pass!(tapei)

    deriv = ReverseDiff.deriv(tp)
    dgrad[:,i] .= vec(deriv)
    ReverseDiff.pull_value!(output)

    if StochasticDiffEq.is_diagonal_noise(prob)
      dλ !== nothing && (dλ[:,i] .= vec(ReverseDiff.deriv(tu)))
      dy !== nothing && (dy[i] = ReverseDiff.value(output))
    else
      dλ !== nothing && (dλ[:,i] .= vec(ReverseDiff.deriv(tu)))
      dy !== nothing && (dy[:,i] .= vec(ReverseDiff.value(output)))
    end
  end
  return
end


function _jacNoise!(λ, y, p, t, S::SensitivityFunction, isnoise::ZygoteNoise, dgrad, dλ, dy)
  @unpack sensealg, f = S
  prob = getprob(S)

  if StochasticDiffEq.is_diagonal_noise(prob)
    if DiffEqBase.isinplace(prob)
      for (i, λi) in enumerate(λ)
        _dy, back = Zygote.pullback(y, prob.p) do u, p
          out_ = Zygote.Buffer(similar(u))
          f(out_, u, p, t)
          copy(out_[i])
        end
        tmp1,tmp2 = back(λi) #issue: tmp2 = zeros(p)
        dgrad[:,i] .= vec(tmp2)
        dλ !== nothing && (dλ[:,i] .= vec(tmp1))
        dy !== nothing && (dy[i] = _dy)
      end
    else
      for (i, λi) in enumerate(λ)
        _dy, back = Zygote.pullback(y, prob.p) do u, p
          f(u, p, t)[i]
        end
        tmp1,tmp2 = back(λi)
        dgrad[:,i] .= vec(tmp2)
        dλ !== nothing && (dλ[:,i] .= vec(tmp1))
        dy !== nothing && (dy[i] = _dy)
      end
    end
  else
    if DiffEqBase.isinplace(prob)
      for (i, λi) in enumerate(λ)
        _dy, back = Zygote.pullback(y, prob.p) do u, p
          out_ = Zygote.Buffer(similar(prob.noise_rate_prototype))
          f(out_, u, p, t)
          copy(out_[:,i])
        end
        tmp1,tmp2 = back(λ)#issue with Zygote.Buffer
        dgrad[:,i] .= vec(tmp2)
        dλ !== nothing && (dλ[:,i] .= vec(tmp1))
        dy !== nothing && (dy[:,i] .= vec(_dy))
      end
    else
      for (i, λi) in enumerate(λ)
         _dy, back = Zygote.pullback(y, prob.p) do u, p
          f(u, p, t)[:,i]
        end
        tmp1,tmp2 = back(λ)
        dgrad[:,i] .= vec(tmp2)
        dλ !== nothing && (dλ[:,i] .= vec(tmp1))
        dy !== nothing && (dy[:,i] .= vec(_dy))
      end
    end
  end
  return
end


function accumulate_cost!(dλ, y, p, t, S::SensitivityFunction, dgrad=nothing)
  @unpack dg, dg_val, g, g_grad_config = S.diffcache
  if dg != nothing
    if !(dg isa Tuple)
      dg(dg_val,y,p,t)
      dλ .+= vec(dg_val)
    else
      dg[1](dg_val[1],y,p,t)
      dλ .+= vec(dg_val[1])
      if dgrad !== nothing
        dg[2](dg_val[2],y,p,t)
        dgrad .-= vec(dg_val[2])
      end
    end
  else
    g.t = t
    gradient!(dg_val, g, y, S.sensealg, g_grad_config)
    dλ .+= vec(dg_val)
  end
  return nothing
end

function build_jac_config(alg,uf,u)
  if alg_autodiff(alg)
    jac_config = ForwardDiff.JacobianConfig(uf,u,u,
                 ForwardDiff.Chunk{determine_chunksize(u,alg)}())
  else
    if diff_type(alg) != Val{:complex}
      jac_config = FiniteDiff.JacobianCache(similar(u),similar(u),
                                                 similar(u),diff_type(alg))
    else
      tmp = Complex{eltype(u)}.(u)
      du1 = Complex{eltype(u)}.(du1)
      jac_config = FiniteDiff.JacobianCache(tmp,du1,nothing,diff_type(alg))
    end
  end
  jac_config
end

function build_param_jac_config(alg,uf,u,p)
  if alg_autodiff(alg)
    jac_config = ForwardDiff.JacobianConfig(uf,u,p,
                 ForwardDiff.Chunk{determine_chunksize(p,alg)}())
  else
    if diff_type(alg) != Val{:complex}
      jac_config = FiniteDiff.JacobianCache(similar(p),similar(u),
                                                 similar(u),diff_type(alg))
    else
      tmp = Complex{eltype(p)}.(p)
      du1 = Complex{eltype(u)}.(u)
      jac_config = FiniteDiff.JacobianCache(tmp,du1,nothing,diff_type(alg))
    end
  end
  jac_config
end

function build_grad_config(alg,tf,du1,t)
  if alg_autodiff(alg)
    grad_config = ForwardDiff.GradientConfig(tf,du1,
                                    ForwardDiff.Chunk{determine_chunksize(du1,alg)}())
  else
    grad_config = FiniteDiff.GradientCache(du1,t,diff_type(alg))
  end
  grad_config
end
