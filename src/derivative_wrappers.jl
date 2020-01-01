# Not in DiffEqDiffTools because `u` -> scalar isn't used anywhere else,
# but could be upstreamed.
mutable struct UGradientWrapper{fType,tType,P} <: Function
  f::fType
  t::tType
  p::P
end

(ff::UGradientWrapper)(uprev) = ff.f(uprev,ff.p,ff.t)

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

function jacobian!(J::AbstractMatrix{<:Number}, f, x::AbstractArray{<:Number},
                   fx::AbstractArray{<:Number}, alg::DiffEqBase.AbstractSensitivityAlgorithm, jac_config)
  if alg_autodiff(alg)
    ForwardDiff.jacobian!(J, f, fx, x, jac_config)
  else
    DiffEqDiffTools.finite_difference_jacobian!(J, f, x, jac_config)
  end
  nothing
end

function gradient!(df::AbstractArray{<:Number}, f,
                   x::Union{Number,AbstractArray{<:Number}},
                   alg::DiffEqBase.AbstractSensitivityAlgorithm, grad_config)
    if alg_autodiff(alg)
        ForwardDiff.gradient!(df, f, x, grad_config)
    else
        DiffEqDiffTools.finite_difference_gradient!(df, f, x, grad_config)
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

function vecjacobian!(dλ, λ, p, t, S; dgrad=nothing, dy=nothing)
  @unpack y, sol, sensealg = S
  f = sol.prob.f
  isautojacvec = get_jacvec(sensealg)
  if !isautojacvec
    @unpack J, uf, f_cache, jac_config = S
    if DiffEqBase.has_jac(f)
      f.jac(J,y,p,t) # Calculate the Jacobian into J
    else
      uf.t = t
      jacobian!(J, uf, y, f_cache, sensealg, jac_config)
    end
    mul!(dλ',λ',J)

    if dgrad !== nothing
      @unpack pJ, pf, paramjac_config = S
      if DiffEqBase.has_paramjac(f)
        f.paramjac(pJ,y,sol.prob.p,t) # Calculate the parameter Jacobian into pJ
      else
        jacobian!(pJ, pf, sol.prob.p, f_cache, sensealg, paramjac_config)
      end
      mul!(dgrad',λ',pJ)
    end
    dy !== nothing && f(dy, y, p, t)
  else
    if DiffEqBase.isinplace(sol.prob)
      _dy, back = Tracker.forward(y, sol.prob.p) do u, p
        out_ = map(zero, u)
        f(out_, u, p, t)
        Tracker.collect(out_)
      end
      tmp1, tmp2 = Tracker.data.(back(λ))
      dλ[:] = tmp1
      dgrad !== nothing && (dgrad[:] = tmp2)
      dy !== nothing && (dy[:] = vec(Tracker.data(_dy)))
    elseif !(sol.prob.p isa Zygote.Params)
      _dy, back = Zygote.pullback(y, sol.prob.p) do u, p
        vec(f(u, p, t))
      end
      tmp1,tmp2 = back(λ)
      dλ[:] .= tmp1
      dgrad !== nothing && (dgrad[:] .= tmp2)
      dy !== nothing && (dy[:] .= vec(_dy))
    else # Not in-place and p is a Params

      # This is the hackiest hack of the west specifically to get Zygote
      # Implicit parameters to work. This should go away ASAP!

      _dy, back = Zygote.pullback(y, S.sol.prob.p) do u, p
        vec(f(u, p, t))
      end

      _idy, iback = Zygote.pullback(S.sol.prob.p) do
        vec(f(y, p, t))
      end

      igs = iback(λ)
      vs = zeros(Float32, sum(length.(S.sol.prob.p)))
      i = 1
      for p in S.sol.prob.p
        g = igs[p]
        g isa AbstractArray || continue
        vs[i:i+length(g)-1] = g
        i += length(g)
      end
      eback = back(λ)
      dλ[:] = eback[1]
      dgrad !== nothing && (dgrad[:] = vec(vs))
      dy !== nothing && (dy[:] = vec(_dy))
    end
  end
  return nothing
end

function build_jac_config(alg,uf,u)
  if alg_autodiff(alg)
    jac_config = ForwardDiff.JacobianConfig(uf,u,u,
                 ForwardDiff.Chunk{determine_chunksize(u,alg)}())
  else
    if diff_type(alg) != Val{:complex}
      jac_config = DiffEqDiffTools.JacobianCache(similar(u),similar(u),
                                                 similar(u),diff_type(alg))
    else
      tmp = Complex{eltype(u)}.(u)
      du1 = Complex{eltype(u)}.(du1)
      jac_config = DiffEqDiffTools.JacobianCache(tmp,du1,nothing,diff_type(alg))
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
      jac_config = DiffEqDiffTools.JacobianCache(similar(p),similar(u),
                                                 similar(u),diff_type(alg))
    else
      tmp = Complex{eltype(p)}.(p)
      du1 = Complex{eltype(u)}.(u)
      jac_config = DiffEqDiffTools.JacobianCache(tmp,du1,nothing,diff_type(alg))
    end
  end
  jac_config
end

function build_grad_config(alg,tf,du1,t)
  if alg_autodiff(alg)
    grad_config = ForwardDiff.GradientConfig(tf,du1,
                                    ForwardDiff.Chunk{determine_chunksize(du1,alg)}())
  else
    grad_config = DiffEqDiffTools.GradientCache(du1,t,diff_type(alg))
  end
  grad_config
end
