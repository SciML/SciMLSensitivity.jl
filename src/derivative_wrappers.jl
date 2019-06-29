struct SensitivityAlg{CS,AD,FDT} <: DiffEqBase.DEAlgorithm
  autojacvec::Bool
  quad::Bool
  backsolve::Bool
  checkpointing::Bool
end
Base.@pure function SensitivityAlg(;chunk_size=0,autodiff=true,diff_type=Val{:central},
                                   autojacvec=autodiff,quad=true,backsolve=false,checkpointing=false)
  checkpointing && (backsolve=false)
  backsolve && (quad = false)
  SensitivityAlg{chunk_size,autodiff,diff_type}(autojacvec,quad,backsolve,checkpointing)
end

# Not in DiffEqDiffTools because `u` -> scalar isn't used anywhere else,
# but could be upstreamed.
mutable struct UGradientWrapper{fType,tType,P} <: Function
  f::fType
  t::tType
  p::P
end

(ff::UGradientWrapper)(uprev) = ff.f(uprev,ff.p,ff.t)

Base.@pure function determine_chunksize(u,alg::SensitivityAlg)
  determine_chunksize(u,get_chunksize(alg))
end

Base.@pure function determine_chunksize(u,CS)
  if CS != 0
    return CS
  else
    return ForwardDiff.pickchunksize(length(u))
  end
end

@inline alg_autodiff(alg::SensitivityAlg{CS,AD,FDT}) where {CS,AD,FDT} = AD
@inline get_chunksize(alg::SensitivityAlg{CS,AD,FDT}) where {CS,AD,FDT} = CS
@inline diff_type(alg::SensitivityAlg{CS,AD,FDT}) where {CS,AD,FDT} = FDT
@inline get_jacvec(alg::SensitivityAlg{CS,AD,FDT}) where {CS,AD,FDT} = alg.autojacvec
@inline isquad(alg::SensitivityAlg{CS,AD,FDT}) where {CS,AD,FDT} = alg.quad
@inline isbcksol(alg::SensitivityAlg{CS,AD,FDT}) where {CS,AD,FDT} = alg.backsolve
@inline ischeckpointing(alg::SensitivityAlg{CS,AD,FDT}) where {CS,AD,FDT} = alg.checkpointing

function jacobian!(J::AbstractMatrix{<:Number}, f, x::AbstractArray{<:Number},
                   fx::AbstractArray{<:Number}, alg::SensitivityAlg, jac_config)
  if alg_autodiff(alg)
    ForwardDiff.jacobian!(J, f, fx, x, jac_config)
  else
    DiffEqDiffTools.finite_difference_jacobian!(J, f, x, jac_config)
  end
  nothing
end

function gradient!(df::AbstractArray{<:Number}, f,
                   x::Union{Number,AbstractArray{<:Number}},
                   alg::SensitivityAlg, grad_config)
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
                      v, alg::SensitivityAlg, config)
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
