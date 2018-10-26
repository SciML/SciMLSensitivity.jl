struct SensitivityAlg{CS,AD,FDT,Jv} <: DiffEqBase.DEAlgorithm end
Base.@pure function SensitivityAlg(;chunk_size=0,autodiff=true,diff_type=Val{:central},autojacvec=true)
  SensitivityAlg{chunk_size,autodiff,typeof(diff_type),autojacvec}()
end

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

Base.@pure alg_autodiff(alg::SensitivityAlg{CS,AD,FDT,Jv}) where {CS,AD,FDT,Jv} = AD
Base.@pure get_chunksize(alg::SensitivityAlg{CS,AD,FDT,Jv}) where {CS,AD,FDT,Jv} = CS
Base.@pure diff_type(alg::SensitivityAlg{CS,AD,FDT,Jv}) where {CS,AD,FDT,Jv} = FDT
Base.@pure get_jacvec(alg::SensitivityAlg{CS,AD,FDT,Jv}) where {CS,AD,FDT,Jv} = Jv

function jacobian!(J::AbstractMatrix{<:Number}, f, x::AbstractArray{<:Number},
                   fx::AbstractArray{<:Number}, alg::SensitivityAlg, jac_config)
  if alg_autodiff(alg)
    ForwardDiff.jacobian!(J, f, fx, x, jac_config)
  else
    DiffEqDiffTools.finite_difference_jacobian!(J, f, x, jac_config)
  end
  nothing
end

"""
  jacobianvec!(Jv, f, x, v, alg, (buffer, seed)) -> nothing

``Jv <- J(f(x))v``
"""
function jacobianvec!(Jv::AbstractArray{<:Number}, f, x::AbstractArray{<:Number},
                      v, alg::SensitivityAlg, config)
  buffer, seed = config
  if alg_autodiff(alg)
    TD = typeof(first(seed))
    T  = typeof(first(seed).partials)
    @. seed = TD(x, T(tuple(v)))
    f(buffer, seed)
    Jv .= ForwardDiff.partials.(buffer, 1)
  else
      error("Jacobian*vector computation is for automatic differentiation only!")
  end
  nothing
end

"""
  vecjacobian!(vJ, v, f, x, alg, config) -> nothing

``Jv <- v'J(f(x))``
"""
function vecjacobian!(vJ::AbstractArray{<:Number}, v, f, x::AbstractArray{<:Number},
                      alg::SensitivityAlg, config)
  tp = ReverseDiff.InstructionTape()
  tx = ReverseDiff.track(x, tp)
  ty = f(tx)
  ReverseDiff.increment_deriv!(ty, v)
  ReverseDiff.reverse_pass!(tp)
  copyto!(vJ, ReverseDiff.deriv(tx))
  return nothing
end

function build_jac_config(alg,uf,u)
  if alg_autodiff(alg)
    jac_config = ForwardDiff.JacobianConfig(uf,u,u,
                 ForwardDiff.Chunk{determine_chunksize(u,alg)}())
  else
    if alg.diff_type != Val{:complex}
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
    if alg.diff_type != Val{:complex}
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
