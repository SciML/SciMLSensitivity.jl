struct SensitivityAlg{CS,AD,FDT} <: DiffEqBase.DEAlgorithm end
Base.@pure function SensitivityAlg(;chunk_size=0,autodiff=true,diff_type=Val{:central})
  SensitivityAlg{chunk_size,autodiff,typeof(diff_type)}()
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

Base.@pure alg_autodiff(alg::SensitivityAlg{CS,AD,FDT}) where {CS,AD,FDT} = AD
Base.@pure get_chunksize(alg::SensitivityAlg{CS,AD,FDT}) where {CS,AD,FDT} = CS
Base.@pure diff_type(alg::SensitivityAlg{CS,AD,FDT}) where {CS,AD,FDT} = FDT

function jacobian!(J::AbstractMatrix{<:Number}, f, x::AbstractArray{<:Number},
                   fx::AbstractArray{<:Number}, alg::SensitivityAlg, jac_config)
    if alg_autodiff(alg)
      ForwardDiff.jacobian!(J, f, fx, x, jac_config)
    else
      DiffEqDiffTools.finite_difference_jacobian!(J, f, x, jac_config)
    end
    nothing
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
                 ForwardDiff.Chunk{determine_chunksize(u,alg)}())
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
