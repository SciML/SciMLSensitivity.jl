struct SensitivityAlg{CS,AD,FDT} <: DiffEqBase.AbstractSensitivityAlgorithm{CS,AD,FDT}
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

struct ForwardSensitivity{CS,AD,FDT} <: DiffEqBase.AbstractSensitivityAlgorithm{CS,AD,FDT}
  autojacvec::Bool
end
Base.@pure function ForwardSensitivity(;
                                       chunk_size=0,autodiff=true,
                                       diff_type=Val{:central},
                                       autojacvec=autodiff)
  ForwardSensitivity{chunk_size,autodiff,diff_type}(autojacvec)
end

struct ForwardDiffSensitivity{CS,CTS} <: DiffEqBase.AbstractSensitivityAlgorithm{CS,Nothing,Nothing}
end
Base.@pure function ForwardDiffSensitivity(;chunk_size=0,convert_tspan=true)
  ForwardDiffSensitivity{chunk_size,convert_tspan}()
end

@inline convert_tspan(::ForwardDiffSensitivity{CS,CTS}) where {CS,CTS} = CTS
@inline alg_autodiff(alg::DiffEqBase.AbstractSensitivityAlgorithm{CS,AD,FDT}) where {CS,AD,FDT} = AD
@inline get_chunksize(alg::DiffEqBase.AbstractSensitivityAlgorithm{CS,AD,FDT}) where {CS,AD,FDT} = CS
@inline diff_type(alg::DiffEqBase.AbstractSensitivityAlgorithm{CS,AD,FDT}) where {CS,AD,FDT} = FDT
@inline get_jacvec(alg::DiffEqBase.AbstractSensitivityAlgorithm{CS,AD,FDT}) where {CS,AD,FDT} = alg.autojacvec
@inline isquad(alg::DiffEqBase.AbstractSensitivityAlgorithm{CS,AD,FDT}) where {CS,AD,FDT} = alg.quad
@inline isbcksol(alg::DiffEqBase.AbstractSensitivityAlgorithm{CS,AD,FDT}) where {CS,AD,FDT} = alg.backsolve
@inline ischeckpointing(alg::DiffEqBase.AbstractSensitivityAlgorithm{CS,AD,FDT}) where {CS,AD,FDT} = alg.checkpointing
