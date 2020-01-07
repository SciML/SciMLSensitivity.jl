SensitivityAlg(args...;kwargs...) = @error("The SensitivtyAlg choice mechanism was completely overhauled. Please consult the local sensitivity documentation for more information")

abstract type AbstractForwardSensitivityAlgorithm{CS,AD,FDT} <: DiffEqBase.AbstractSensitivityAlgorithm{CS,AD,FDT} end
abstract type AbstractAdjointSensitivityAlgorithm{CS,AD,FDT} <: DiffEqBase.AbstractSensitivityAlgorithm{CS,AD,FDT} end

struct ForwardSensitivity{CS,AD,FDT} <: AbstractForwardSensitivityAlgorithm{CS,AD,FDT}
  autojacvec::Bool
end
Base.@pure function ForwardSensitivity(;
                                       chunk_size=0,autodiff=true,
                                       diff_type=Val{:central},
                                       autojacvec=autodiff)
  ForwardSensitivity{chunk_size,autodiff,diff_type}(autojacvec)
end

struct ForwardDiffSensitivity{CS,CTS} <: AbstractForwardSensitivityAlgorithm{CS,Nothing,Nothing}
end
Base.@pure function ForwardDiffSensitivity(;chunk_size=0,convert_tspan=true)
  ForwardDiffSensitivity{chunk_size,convert_tspan}()
end

struct BacksolveAdjoint{CS,AD,FDT} <: AbstractAdjointSensitivityAlgorithm{CS,AD,FDT}
  autojacvec::Bool
  checkpointing::Bool
end
Base.@pure function BacksolveAdjoint(;chunk_size=0,autodiff=true,
                                      diff_type=Val{:central},
                                      autojacvec=autodiff,
                                      checkpointing=true)
  BacksolveAdjoint{chunk_size,autodiff,diff_type}(autojacvec,checkpointing)
end

struct InterpolatingAdjoint{CS,AD,FDT} <: AbstractAdjointSensitivityAlgorithm{CS,AD,FDT}
  autojacvec::Bool
  checkpointing::Bool
end
Base.@pure function InterpolatingAdjoint(;chunk_size=0,autodiff=true,
                                         diff_type=Val{:central},
                                         autojacvec=autodiff,
                                         checkpointing=false)
  InterpolatingAdjoint{chunk_size,autodiff,diff_type}(autojacvec,checkpointing)
end

struct QuadratureAdjoint{CS,AD,FDT} <: AbstractAdjointSensitivityAlgorithm{CS,AD,FDT}
  autojacvec::Bool
end
Base.@pure function QuadratureAdjoint(;chunk_size=0,autodiff=true,
                                         diff_type=Val{:central},
                                         autojacvec=autodiff)
  QuadratureAdjoint{chunk_size,autodiff,diff_type}(autojacvec)
end

struct TrackerAdjoint <: AbstractAdjointSensitivityAlgorithm{nothing,true,nothing} end
struct ZygoteAdjoint <: AbstractAdjointSensitivityAlgorithm{nothing,true,nothing} end

@inline convert_tspan(::ForwardDiffSensitivity{CS,CTS}) where {CS,CTS} = CTS
@inline alg_autodiff(alg::DiffEqBase.AbstractSensitivityAlgorithm{CS,AD,FDT}) where {CS,AD,FDT} = AD
@inline get_chunksize(alg::DiffEqBase.AbstractSensitivityAlgorithm{CS,AD,FDT}) where {CS,AD,FDT} = CS
@inline diff_type(alg::DiffEqBase.AbstractSensitivityAlgorithm{CS,AD,FDT}) where {CS,AD,FDT} = FDT
@inline get_jacvec(alg::DiffEqBase.AbstractSensitivityAlgorithm) = alg.autojacvec
@inline ischeckpointing(alg::DiffEqBase.AbstractSensitivityAlgorithm{CS,AD,FDT}) where {CS,AD,FDT} = isdefined(alg, :checkpointing) ? alg.checkpointing : false
