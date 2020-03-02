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

struct BacksolveAdjoint{CS,AD,FDT,VJP} <: AbstractAdjointSensitivityAlgorithm{CS,AD,FDT}
  autojacvec::VJP
  checkpointing::Bool
end
Base.@pure function BacksolveAdjoint(;chunk_size=0,autodiff=true,
                                      diff_type=Val{:central},
                                      autojacvec=autodiff,
                                      checkpointing=true)
  BacksolveAdjoint{chunk_size,autodiff,diff_type,typeof(autojacvec)}(autojacvec,checkpointing)
end

struct InterpolatingAdjoint{CS,AD,FDT,VJP} <: AbstractAdjointSensitivityAlgorithm{CS,AD,FDT}
  autojacvec::VJP
  checkpointing::Bool
end
Base.@pure function InterpolatingAdjoint(;chunk_size=0,autodiff=true,
                                         diff_type=Val{:central},
                                         autojacvec=autodiff,
                                         checkpointing=false)
  InterpolatingAdjoint{chunk_size,autodiff,diff_type,typeof(autojacvec)}(autojacvec,checkpointing)
end

struct QuadratureAdjoint{CS,AD,FDT,VJP} <: AbstractAdjointSensitivityAlgorithm{CS,AD,FDT}
  autojacvec::VJP
  abstol::Float64
  reltol::Float64
end
Base.@pure function QuadratureAdjoint(;chunk_size=0,autodiff=true,
                                         diff_type=Val{:central},
                                         autojacvec=autodiff,abstol=1e-6,
                                         reltol=1e-3)
  QuadratureAdjoint{chunk_size,autodiff,diff_type,typeof(autojacvec)}(autojacvec,abstol,reltol)
end

struct TrackerAdjoint <: AbstractAdjointSensitivityAlgorithm{nothing,true,nothing} end
struct ZygoteAdjoint <: AbstractAdjointSensitivityAlgorithm{nothing,true,nothing} end

abstract type VJPChoice end
struct ZygoteVJP <: VJPChoice end
struct TrackerVJP <: VJPChoice end
struct ReverseDiffVJP <: VJPChoice end

@inline convert_tspan(::ForwardDiffSensitivity{CS,CTS}) where {CS,CTS} = CTS
@inline alg_autodiff(alg::DiffEqBase.AbstractSensitivityAlgorithm{CS,AD,FDT}) where {CS,AD,FDT} = AD
@inline get_chunksize(alg::DiffEqBase.AbstractSensitivityAlgorithm{CS,AD,FDT}) where {CS,AD,FDT} = CS
@inline diff_type(alg::DiffEqBase.AbstractSensitivityAlgorithm{CS,AD,FDT}) where {CS,AD,FDT} = FDT
@inline function get_jacvec(alg::DiffEqBase.AbstractSensitivityAlgorithm)
  alg.autojacvec isa Bool ? alg.autojacvec : true
end
@inline ischeckpointing(alg::DiffEqBase.AbstractSensitivityAlgorithm, ::Vararg) = isdefined(alg, :checkpointing) ? alg.checkpointing : false
@inline ischeckpointing(alg::InterpolatingAdjoint, sol) = alg.checkpointing || !sol.dense
