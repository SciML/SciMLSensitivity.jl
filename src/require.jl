isgpu(x) = false
function __init__()
    @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
        gpu_or_cpu(x::CuArrays.CuArray) = CuArrays.CuArray
        gpu_or_cpu(x::TrackedArray{<:Any,<:Any,<:CuArrays.CuArray}) = CuArrays.CuArray
        gpu_or_cpu(x::Transpose{<:Any,<:CuArrays.CuArray}) = CuArrays.CuArray
        gpu_or_cpu(x::Adjoint{<:Any,<:CuArrays.CuArray}) = CuArrays.CuArray
        gpu_or_cpu(x::Adjoint{<:Any,TrackedArray{<:Any,<:Any,<:CuArrays.CuArray}}) = CuArrays.CuArray
        gpu_or_cpu(x::Transpose{<:Any,TrackedArray{<:Any,<:Any,<:CuArrays.CuArray}}) = CuArrays.CuArray
        isgpu(::CuArrays.CuArray) = true
        isgpu(::TrackedArray{<:Any,<:Any,<:CuArrays.CuArray}) = true
        isgpu(::Transpose{<:Any,<:CuArrays.CuArray}) = true
        isgpu(::Adjoint{<:Any,<:CuArrays.CuArray}) = true
        isgpu(::Adjoint{<:Any,TrackedArray{<:Any,<:Any,<:CuArrays.CuArray}}) = true
        isgpu(::Transpose{<:Any,TrackedArray{<:Any,<:Any,<:CuArrays.CuArray}}) = true
    end

    @require CUDA="3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
        gpu_or_cpu(x::CUDA.CuArray) = CUDA.CuArray
        gpu_or_cpu(x::TrackedArray{<:Any,<:Any,<:CUDA.CuArray}) = CUDA.CuArray
        gpu_or_cpu(x::Transpose{<:Any,<:CUDA.CuArray}) = CUDA.CuArray
        gpu_or_cpu(x::Adjoint{<:Any,<:CUDA.CuArray}) = CUDA.CuArray
        gpu_or_cpu(x::Adjoint{<:Any,TrackedArray{<:Any,<:Any,<:CUDA.CuArray}}) = CUDA.CuArray
        gpu_or_cpu(x::Transpose{<:Any,TrackedArray{<:Any,<:Any,<:CUDA.CuArray}}) = CUDA.CuArray
        isgpu(::CUDA.CuArray) = true
        isgpu(::TrackedArray{<:Any,<:Any,<:CUDA.CuArray}) = true
        isgpu(::Transpose{<:Any,<:CUDA.CuArray}) = true
        isgpu(::Adjoint{<:Any,<:CUDA.CuArray}) = true
        isgpu(::Adjoint{<:Any,TrackedArray{<:Any,<:Any,<:CUDA.CuArray}}) = true
        isgpu(::Transpose{<:Any,TrackedArray{<:Any,<:Any,<:CUDA.CuArray}}) = true
    end
end
