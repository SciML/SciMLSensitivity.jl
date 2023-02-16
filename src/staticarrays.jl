### Projecting a tuple to SMatrix leads to ChainRulesCore._projection_mismatch by default, so overloaded here
function (project::ProjectTo{<:Tangent{<:Tuple}})(dx::StaticArraysCore.SArray)
    dy = reshape(dx, axes(project.elements))  # allows for dx::OffsetArray
    dz = ntuple(i -> project.elements[i](dy[i]), length(project.elements))
    return project_type(project)(dz...)
end

### Project SArray to SArray
function ProjectTo(x::StaticArraysCore.SArray{S, T}) where {S, T}
    return ProjectTo{StaticArraysCore.SArray}(; element = _eltype_projectto(T), axes = S)
end

function (project::ProjectTo{StaticArraysCore.SArray})(dx::AbstractArray{S, M}) where {S, M}
    return StaticArraysCore.SArray{project.axes}(dx)
end

### Adjoint for SArray constructor

function rrule(::Type{T}, x::Tuple) where {T <: StaticArraysCore.SArray}
    project_x = ProjectTo(x)
    Array_pullback(ȳ) = (NoTangent(), project_x(ȳ))
    return T(x), Array_pullback
end
