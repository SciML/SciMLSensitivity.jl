### Projecting a tuple to SMatrix leads to ChainRulesCore._projection_mismatch by default, so overloaded here
function (project::ProjectTo{<:Tangent{<:Tuple}})(dx::StaticArrays.SArray)
    dy = reshape(dx, axes(project.elements))  # allows for dx::OffsetArray
    dz = ntuple(i -> project.elements[i](dy[i]), length(project.elements))
    return project_type(project)(dz...)
end

### Project SArray to SArray
function ProjectTo(x::StaticArrays.SArray{S, T}) where {S, T}
    return ProjectTo{StaticArrays.SArray}(; element = _eltype_projectto(T), axes = S)
end

function (project::ProjectTo{StaticArrays.SArray})(dx::AbstractArray{S, M}) where {S, M}
    return StaticArrays.SArray{project.axes}(dx)
end

### Adjoint for SArray constructor

function rrule(::Type{T}, x::Tuple) where {T <: StaticArrays.SArray}
    project_x = ProjectTo(x)
    Array_pullback(ȳ) = (NoTangent(), project_x(ȳ))
    return T(x), Array_pullback
end
