function isfunctor(x)
    return !isempty(Functors.children(x))
end

function to_nt(s::T) where {T}
    return NamedTuple{propertynames(s)}(map(x -> getproperty(s, x), propertynames(s)))
end
