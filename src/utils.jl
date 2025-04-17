function isfunctor(x)
    !isempty(Functors.children(x))
end

function to_nt(s::T) where T
    NamedTuple{propertynames(s)}(map(x -> getproperty(s, x), propertynames(s)))
end
