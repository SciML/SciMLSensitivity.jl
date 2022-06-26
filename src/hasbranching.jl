const printbranch = false

Cassette.@context HasBranchingCtx

function Cassette.overdub(ctx::HasBranchingCtx, f, args...)
    if Cassette.canrecurse(ctx, f, args...)
        return Cassette.recurse(ctx, f, args...)
    else
        return Cassette.fallback(ctx, f, args...)
    end
end

for (mod, f, n) in DiffRules.diffrules()
    isdefined(@__MODULE__, mod) || continue
    @eval function Cassette.overdub(::HasBranchingCtx, f::Core.Typeof($mod.$f),
                                    x::Vararg{Any, $n})
        f(x...)
    end
end

function _pass(::Type{<:HasBranchingCtx}, reflection::Cassette.Reflection)
    ir = reflection.code_info

    if any(x -> isa(x, GotoIfNot), ir.code)
        printbranch && println("GotoIfNot detected in $(reflection.method)\nir = $ir\n")
        Cassette.insert_statements!(ir.code, ir.codelocs,
                                    (stmt, i) -> i == 1 ? 3 : nothing,
                                    (stmt, i) -> Any[Expr(:call,
                                                          Expr(:nooverdub,
                                                               GlobalRef(Base, :getfield)),
                                                          Expr(:contextslot),
                                                          QuoteNode(:metadata)),
                                                     Expr(:call,
                                                          Expr(:nooverdub,
                                                               GlobalRef(Base, :setindex!)),
                                                          SSAValue(1), true,
                                                          QuoteNode(:has_branching)),
                                                     stmt])
        Cassette.insert_statements!(ir.code, ir.codelocs,
                                    (stmt, i) -> i > 2 && isa(stmt, Expr) ? 1 : nothing,
                                    (stmt, i) -> begin
                                        callstmt = Meta.isexpr(stmt, :(=)) ? stmt.args[2] :
                                                   stmt
                                        Meta.isexpr(stmt, :call) ||
                                            Meta.isexpr(stmt, :invoke) || return Any[stmt]
                                        callstmt = Expr(callstmt.head,
                                                        Expr(:nooverdub, callstmt.args[1]),
                                                        callstmt.args[2:end]...)
                                        return Any[Meta.isexpr(stmt, :(=)) ?
                                                   Expr(:(=), stmt.args[1], callstmt) :
                                                   callstmt]
                                    end)
    end
    return ir
end

const pass = Cassette.@pass _pass

function hasbranching(f, x...)
    metadata = Dict(:has_branching => false)
    Cassette.overdub(Cassette.disablehooks(HasBranchingCtx(; pass, metadata)), f, x...)
    return metadata[:has_branching]
end

Cassette.overdub(::HasBranchingCtx, ::typeof(+), x...) = +(x...)
Cassette.overdub(::HasBranchingCtx, ::typeof(*), x...) = *(x...)
function Cassette.overdub(::HasBranchingCtx, ::typeof(Base.materialize), x...)
    Base.materialize(x...)
end
function Cassette.overdub(::HasBranchingCtx, ::typeof(Base.literal_pow), x...)
    Base.literal_pow(x...)
end
Cassette.overdub(::HasBranchingCtx, ::typeof(Base.getindex), x...) = Base.getindex(x...)
Cassette.overdub(::HasBranchingCtx, ::typeof(Core.Typeof), x...) = Core.Typeof(x...)
function Cassette.overdub(::HasBranchingCtx, ::Type{Base.OneTo{T}},
                          stop) where {T <: Integer}
    Base.OneTo{T}(stop)
end
