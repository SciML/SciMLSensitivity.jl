module ZygoteExt
# Piracy that used to be requires, allowing Zyogote.jl to be specialized for SciML
using ZygoteRules, Zygote, SciMLBase

function ∇tmap(cx, f, args...)
    ys_and_backs = SciMLBase.tmap((args...) -> Zygote._pullback(cx, f, args...), args...)
    if isempty(ys_and_backs)
        ys_and_backs, _ -> (NoTangent(), NoTangent())
    else
        ys, backs = Zygote.unzip(ys_and_backs)
        function ∇tmap_internal(Δ)
            Δf_and_args_zipped = SciMLBase.tmap((f, δ) -> f(δ), backs, Δ)
            Δf_and_args = Zygote.unzip(Δf_and_args_zipped)
            Δf = reduce(Zygote.accum, Δf_and_args[1])
            (Δf, Δf_and_args[2:end]...)
        end
        ys, ∇tmap_internal
    end
end

function ∇responsible_map(cx, f, args...)
    ys_and_backs = SciMLBase.responsible_map((args...) -> Zygote._pullback(cx, f, args...),
                                             args...)
    if isempty(ys_and_backs)
        ys_and_backs, _ -> (NoTangent(), NoTangent())
    else
        ys, backs = Zygote.unzip(ys_and_backs)
        ys,
        function ∇responsible_map_internal(Δ)
            # Apply pullbacks in reverse order. Needed for correctness if `f` is stateful.
            Δf_and_args_zipped = SciMLBase.responsible_map((f, δ) -> f(δ),
                                                           Zygote._tryreverse(SciMLBase.responsible_map,
                                                                              backs, Δ)...)
            Δf_and_args = Zygote.unzip(Zygote._tryreverse(SciMLBase.responsible_map,
                                                          Δf_and_args_zipped))
            Δf = reduce(Zygote.accum, Δf_and_args[1])
            (Δf, Δf_and_args[2:end]...)
        end
    end
end

ZygoteRules.@adjoint function SciMLBase.tmap(f, args::Union{AbstractArray, Tuple}...)
    ∇tmap(__context__, f, args...)
end

ZygoteRules.@adjoint function SciMLBase.responsible_map(f,
                                                        args::Union{AbstractArray, Tuple
                                                                    }...)
    ∇responsible_map(__context__, f, args...)
end
end
