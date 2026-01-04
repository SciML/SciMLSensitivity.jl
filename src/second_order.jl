function _second_order_sensitivities(
        loss, prob, alg, sensealg::ForwardDiffOverAdjoint,
        args...; kwargs...
    )
    return ForwardDiff.jacobian(prob.p) do p
        x = Zygote.gradient(p) do _p
            loss(solve(prob, alg, args...; p = _p, sensealg = sensealg.adjalg, kwargs...))
        end
        first(x)
    end
end

struct SciMLSensitivityTag end

function _second_order_sensitivity_product(
        loss, v, prob, alg,
        sensealg::ForwardDiffOverAdjoint,
        args...; kwargs...
    )
    T = typeof(ForwardDiff.Tag(SciMLSensitivityTag(), eltype(v)))
    θ = ForwardDiff.Dual{T, eltype(v), 1}.(prob.p, ForwardDiff.Partials.(Tuple.(v)))
    _loss = p -> loss(
        solve(
            prob, alg, args...; p, sensealg = sensealg.adjalg, kwargs...
        )
    )
    return getindex.(ForwardDiff.partials.(Zygote.gradient(_loss, θ)[1]), 1)
end
