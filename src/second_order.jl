function _second_order_sensitivities(loss, prob, alg, sensealg::ForwardDiffOverAdjoint,
        args...; kwargs...)
    ForwardDiff.jacobian(prob.p) do p
        x = Zygote.gradient(p) do _p
            loss(solve(prob, alg, args...; p = _p, sensealg = sensealg.adjalg, kwargs...))
        end
        first(x)
    end
end

function _second_order_sensitivity_product(loss, v, prob, alg,
        sensealg::ForwardDiffOverAdjoint,
        args...; kwargs...)
    θ = ForwardDiff.Dual.(prob.p, v)
    _loss = p -> loss(solve(prob, alg, args...; p = p, sensealg = sensealg.adjalg,
        kwargs...))
    getindex.(ForwardDiff.partials.(Zygote.gradient(_loss, θ)[1]), 1)
end
