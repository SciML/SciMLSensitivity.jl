module SciMLSensitivityMooncakeExt

using SciMLSensitivity, Mooncake
import SciMLSensitivity: get_paramjac_config, mooncake_run_ad, MooncakeVJP, MooncakeLoaded

function get_paramjac_config(::MooncakeLoaded, ::MooncakeVJP, pf, p, f, y, _t)
    dy_mem = zero(y)
    λ_mem = zero(y)
    cache = Mooncake.prepare_pullback_cache(pf, dy_mem, y, p, _t)
    return cache, pf, λ_mem, dy_mem
end

function mooncake_run_ad(paramjac_config::Tuple, y, p, t, λ)
    cache, pf, λ_mem, dy_mem = paramjac_config
    λ_mem .= λ
    dy, _ = Mooncake.value_and_pullback!!(cache, λ_mem, pf, dy_mem, y, p, t)
    y_grad = cache.tangents[3]
    p_grad = cache.tangents[4]
    return dy, y_grad, p_grad
end

end
