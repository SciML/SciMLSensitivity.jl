module SciMLSensitivityReactantExt

using SciMLSensitivity: SciMLSensitivity
using Reactant: Reactant, ConcreteRArray, @jit
using Enzyme: Enzyme
import SciMLSensitivity: get_paramjac_config, reactant_run_ad, ReactantVJP, ReactantLoaded

function get_paramjac_config(::ReactantLoaded, ::ReactantVJP, pf, p, f, y, _t)
    dy_mem = zero(y)
    λ_mem = zero(y)
    return pf, λ_mem, dy_mem
end

function _reactant_vjp_kernel(pf, dy_buf, u, p, t, λ)
    du_shadow = zero(u)
    dp_shadow = zero(p)
    dλ_seed = copy(λ)

    Enzyme.autodiff(
        Enzyme.Reverse,
        Enzyme.Const(pf),
        Enzyme.Const,
        Enzyme.Duplicated(dy_buf, dλ_seed),
        Enzyme.Duplicated(u, du_shadow),
        Enzyme.Duplicated(p, dp_shadow),
        Enzyme.Const(t)
    )

    return dy_buf, du_shadow, dp_shadow
end

function reactant_run_ad(paramjac_config, y, p, t, λ)
    pf, _λ_mem, _dy_mem = paramjac_config

    y_ra = ConcreteRArray(y)
    p_ra = ConcreteRArray(p)
    λ_ra = ConcreteRArray(λ)
    dy_ra = ConcreteRArray(zero(y))

    dy, y_grad, p_grad = @jit(
        _reactant_vjp_kernel(pf, dy_ra, y_ra, p_ra, t, λ_ra)
    )

    return Array(dy), Array(y_grad), Array(p_grad)
end

end # module
