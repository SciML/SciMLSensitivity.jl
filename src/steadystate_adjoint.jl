struct SteadyStateAdjointSensitivityFunction{
    C<:AdjointDiffCache,
    Alg<:SteadyStateAdjoint,
    uType,
    SType,
    fType<:ODEFunction,
    CV,
    λType,
    VJPType,
    LS,
} <: SensitivityFunction
    diffcache::C
    sensealg::Alg
    discrete::Bool
    y::uType
    sol::SType
    f::fType
    colorvec::CV
    λ::λType
    vjp::VJPType
    linsolve::LS
end

function SteadyStateAdjointSensitivityFunction(
    g,
    sensealg,
    discrete,
    sol,
    dg,
    colorvec,
)
    @unpack f, p, u0 = sol.prob

    diffcache, y =
        adjointdiffcache(g, sensealg, discrete, sol, dg, f; quad = false)

    λ = zero(y)
    linsolve = sensealg.linsolve(Val{:init}, diffcache.uf, y)
    vjp = similar(λ, length(p))

    SteadyStateAdjointSensitivityFunction(
        diffcache,
        sensealg,
        discrete,
        y,
        sol,
        f,
        colorvec,
        λ,
        vjp,
        linsolve,
    )
end

@noinline function SteadyStateAdjointProblem(
    sol,
    sensealg::SteadyStateAdjoint,
    g,
    dg;
    save_idxs = nothing,
)
    @unpack f, p = sol.prob

    discrete = false

    p === DiffEqBase.NullParameters() && error(
        "Your model does not have parameters, and thus it is impossible to calculate the derivative of the solution with respect to the parameters. Your model must have parameters to use parameter sensitivity calculations!",
    )

    sense = SteadyStateAdjointSensitivityFunction(
        g,
        sensealg,
        discrete,
        sol,
        dg,
        f.colorvec,
    )
    @unpack diffcache, y, sol, λ, vjp, linsolve = sense

    _save_idxs = save_idxs === nothing ? Colon() : save_idxs
    if dg !== nothing
        if g !== nothing
            dg(vec(diffcache.dg_val), y, p, nothing, nothing)
        else
            if typeof(_save_idxs) <: Number
                diffcache.dg_val[_save_idxs] = dg[_save_idxs]
            elseif typeof(dg) <: Number
                @. diffcache.dg_val[_save_idxs] = dg
            else
                @. diffcache.dg_val[_save_idxs] = dg[_save_idxs]
            end
        end
    else
        if g !== nothing
            gradient!(
                vec(diffcache.dg_val),
                diffcache.g,
                y,
                sensealg,
                diffcache.g_grad_config,
            )
        end
    end

    linsolve(
        vec(λ),
        Diagonal(one.(vec(y))) - VecJacOperator(f, y, p; autodiff = true),
        vec(diffcache.dg_val),
    )

    vecjacobian!(
        vec(diffcache.dg_val),
        y,
        λ,
        p,
        nothing,
        sense,
        dgrad = vjp,
        dy = nothing,
    )

    if g !== nothing
        # compute del g/del p
        dg_dp_val = zero(p)
        dg_dp = ParamGradientWrapper(g, nothing, y)
        dg_dp_config = build_grad_config(sensealg, dg_dp, p, p)
        gradient!(dg_dp_val, dg_dp, p, sensealg, dg_dp_config)

        @. dg_dp_val = dg_dp_val + vjp
        return dg_dp_val
    else
        return vjp
    end
end
