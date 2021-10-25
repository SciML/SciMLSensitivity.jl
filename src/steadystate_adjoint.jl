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
    needs_jac,
)
    @unpack f, p, u0 = sol.prob

    diffcache, y = adjointdiffcache(
        g,
        sensealg,
        discrete,
        sol,
        dg,
        f;
        quad = false,
        needs_jac = needs_jac,
    )

    λ = zero(y)
    linsolve =
        needs_jac ? nothing : sensealg.linsolve(Val{:init}, diffcache.uf, y)
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
    @unpack f, p, u0 = sol.prob

    discrete = false
    # TODO: What is the correct heuristic? Can we afford to compute Jacobian for
    #       cases where the length(u0) > 100 and if yes till what threshold
    needs_jac = length(u0) <= 100

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
        needs_jac,
    )
    @unpack diffcache, y, sol, λ, vjp, linsolve = sense

    if needs_jac
        if DiffEqBase.has_jac(f)
            f.jac(diffcache.J, y, p, nothing)
        else
            if DiffEqBase.isinplace(sol.prob)
                jacobian!(
                    diffcache.J,
                    diffcache.uf,
                    y,
                    diffcache.f_cache,
                    sensealg,
                    diffcache.jac_config,
                )
            else
                temp = jacobian(diffcache.uf, y, sensealg)
                @. diffcache.J = temp
            end
        end
    end

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

    if !needs_jac
        # FIXME: Won't work if the matrix A is explicitly required for solving
        #        the linear system.
        linsolve(
            vec(λ),
            VecJacOperator(f, y, p; autodiff = true),
            vec(diffcache.dg_val),
        )
    else
        copyto!(vec(λ), diffcache.J' \ vec(diffcache.dg_val'))
    end

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

        @. dg_dp_val = dg_dp_val - vjp
        return dg_dp_val
    else
        return -vjp
    end
end
