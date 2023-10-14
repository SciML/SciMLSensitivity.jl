struct SteadyStateAdjointSensitivityFunction{C <: AdjointDiffCache,
    Alg <: SteadyStateAdjoint, uType, SType, fType <: ODEFunction, CV, λType, VJPType,
    LS} <: SensitivityFunction
    diffcache::C
    sensealg::Alg
    y::uType
    sol::SType
    f::fType
    colorvec::CV
    λ::λType
    vjp::VJPType
    linsolve::LS
end

TruncatedStacktraces.@truncate_stacktrace SteadyStateAdjointSensitivityFunction

function SteadyStateAdjointSensitivityFunction(g, sensealg, alg, sol, dgdu, dgdp, f,
    colorvec, needs_jac)
    @unpack p, u0 = sol.prob

    diffcache, y = adjointdiffcache(g, sensealg, false, sol, dgdu, dgdp, f, alg;
        quad = false, needs_jac)

    λ = zero(y)
    linsolve = needs_jac ? nothing : sensealg.linsolve
    vjp = allocate_vjp(λ, p)

    return SteadyStateAdjointSensitivityFunction(diffcache, sensealg, y, sol, f, colorvec,
        λ, vjp, linsolve)
end

@noinline function SteadyStateAdjointProblem(sol, sensealg::SteadyStateAdjoint, alg,
    dgdu::DG1 = nothing, dgdp::DG2 = nothing,
    g::G = nothing; kwargs...) where {DG1, DG2, G}
    @unpack f, p, u0 = sol.prob

    if sol.prob isa NonlinearProblem
        f = ODEFunction(f)
    end

    dgdu === nothing && dgdp === nothing && g === nothing &&
        error("Either `dgdu`, `dgdp`, or `g` must be specified.")

    needs_jac = if has_adjoint(f)
        false
        # TODO: What is the correct heuristic? Can we afford to compute Jacobian for
        #       cases where the length(u0) > 50 and if yes till what threshold
    elseif sensealg.linsolve === nothing
        length(u0) ≤ 50
    else
        LinearSolve.needs_concrete_A(sensealg.linsolve)
    end

    p === DiffEqBase.NullParameters() &&
        error("Your model does not have parameters, and thus it is impossible to calculate the derivative of the solution with respect to the parameters. Your model must have parameters to use parameter sensitivity calculations!")

    sense = SteadyStateAdjointSensitivityFunction(g, sensealg, alg, sol, dgdu, dgdp,
        f, f.colorvec, needs_jac)
    @unpack diffcache, y, sol, λ, vjp, linsolve = sense

    if needs_jac
        if DiffEqBase.has_jac(f)
            f.jac(diffcache.J, y, p, nothing)
        else
            if DiffEqBase.isinplace(sol.prob)
                jacobian!(diffcache.J, diffcache.uf, y, diffcache.f_cache,
                    sensealg, diffcache.jac_config)
            else
                diffcache.J .= jacobian(diffcache.uf, y, sensealg)
            end
        end
    end

    if dgdp === nothing && g === nothing
        dgdu_val = diffcache.dg_val
        dgdp_val = nothing
    else
        dgdu_val, dgdp_val = diffcache.dg_val
    end

    if dgdu !== nothing
        dgdu(dgdu_val, y, p, nothing, nothing)
    else
        if g !== nothing
            if dgdp_val !== nothing
                gradient!(vec(dgdu_val), diffcache.g[1], y, sensealg,
                    diffcache.g_grad_config[1])
            else
                gradient!(vec(dgdu_val), diffcache.g, y, sensealg, diffcache.g_grad_config)
            end
        end
    end

    if !needs_jac
        __f = y -> f(y, p, nothing)
        operator = VecJac(__f, y; autodiff = get_autodiff_from_vjp(sensealg.autojacvec))
        linear_problem = LinearProblem(operator, vec(dgdu_val); u0 = vec(λ))
    else
        linear_problem = LinearProblem(diffcache.J', vec(dgdu_val'); u0 = vec(λ))
    end

    # Zygote pullback function won't work with deepcopy
    solve(linear_problem, linsolve; alias_A = true, sensealg.linsolve_kwargs...) # u is vec(λ)

    try
        vecjacobian!(vec(dgdu_val), y, λ, p, nothing, sense; dgrad = vjp, dy = nothing)
    catch e
        if sense.sensealg.autojacvec === nothing
            @warn "Automatic AD choice of autojacvec failed in nonlinear solve adjoint, failing back to ODE adjoint + numerical vjp"
            vecjacobian!(vec(dgdu_val), y, λ, p, nothing, false, dgrad = vjp, dy = nothing)
        else
            @warn "AD choice of autojacvec failed in nonlinear solve adjoint"
            throw(e)
        end
    end

    if g !== nothing || dgdp !== nothing
        # compute del g/del p
        if dgdp !== nothing
            dgdp(dgdp_val, y, p, nothing, nothing)
        else
            @unpack g_grad_config = diffcache
            gradient!(dgdp_val, diffcache.g[2], p, sensealg, g_grad_config[2])
        end
        recursive_sub!(dgdp_val, vjp)
        return dgdp_val
    else
        recursive_neg!(vjp)
        return vjp
    end
end
