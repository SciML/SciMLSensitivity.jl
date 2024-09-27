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

function SteadyStateAdjointSensitivityFunction(g, sensealg, alg, sol, dgdu, dgdp, f,
        colorvec, needs_jac)
    (; p, u0) = sol.prob

    diffcache, y = adjointdiffcache(g, sensealg, false, sol, dgdu, dgdp, f, alg;
        quad = false, needs_jac)

    λ = zero(y)
    linsolve = needs_jac ? nothing : sensealg.linsolve
    vjp = allocate_vjp(λ, p)

    return SteadyStateAdjointSensitivityFunction(diffcache, sensealg, y, sol, f, colorvec,
        λ, vjp, linsolve)
end

@inline __needs_concrete_A(l) = LinearSolve.needs_concrete_A(l)
@inline __needs_concrete_A(::Nothing) = false

@noinline function SteadyStateAdjointProblem(sol, sensealg::SteadyStateAdjoint, alg,
        dgdu::DG1 = nothing, dgdp::DG2 = nothing, g::G = nothing;
        kwargs...) where {DG1, DG2, G}
    (; f, p, u0) = sol.prob

    sol.prob isa AbstractNonlinearProblem && (f = ODEFunction(f))

    dgdu === nothing && dgdp === nothing && g === nothing &&
        error("Either `dgdu`, `dgdp`, or `g` must be specified.")

    needs_jac = ifelse(has_adjoint(f), false,
        ifelse(sensealg.linsolve === nothing, length(u0) ≤ 50,
            __needs_concrete_A(sensealg.linsolve)))

    p === SciMLBase.NullParameters() &&
        error("Your model does not have parameters, and thus it is impossible to calculate the derivative of the solution with respect to the parameters. Your model must have parameters to use parameter sensitivity calculations!")

    sense = SteadyStateAdjointSensitivityFunction(g, sensealg, alg, sol, dgdu, dgdp,
        f, f.colorvec, needs_jac)
    (; diffcache, y, sol, λ, vjp, linsolve) = sense

    if needs_jac
        if SciMLBase.has_jac(f)
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
        # Current SciMLJacobianOperators requires specifying the problem as a NonlinearProblem
        usize = size(y)
        if SciMLBase.isinplace(f)
            nlfunc = NonlinearFunction{true}((du, u, p) -> unwrapped_f(f)(
                reshape(u, usize), reshape(u, usize), p, nothing))
        else
            nlfunc = NonlinearFunction{false}((u, p) -> unwrapped_f(f)(
                reshape(u, usize), p, nothing))
        end
        nlprob = NonlinearProblem(nlfunc, vec(λ), p)
        operator = VecJacOperator(
            nlprob, vec(y), (λ); autodiff = get_autodiff_from_vjp(sensealg.autojacvec))
        soperator = StatefulJacobianOperator(operator, vec(λ), p)
        linear_problem = LinearProblem(soperator, vec(dgdu_val); u0 = vec(λ))
        solve(linear_problem, linsolve; alias_A = true, sensealg.linsolve_kwargs...)
    else
        if linsolve === nothing && isempty(sensealg.linsolve_kwargs)
            # For the default case use `\` to avoid any form of unnecessary cache allocation
            vec(λ) .= diffcache.J' \ vec(dgdu_val)
        else
            linear_problem = LinearProblem(diffcache.J', vec(dgdu_val'); u0 = vec(λ))
            solve(linear_problem, linsolve; alias_A = true, sensealg.linsolve_kwargs...) # u is vec(λ)
        end
    end

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
            (; g_grad_config) = diffcache
            gradient!(dgdp_val, diffcache.g[2], p, sensealg, g_grad_config[2])
        end
        recursive_sub!(dgdp_val, vjp)
        return dgdp_val
    else
        recursive_neg!(vjp)
        return vjp
    end
end
