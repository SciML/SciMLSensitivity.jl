struct SteadyStateAdjointSensitivityFunction{
                                             C <: AdjointDiffCache,
                                             Alg <: SteadyStateAdjoint,
                                             uType,
                                             SType,
                                             fType <: ODEFunction,
                                             CV,
                                             λType,
                                             VJPType,
                                             LS
                                             } <: SensitivityFunction
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

function Base.show(io::IO,
                   t::Type{
                           SteadyStateAdjointSensitivityFunction{C, Alg, uType, SType,
                                                                 fType, CV, λType, VJPType,
                                                                 LS}}) where {C, Alg, uType,
                                                                              SType, fType,
                                                                              CV, λType,
                                                                              VJPType, LS}
    if TruncatedStacktraces.VERBOSE[]
        print(io,
              "SteadyStateAdjointSensitivityFunction{$C,$Alg,$uType,$SType,$fType,$CV,$λType,$VJPType,$LS}")
    else
        print(io, "SteadyStateAdjointSensitivityFunction{…}")
    end
end

function SteadyStateAdjointSensitivityFunction(g, sensealg, alg, sol, dgdu, dgdp, f,
                                               colorvec, needs_jac)
    @unpack p, u0 = sol.prob

    diffcache, y = adjointdiffcache(g, sensealg, false, sol, dgdu, dgdp, f, alg;
                                    quad = false, needs_jac)

    λ = zero(y)
    linsolve = needs_jac ? nothing : sensealg.linsolve
    vjp = similar(λ, length(p))

    SteadyStateAdjointSensitivityFunction(diffcache, sensealg, y, sol, f, colorvec, λ, vjp,
                                          linsolve)
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

    # TODO: What is the correct heuristic? Can we afford to compute Jacobian for
    #       cases where the length(u0) > 50 and if yes till what threshold
    needs_jac = if sensealg.linsolve === nothing
        length(u0) <= 50
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
        operator = FixedVecJacOperator(f, y, p, Val(DiffEqBase.isinplace(sol.prob)))
        linear_problem = LinearProblem(operator, vec(dgdu_val); u0 = vec(λ))
    else
        linear_problem = LinearProblem(diffcache.J', vec(dgdu_val'); u0 = vec(λ))
    end

    solve(linear_problem, linsolve) # u is vec(λ)

    try
        vecjacobian!(vec(dgdu_val), y, λ, p, nothing, sense; dgrad = vjp, dy = nothing)
    catch e
        if sense.sensealg.autojacvec === nothing
            @warn "Automatic AD choice of autojacvec failed in nonlinear solve adjoint, failing back to ODE adjoint + numerical vjp"
            vecjacobian!(vec(dgdu_val), y, λ, p, nothing, false, dgrad = vjp,
                         dy = nothing)
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
        dgdp_val .-= vjp
        return dgdp_val
    else
        vjp .*= -1
        return vjp
    end
end

function FixedVecJacOperator(f_in, y, p, ::Val{false})
    # NOTE: Zygote doesn't support inplace
    input, f_pb = Zygote.pullback(x -> f_in(x, p, nothing), y)
    f = __PullbackFunction(f_pb)
    output = f(input)[1]
    function f_operator!(du, u, p, t)
        du .= vec(f(reshape(u, size(input)))[1])
        return du
    end
    op = FunctionOperator(f_operator!, vec(input), vec(output))
    return op
end

function FixedVecJacOperator(f, y, p, ::Val{true})
    function f_operator!(du, u, p, t)
        num_vecjac!(du, (_du, _u) -> f(reshape(_du, size(y)), _u, p, t),
                    y, reshape(u, size(y)))
        return du
    end
    return FunctionOperator(f_operator!, vec(y); p)
end

# LinearSolve tries to deepcopy the operator
# Having a Zygote Pullback causes `deepcopy` to error with
# `deepcopy of Modules not supported`.
# This is a hacky workaround for that
struct __PullbackFunction{F <: Function}
    f::F
end

(f::__PullbackFunction)(args...; kwargs...) = f.f(args...; kwargs...)

Base.deepcopy_internal(f::__PullbackFunction, ::IdDict) = f
