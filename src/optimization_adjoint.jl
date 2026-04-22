# SensitivityFunction subtype for the OptimizationAdjoint VJP path.
# f = (_, q_full, _) -> F(x*, q_full) = [∇_x L; g; h_I], OOP, output size M = n_x + n_eq + n_act.
# y is a zeros(M) dummy state used to size AD buffers in vecjacobian! backends.
# λ holds the adjoint cotangent (λ_full[1:M]) from the KKT solve.
# dp is the pre-allocated output gradient buffer (size n_p), written by vecjacobian!.
struct OptimizationAdjointSensitivityFunction{
        C <: AdjointDiffCache,
        Alg <: OptimizationAdjoint,
        F,
        SolType,
        yType,
        λType,
        dpType,
    } <: SensitivityFunction
    diffcache::C
    sensealg::Alg
    f::F
    sol::SolType
    y::yType
    λ::λType
    dp::dpType
end

# Override inplace_sensitivity: f is always OOP for optimization
inplace_sensitivity(::OptimizationAdjointSensitivityFunction) = false

# Evaluate OptimizationFunction auxiliary fields (grad, hess, cons_j, lag_h).
# Dispatched on:
#   Val{iip}   — from OptimizationFunction{iip}: true = in-place (leading buffer), false = oop
#   Val{has_p} — true = AbstractOptimizationProblem (p explicit), false = OptimizationCache (p baked in)
function _opt_eval_vec(fn, n, x, p, ::Val{true}, ::Val{true})
    out = zeros(eltype(x), n); fn(out, x, p); out
end
function _opt_eval_vec(fn, n, x, _, ::Val{true}, ::Val{false})
    out = zeros(eltype(x), n); fn(out, x); out
end
_opt_eval_vec(fn, _, x, p, ::Val{false}, ::Val{true})  = fn(x, p)
_opt_eval_vec(fn, _, x, _, ::Val{false}, ::Val{false}) = fn(x)

function _opt_eval_mat(fn, m, n, x, p, ::Val{true}, ::Val{true})
    out = zeros(eltype(x), m, n); fn(out, x, p); out
end
function _opt_eval_mat(fn, m, n, x, _, ::Val{true}, ::Val{false})
    out = zeros(eltype(x), m, n); fn(out, x); out
end
_opt_eval_mat(fn, _, _, x, p, ::Val{false}, ::Val{true})  = fn(x, p)
_opt_eval_mat(fn, _, _, x, _, ::Val{false}, ::Val{false}) = fn(x)

function _opt_eval_lag_h(fn, n, x, σ, μ, p, ::Val{true}, ::Val{true})
    H = zeros(eltype(x), n, n); fn(H, x, σ, μ, p); H
end
function _opt_eval_lag_h(fn, n, x, σ, μ, _, ::Val{true}, ::Val{false})
    H = zeros(eltype(x), n, n); fn(H, x, σ, μ); H
end
_opt_eval_lag_h(fn, _, x, σ, μ, p, ::Val{false}, ::Val{true})  = fn(x, σ, μ, p)
_opt_eval_lag_h(fn, _, x, σ, μ, _, ::Val{false}, ::Val{false}) = fn(x, σ, μ)

function OptimizationAdjointSensitivityFunction(
        prob,
        opt_sol,
        sensealg::OptimizationAdjoint{CS, AD, FDT, VJP, LS, LK, OAD, AT},
        p,
        Δu
    ) where {CS, AD, FDT, VJP, LS, LK, OAD, AT}
    x_star = opt_sol.u

    lcons = prob.lcons
    ucons = prob.ucons
    has_cons = lcons !== nothing && ucons !== nothing

    # Wrap in-place cons!(res, x, p) into an out-of-place helper.
    # promote_type handles ForwardDiff Dual propagation when either x or q contains duals.
    # When prob is an OptimizationCache, prob.f.cons is a 2-arg closure
    # `(res, x) -> f.cons(res, x, captured_p)` from OptimizationBase.instantiate_function.
    # The captured field names are mangled (e.g. `#95#f`), so we search by type to find
    # the captured OptimizationFunction, regardless of field ordering.
    # Very unfortunate that this is needed, but sensitivity requires the three arg version of
    # the constraint function. To make this stable and not hacky there needs to be a way to access
    # it without going inside of the closure.
    if has_cons
        n_cons = length(lcons)
        _cons3 = if applicable(prob.f.cons, zeros(n_cons), x_star, p)
            prob.f.cons              # AbstractOptimizationProblem: already (res, x, p)
        else
            captured_f = let cl = prob.f.cons
                getfield(cl, only(fname for fname in fieldnames(typeof(cl))
                                  if getfield(cl, fname) isa SciMLBase.AbstractOptimizationFunction))
            end
            captured_f.cons
        end
        cons_cache = LazyBufferCache(_ -> (n_cons,))
        eval_cons = function (x, q)
            # When eltype(x) and eltype(q) match, reuse the cache.  Otherwise infer the
            # arithmetic result type via promote_op (compile-time inference) — ForwardDiff's
            # @define_binary_dual_op bypasses promote_type, so Dual + TrackedReal returns
            # Dual{Tag, TrackedReal, N} which promote_type would not predict.
            res = if eltype(x) === eltype(q)
                cons_cache[q]
            else
                T = Base.promote_op(+, eltype(x), eltype(q))
                Vector{T}(undef, n_cons)
            end
            _cons3(res, x, q)
            return res
        end
    else
        n_cons = 0
        eval_cons = (_, _) -> eltype(x_star)[]
    end

    # Classify constraints: equality where lcons[i] == ucons[i]
    eq_idx   = has_cons ? findall(i -> lcons[i] == ucons[i], eachindex(lcons)) : Int[]
    ineq_idx = has_cons ? findall(i -> lcons[i] != ucons[i], eachindex(lcons)) : Int[]

    # Evaluate constraints at solution
    c_val = eval_cons(x_star, p)

    # Find active inequality constraints (proximity-based initial estimate).
    # Refined below via multiplier sign check to avoid spurious active constraints.
    atol = sensealg.active_tol === nothing ? sqrt(eps(eltype(x_star))) : sensealg.active_tol
    active_lb = filter(i -> abs(c_val[i] - lcons[i]) <= atol, ineq_idx)
    active_ub = filter(i -> abs(c_val[i] - ucons[i]) <= atol, ineq_idx)

    # Equality constraint residual: g(x,p) = cons(x,p)[eq_idx] - lcons[eq_idx]
    g(x, q) = eval_cons(x, q)[eq_idx] .- lcons[eq_idx]

    n_eq = length(eq_idx)
    n_x  = length(x_star)

    lb = prob.lb
    ub = prob.ub
    active_lb_var = lb !== nothing ? findall(i -> abs(x_star[i] - lb[i]) <= atol, 1:n_x) : Int[]
    active_ub_var = ub !== nothing ? findall(i -> abs(x_star[i] - ub[i]) <= atol, 1:n_x) : Int[]

    opt_f     = prob.f
    iip_val   = Val{SciMLBase.isinplace(opt_f)}()
    has_p_val = Val{prob isa SciMLBase.AbstractOptimizationProblem}()

    objective_ad = sensealg.objective_ad

    # ---- ∇f at x_star: use stored gradient if available ----
    ∇f = if opt_f.grad !== nothing
        _opt_eval_vec(opt_f.grad, n_x, x_star, p, iip_val, has_p_val)
    else
        DI.gradient(x -> prob.f(x, p), objective_ad, x_star)
    end

    # Precompute full constraint Jacobian once if cons_j is available (reused across passes).
    J_full = has_cons && opt_f.cons_j !== nothing ?
        _opt_eval_mat(opt_f.cons_j, n_cons, n_x, x_star, p, iip_val, has_p_val) : nothing

    # Equality constraint Jacobian (fixed; independent of active set).
    Jxg = if isempty(eq_idx)
        zeros(eltype(x_star), 0, n_x)
    elseif J_full !== nothing
        J_full[eq_idx, :]
    else
        DI.jacobian(x -> g(x, p), objective_ad, x_star)
    end

    # Active ineq lower bound: h_lb(x,p) = lcons[i] - cons(x,p)[i]  (= 0 when active)
    # Active ineq upper bound: h_ub(x,p) = cons(x,p)[i] - ucons[i]  (= 0 when active)
    # Variable bound active lower: h_lb_var = lb[i] - x[i]  (∂/∂x = -eᵢ, ∂/∂p = 0)
    # Variable bound active upper: h_ub_var = x[i] - ub[i]  (∂/∂x = +eᵢ, ∂/∂p = 0)
    n_act   = length(active_lb) + length(active_ub)
    n_bound = length(active_lb_var) + length(active_ub_var)

    h_I = (x, q) -> begin
        c = eval_cons(x, q)
        vcat(
            isempty(active_lb) ? eltype(x_star)[] : lcons[active_lb] .- c[active_lb],
            isempty(active_ub) ? eltype(x_star)[] : c[active_ub] .- ucons[active_ub]
        )
    end

    Jxhι_cons = if n_act == 0
        zeros(eltype(x_star), 0, n_x)
    elseif J_full !== nothing
        vcat(
            isempty(active_lb) ? zeros(eltype(x_star), 0, n_x) : -J_full[active_lb, :],
            isempty(active_ub) ? zeros(eltype(x_star), 0, n_x) :  J_full[active_ub, :]
        )
    else
        DI.jacobian(x -> h_I(x, p), objective_ad, x_star)
    end

    Jx_bound = zeros(eltype(x_star), n_bound, n_x)
    for (j, i) in enumerate(active_lb_var); Jx_bound[j, i] = -one(eltype(x_star)); end
    for (j, i) in enumerate(active_ub_var)
        Jx_bound[length(active_lb_var) + j, i] = one(eltype(x_star))
    end
    Jxhι = vcat(Jxhι_cons, Jx_bound)

    # Dual variables from stationarity: [Jxg; Jxhι]' * [y*; z_I*; z_bound] = -∇f
    n_act_total = n_act + n_bound
    dual_vars = if n_eq + n_act_total == 0
        eltype(x_star)[]
    else
        solve(LinearProblem(Matrix(vcat(Jxg, Jxhι)'), -∇f), LinearSolve.QRFactorization()).u
    end
    y_star       = n_eq    > 0 ? dual_vars[1:n_eq]                    : eltype(x_star)[]
    zI_star      = n_act   > 0 ? dual_vars[(n_eq+1):(n_eq+n_act)]     : eltype(x_star)[]
    z_bound_star = n_bound > 0 ? dual_vars[(n_eq+n_act+1):end]        : eltype(x_star)[]

    # Multiplier sign check: KKT requires all inequality multipliers ≥ 0 at a minimum.
    # Negative multipliers indicate spuriously-included constraints (close to bound but inactive).
    # Drop those and redo only the Jxhι build and dual solve — no extra cost if all signs are good.
    mtol = sqrt(eps(eltype(x_star)))
    if (n_act > 0 && any(<(-mtol), zI_star)) ||
       (n_bound > 0 && any(<(-mtol), z_bound_star))
        n_lb     = length(active_lb)
        n_lb_var = length(active_lb_var)
        active_lb     = active_lb[findall(j -> zI_star[j]          >= -mtol, 1:n_lb)]
        active_ub     = active_ub[findall(j -> zI_star[n_lb+j]     >= -mtol, 1:length(active_ub))]
        active_lb_var = active_lb_var[findall(j -> z_bound_star[j]              >= -mtol, 1:n_lb_var)]
        active_ub_var = active_ub_var[findall(j -> z_bound_star[n_lb_var+j]     >= -mtol,
                                              1:length(active_ub_var))]
        n_act   = length(active_lb) + length(active_ub)
        n_bound = length(active_lb_var) + length(active_ub_var)

        h_I = (x, q) -> begin
            c = eval_cons(x, q)
            vcat(
                isempty(active_lb) ? eltype(x_star)[] : lcons[active_lb] .- c[active_lb],
                isempty(active_ub) ? eltype(x_star)[] : c[active_ub] .- ucons[active_ub]
            )
        end

        Jxhι_cons = if n_act == 0
            zeros(eltype(x_star), 0, n_x)
        elseif J_full !== nothing
            vcat(
                isempty(active_lb) ? zeros(eltype(x_star), 0, n_x) : -J_full[active_lb, :],
                isempty(active_ub) ? zeros(eltype(x_star), 0, n_x) :  J_full[active_ub, :]
            )
        else
            DI.jacobian(x -> h_I(x, p), objective_ad, x_star)
        end

        Jx_bound = zeros(eltype(x_star), n_bound, n_x)
        for (j, i) in enumerate(active_lb_var); Jx_bound[j, i] = -one(eltype(x_star)); end
        for (j, i) in enumerate(active_ub_var)
            Jx_bound[length(active_lb_var) + j, i] = one(eltype(x_star))
        end
        Jxhι = vcat(Jxhι_cons, Jx_bound)

        n_act_total = n_act + n_bound
        dual_vars = if n_eq + n_act_total == 0
            eltype(x_star)[]
        else
            solve(LinearProblem(Matrix(vcat(Jxg, Jxhι)'), -∇f), LinearSolve.QRFactorization()).u
        end
        y_star  = n_eq  > 0 ? dual_vars[1:n_eq]                : eltype(x_star)[]
        zI_star = n_act > 0 ? dual_vars[(n_eq+1):(n_eq+n_act)] : eltype(x_star)[]
    end

    # Lagrangian with fixed multipliers (used for p-derivative computations below)
    L = function(x, q)
        val = prob.f(x, q)
        n_eq  > 0 && (val += dot(y_star,  g(x, q)))
        n_act > 0 && (val += dot(zI_star, h_I(x, q)))
        return val
    end

    # ---- Lagrangian Hessian w.r.t. x: use lag_h if available, else hess (unconstrained), else AD ----
    # lag_h(H, u, σ, μ, p) computes Hessian of σ*f + Σ μᵢ*consᵢ.
    # Mapping from our dual vars to the full μ vector:
    #   μ[eq_idx[j]]    =  y_star[j]               (g = cons[eq] - lcons, same sign as cons)
    #   μ[active_lb[j]] = -zI_star[j]              (h_lb = lcons - cons  → -cons contribution)
    #   μ[active_ub[j]] =  zI_star[n_lb + j]       (h_ub = cons - ucons  → +cons contribution)
    Lxx = if opt_f.lag_h !== nothing
        mu_full = zeros(eltype(x_star), n_cons)
        for (j, i) in enumerate(eq_idx);    mu_full[i]  = y_star[j]                          end
        for (j, i) in enumerate(active_lb); mu_full[i] -= zI_star[j]                          end
        for (j, i) in enumerate(active_ub); mu_full[i] += zI_star[length(active_lb) + j]      end
        _opt_eval_lag_h(opt_f.lag_h, n_x, x_star, one(eltype(x_star)), mu_full, p, iip_val, has_p_val)
    elseif !has_cons && opt_f.hess !== nothing
        _opt_eval_mat(opt_f.hess, n_x, n_x, x_star, p, iip_val, has_p_val)
    else
        DI.hessian(x -> L(x, p), objective_ad, x_star)
    end

    N = n_x + n_eq + n_act_total
    KKT = zeros(eltype(x_star), N, N)
    KKT[1:n_x, 1:n_x] = Lxx
    if n_eq > 0
        KKT[1:n_x, (n_x + 1):(n_x + n_eq)]  = Jxg'
        KKT[(n_x + 1):(n_x + n_eq), 1:n_x]  = Jxg
    end
    if n_act_total > 0
        KKT[1:n_x, (n_x + n_eq + 1):N]      = Jxhι'
        KKT[(n_x + n_eq + 1):N, 1:n_x]      = Jxhι
    end

    # KKT is symmetric, so KKT' = KKT. Solve KKT * λ_full = [Δu; 0; ...; 0] once for all parameters.
    rhs_adj = vcat(Δu, zeros(eltype(x_star), n_eq + n_act_total))
    λ_full  = solve(LinearProblem(KKT, rhs_adj), sensealg.linsolve;
                    sensealg.linsolve_kwargs...).u

    if p === nothing || p isa SciMLBase.NullParameters
        tunables, repack = p, identity
    elseif isscimlstructure(p)
        tunables, repack, _ = canonicalize(Tunable(), p)
    else
        tunables, repack = p, identity
    end

    autojacvec = sensealg.autojacvec

    # f_F: OOP function (_, q_full, _) -> F(x*, q_full), the KKT residual as a function of p.
    # F = [∇_x L(x*, q); g(x*, q); h_I(x*, q)]  — output size M = n_x + n_eq + n_act.
    # Variable-bound rows are omitted since ∂(lb - x)/∂p = 0, so they don't contribute to dp.
    # y = zeros(M) is passed as the dummy state; f_F ignores it, but backends use it to
    # size their buffers, so buffers will be M-sized and match the output of f_F.
    M = n_x + n_eq + n_act
    y = zeros(eltype(x_star), M)
    f_F = let L = L, g = g, h_I = h_I, x_star = x_star,
              objective_ad = objective_ad, n_eq = n_eq, n_act = n_act
        function(_, q_full, _)
            grad_L = DI.gradient(x -> L(x, q_full), objective_ad, x_star)
            n_eq == 0 && n_act == 0 && return grad_L
            n_eq  > 0 && n_act == 0 && return vcat(grad_L, g(x_star, q_full))
            n_eq == 0 && n_act  > 0 && return vcat(grad_L, h_I(x_star, q_full))
            vcat(grad_L, g(x_star, q_full), h_I(x_star, q_full))
        end
    end

    # λ: adjoint cotangent for f_F — drop the variable-bound rows of λ_full (∂/∂p = 0).
    λ = λ_full[1:M]

    # Build pf and paramjac_config via the same adjointdiffcache machinery used by
    # SteadyStateAdjoint. f_F is OOP, output size M, no time argument.
    # For Bool dispatch: pf = ParamGradientWrapper(f_F, nothing, y), pJ = M × n_p matrix.
    # For VJP backends: pf/paramjac_config built by get_pf/get_paramjac_config as usual.
    _needs_repack = isscimlstructure(p) && !(p isa AbstractArray)
    pf, paramjac_config, pJ = if autojacvec isa ReverseDiffVJP
        # 2-input tape (no time): mirrors the AbstractNonlinearProblem branch in adjointdiffcache.
        # get_paramjac_config always builds a 3-input (y, p, [t]) tape, which fails for t=nothing.
        _tape = ReverseDiff.GradientTape((y, tunables)) do u, q
            vec(f_F(u, _needs_repack ? repack(q) : q, nothing))
        end
        _config = compile_tape(autojacvec) ? ReverseDiff.compile(_tape) : _tape
        nothing, _config, nothing
    elseif autojacvec isa EnzymeVJP
        _pf = f_F  # OOP: Enzyme.make_zero(f) called inline in _vecjacobian!
        _needs_shadow = _needs_repack
        _shadow_p = _needs_shadow ? repack(zero(tunables)) : nothing
        _config = get_paramjac_config(autojacvec, p, f_F, y, tunables, nothing;
                                      numindvar = M, alg = nothing)
        _config = (_config..., Enzyme.make_zero(_pf), _shadow_p)
        _pf, _config, nothing
    elseif autojacvec isa MooncakeVJP
        _pf = let f_F = f_F
            (out, _, q_full, _) -> (out .= f_F(nothing, q_full, nothing); out)
        end
        _pf = if _needs_repack
            let _pf = _pf, repack = repack
                (out, u, q_t, t) -> _pf(out, u, repack(q_t), t)
            end
        else
            _pf
        end
        _config = get_paramjac_config(MooncakeLoaded(), autojacvec, _pf, tunables, f_F, y, nothing)
        _pf, _config, nothing
    elseif autojacvec isa ZygoteVJP
        nothing, nothing, nothing
    elseif autojacvec isa Bool
        # Bool dispatch: ParamGradientWrapper (OOP) + pJ matrix
        _pgrad_f = _needs_repack ?
            (u, q_t, t) -> f_F(u, repack(q_t), t) :
            f_F
        _pf = ParamGradientWrapper(_pgrad_f, nothing, y)
        _pJ = zeros(eltype(x_star), M, length(tunables))
        _pf, nothing, _pJ
    else
        nothing, nothing, nothing
    end

    diffcache = AdjointDiffCache(
        nothing, pf, nothing, nothing, pJ,            # uf, pf, g, J, pJ
        nothing,                                       # dg_val
        nothing, nothing,                             # jac_config, g_grad_config
        paramjac_config,
        nothing, nothing, nothing,                    # jac_noise, paramjac_noise, f_cache
        nothing, nothing,                             # dgdu, dgdp
        nothing, nothing, nothing,                    # diffvar_idxs, algevar_idxs, factorized_mass_matrix
        false,                                        # issemiexplicitdae
        tunables, repack
    )

    dp = zeros(eltype(x_star), length(tunables))

    return OptimizationAdjointSensitivityFunction(diffcache, sensealg, f_F, opt_sol, y, λ, dp)
end

"""
    OptimizationAdjointProblem(prob, opt_sol, sensealg, p, Δu) -> dp

Compute the parameter sensitivity `dp = dG/dp` for a scalar loss `G` via one adjoint KKT solve.

Given `Δu = dG/dx* ∈ Rⁿˣ` (the cotangent of the optimal solution supplied by the caller),
solves the adjoint system `KKT * λ_full = [Δu; 0; ...; 0]` (one linear solve, exploiting
KKT symmetry), then returns `dp = -(∂F/∂p)' · λ` via a single `vecjacobian!` call, where
`F(x*, p) = [∇_x L(x*, p); g(x*, p); h_I(x*, p)]` is the KKT residual and `λ = λ_full[1:M]`
(variable-bound rows are dropped since `∂(lb-x)/∂p = 0`).

The VJP is computed via `sensealg.autojacvec` (falls back to ForwardDiff otherwise).
"""
function OptimizationAdjointProblem(
        prob,
        opt_sol,
        sensealg::OptimizationAdjoint,
        p,
        Δu
    )
    S = OptimizationAdjointSensitivityFunction(prob, opt_sol, sensealg, p, Δu)
    vecjacobian!(nothing, S.y, S.λ, S.diffcache.tunables, nothing, S; dgrad = S.dp)
    return -S.dp
end
