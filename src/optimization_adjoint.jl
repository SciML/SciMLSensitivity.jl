# Differentiation helpers: dispatch on autodiff type parameter (Val{true} = ForwardDiff,
# Val{false} = FiniteDiff with the given FDT scheme)
_optimization_grad(f, x, ::Val{true}, ::FDT) where {FDT} = ForwardDiff.gradient(f, x)
function _optimization_grad(f, x, ::Val{false}, ::FDT) where {FDT}
    FiniteDiff.finite_difference_gradient(f, x, FDT())
end

_optimization_jac(f, x, ::Val{true}, ::FDT) where {FDT} = ForwardDiff.jacobian(f, x)
function _optimization_jac(f, x, ::Val{false}, ::FDT) where {FDT}
    FiniteDiff.finite_difference_jacobian(f, x, FDT())
end

_optimization_hess(f, x, ::Val{true}, ::FDT) where {FDT} = ForwardDiff.hessian(f, x)
function _optimization_hess(f, x, ::Val{false}, ::FDT) where {FDT}
    FiniteDiff.finite_difference_jacobian(
        y -> FiniteDiff.finite_difference_gradient(f, y, FDT()), x, FDT())
end

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

"""
    OptimizationAdjointProblem(prob, opt_sol, sensealg, p) -> Jpx

Compute the KKT-based parameter Jacobian `Jpx` (n_x × n_p) for a constrained
`OptimizationProblem`, where `Jpx[i,j] = ∂x*[i]/∂p[j]`.

Uses the implicit function theorem applied to the KKT conditions:

    [∇²_xx L,  J_x g^T,  J_x h_I^T] [J_p x ]   [∇²_xp L]
    [J_x g,    0,        0          ] [J_p y ] = -[J_p g  ]
    [J_x h_I,  0,        0          ] [J_p z_I]  [J_p h_I ]

where g are equality constraints, h_I are active inequality constraints, and
y*, z_I* are the corresponding dual variables.
"""
function OptimizationAdjointProblem(
        prob,
        opt_sol,
        sensealg::OptimizationAdjoint{CS, AD, FDT},
        p
    ) where {CS, AD, FDT}
    x_star = opt_sol.u
    ad_val  = Val{AD}()
    fdt_val = FDT()

    lcons = prob.lcons
    ucons = prob.ucons
    has_cons = lcons !== nothing && ucons !== nothing

    # Wrap in-place cons!(res, x, p) into an out-of-place helper.
    # promote_type handles ForwardDiff Dual propagation when either x or q contains duals.
    # When prob is an OptimizationCache, prob.f.cons is a 2-arg closure
    # `(res, x) -> f.cons(res, x, captured_p)` from OptimizationBase.instantiate_function.
    # The captured field names are mangled (e.g. `#95#f`), so we search by type to find
    # the captured OptimizationFunction, regardless of field ordering.
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
        eval_cons = function (x, q)
            T = promote_type(eltype(x), eltype(q))
            res = zeros(T, n_cons)
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

    # Find active inequality constraints
    atol = sensealg.active_tol === nothing ? sqrt(eps(eltype(x_star))) : sensealg.active_tol
    active_lb = filter(i -> abs(c_val[i] - lcons[i]) <= atol, ineq_idx)
    active_ub = filter(i -> abs(c_val[i] - ucons[i]) <= atol, ineq_idx)

    # Constraint residual functions shifted to = 0 at optimum
    # Equality: g(x,p) = cons(x,p)[eq_idx] - lcons[eq_idx]
    # Active ineq lower bound: h_lb(x,p) = lcons[i] - cons(x,p)[i]  (= 0 when active)
    # Active ineq upper bound: h_ub(x,p) = cons(x,p)[i] - ucons[i]  (= 0 when active)
    g(x, q)   = eval_cons(x, q)[eq_idx] .- lcons[eq_idx]
    h_I(x, q) = vcat(
        isempty(active_lb) ? eltype(x_star)[] : lcons[active_lb] .- eval_cons(x, q)[active_lb],
        isempty(active_ub) ? eltype(x_star)[] : eval_cons(x, q)[active_ub] .- ucons[active_ub]
    )

    n_eq  = length(eq_idx)
    n_act = length(active_lb) + length(active_ub)
    n_x   = length(x_star)

    # Variable bounds (lb/ub) as additional active inequality constraints.
    # h_lb_var: lb[i] - x[i] = 0  when active  →  ∂/∂x = -e_i,  ∂/∂p = 0
    # h_ub_var: x[i] - ub[i] = 0  when active  →  ∂/∂x = +e_i,  ∂/∂p = 0
    lb = prob.lb
    ub = prob.ub
    active_lb_var = lb !== nothing ? findall(i -> abs(x_star[i] - lb[i]) <= atol, 1:n_x) : Int[]
    active_ub_var = ub !== nothing ? findall(i -> abs(x_star[i] - ub[i]) <= atol, 1:n_x) : Int[]
    n_bound = length(active_lb_var) + length(active_ub_var)
    opt_f = prob.f
    iip_val   = Val{SciMLBase.isinplace(opt_f)}()
    has_p_val = Val{prob isa SciMLBase.AbstractOptimizationProblem}()

    # ---- ∇f at x_star: use stored gradient if available ----
    ∇f = if opt_f.grad !== nothing
        _opt_eval_vec(opt_f.grad, n_x, x_star, p, iip_val, has_p_val)
    else
        _optimization_grad(x -> prob.f(x, p), x_star, ad_val, fdt_val)
    end

    # ---- Constraint Jacobians w.r.t. x: use cons_j if available ----
    # cons_j gives the full (n_cons × n_x) Jacobian in one call; slice for eq/active ineq.
    # Sign convention: active_lb rows are negated because h_lb = lcons - cons(x,p).
    if has_cons && opt_f.cons_j !== nothing
        J_full = _opt_eval_mat(opt_f.cons_j, n_cons, n_x, x_star, p, iip_val, has_p_val)
        Jxg  = isempty(eq_idx) ? zeros(eltype(x_star), 0, n_x) : J_full[eq_idx, :]
        Jxhι = n_act == 0      ? zeros(eltype(x_star), 0, n_x) :
               vcat(isempty(active_lb) ? zeros(eltype(x_star), 0, n_x) : -J_full[active_lb, :],
                    isempty(active_ub) ? zeros(eltype(x_star), 0, n_x) :  J_full[active_ub, :])
    else
        Jxg  = isempty(eq_idx) ? zeros(eltype(x_star), 0, n_x) :
               _optimization_jac(x -> g(x, p),   x_star, ad_val, fdt_val)
        Jxhι = n_act == 0      ? zeros(eltype(x_star), 0, n_x) :
               _optimization_jac(x -> h_I(x, p), x_star, ad_val, fdt_val)
    end

    # Append trivial Jacobian rows for active variable bounds
    if n_bound > 0
        Jx_bound = zeros(eltype(x_star), n_bound, n_x)
        for (j, i) in enumerate(active_lb_var)
            Jx_bound[j, i] = -one(eltype(x_star))
        end
        for (j, i) in enumerate(active_ub_var)
            Jx_bound[length(active_lb_var) + j, i] = one(eltype(x_star))
        end
        Jxhι = vcat(Jxhι, Jx_bound)
    end
    n_act_total = n_act + n_bound

    # Dual variables from stationarity condition: constraint_jac^T * [y*; z_I*; z_bound] = -∇f(x*)
    constraint_jac = vcat(Jxg, Jxhι)   # (n_eq + n_act_total) × n_x
    # Solve overdetermined stationarity system via QR (n_x equations, n_eq+n_act_total unknowns)
    dual_vars = if n_eq + n_act_total == 0
        eltype(x_star)[]
    else
        dual_prob = LinearProblem(Matrix(constraint_jac'), -∇f)
        solve(dual_prob, LinearSolve.QRFactorization()).u
    end
    y_star  = n_eq  > 0 ? dual_vars[1:n_eq]                    : eltype(x_star)[]
    zI_star = n_act > 0 ? dual_vars[(n_eq + 1):(n_eq + n_act)] : eltype(x_star)[]

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
        _optimization_hess(x -> L(x, p), x_star, ad_val, fdt_val)
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

    # RHS: parameter Jacobians
    # Variable bounds don't depend on p, so their p-Jacobian rows are zero.
    Lxp  = _optimization_jac(
        q -> _optimization_grad(x -> L(x, q), x_star, ad_val, fdt_val), p, ad_val, fdt_val)
    Jpg  = n_eq  > 0 ? _optimization_jac(q -> g(x_star, q),   p, ad_val, fdt_val) :
                       zeros(eltype(x_star), 0, length(p))
    Jphι = n_act > 0 ? _optimization_jac(q -> h_I(x_star, q), p, ad_val, fdt_val) :
                       zeros(eltype(x_star), 0, length(p))
    RHS_p = vcat(Lxp, Jpg, Jphι, zeros(eltype(x_star), n_bound, length(p)))   # (N × n_p)

    # Solve KKT system column-by-column, reusing the factorization via the cache interface
    n_p = size(RHS_p, 2)
    Jpx = zeros(eltype(x_star), n_x, n_p)
    kkt_cache = LinearSolve.init(LinearProblem(KKT, -RHS_p[:, 1]), sensealg.linsolve;
        sensealg.linsolve_kwargs...)
    for j in 1:n_p
        kkt_cache.b = -RHS_p[:, j]
        Jpx[:, j] = LinearSolve.solve!(kkt_cache).u[1:n_x]
    end
    return Jpx   # (n_x × n_p)
end
