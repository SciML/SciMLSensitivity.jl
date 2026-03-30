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
        prob::AbstractOptimizationProblem,
        opt_sol,
        sensealg::OptimizationAdjoint{CS, AD, FDT},
        p
    ) where {CS, AD, FDT}
    x_star = opt_sol.u
    ad_val  = Val{AD}()
    fdt_val = FDT()

    lcons = prob.lcons
    ucons = prob.ucons
    n_cons = length(lcons)

    # Wrap in-place cons!(res, x, p) into an out-of-place helper.
    # promote_type handles ForwardDiff Dual propagation when either x or q contains duals.
    function eval_cons(x, q)
        T = promote_type(eltype(x), eltype(q))
        res = zeros(T, n_cons)
        prob.f.cons(res, x, q)
        return res
    end

    # Classify constraints: equality where lcons[i] == ucons[i]
    eq_idx   = findall(i -> lcons[i] == ucons[i], eachindex(lcons))
    ineq_idx = findall(i -> lcons[i] != ucons[i], eachindex(lcons))

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

    # Jacobians of constraints w.r.t. x (needed for dual variables and KKT matrix)
    Jxg  = isempty(eq_idx) ? zeros(eltype(x_star), 0, length(x_star)) :
           _optimization_jac(x -> g(x, p),   x_star, ad_val, fdt_val)
    Jxhι = n_act == 0      ? zeros(eltype(x_star), 0, length(x_star)) :
           _optimization_jac(x -> h_I(x, p), x_star, ad_val, fdt_val)

    # Dual variables from stationarity condition: constraint_jac^T * [y*; z_I*] = -∇f(x*)
    ∇f = _optimization_grad(x -> prob.f(x, p), x_star, ad_val, fdt_val)
    constraint_jac = vcat(Jxg, Jxhι)   # (n_eq + n_act) × n_x
    # Solve overdetermined stationarity system via QR (n_x equations, n_eq+n_act unknowns)
    dual_vars = if n_eq + n_act == 0
        eltype(x_star)[]
    else
        dual_prob = LinearProblem(Matrix(constraint_jac'), -∇f)
        solve(dual_prob, LinearSolve.QRFactorization(); sensealg.linsolve_kwargs...).u
    end
    y_star  = n_eq  > 0 ? dual_vars[1:n_eq]        : eltype(x_star)[]
    zI_star = n_act > 0 ? dual_vars[(n_eq + 1):end] : eltype(x_star)[]

    # Lagrangian with fixed multipliers
    function L(x, q)
        val = prob.f(x, q)
        n_eq  > 0 && (val += dot(y_star,  g(x, q)))
        n_act > 0 && (val += dot(zI_star, h_I(x, q)))
        return val
    end

    # Assemble KKT matrix
    Lxx = _optimization_hess(x -> L(x, p), x_star, ad_val, fdt_val)

    n_x = length(x_star)
    N   = n_x + n_eq + n_act
    KKT = zeros(eltype(x_star), N, N)
    KKT[1:n_x, 1:n_x] = Lxx
    if n_eq > 0
        KKT[1:n_x, (n_x + 1):(n_x + n_eq)]  = Jxg'
        KKT[(n_x + 1):(n_x + n_eq), 1:n_x]  = Jxg
    end
    if n_act > 0
        KKT[1:n_x, (n_x + n_eq + 1):N]      = Jxhι'
        KKT[(n_x + n_eq + 1):N, 1:n_x]      = Jxhι
    end

    # RHS: parameter Jacobians
    Lxp  = _optimization_jac(
        q -> _optimization_grad(x -> L(x, q), x_star, ad_val, fdt_val), p, ad_val, fdt_val)
    Jpg  = n_eq  > 0 ? _optimization_jac(q -> g(x_star, q),   p, ad_val, fdt_val) :
                       zeros(eltype(x_star), 0, length(p))
    Jphι = n_act > 0 ? _optimization_jac(q -> h_I(x_star, q), p, ad_val, fdt_val) :
                       zeros(eltype(x_star), 0, length(p))
    RHS_p = vcat(Lxp, Jpg, Jphι)   # (N × n_p)

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
