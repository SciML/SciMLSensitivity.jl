# SensitivityFunction subtype for the OptimizationAdjoint VJP path.
# f = (_, q_full, _) -> Φ(q_full) = λ·F(x*, q_full), OOP, scalar output (length-1 vector).
# The KKT cotangent λ is folded into Φ via the λ-contraction (see f_F below), so the residual
# is scalar: the outer VJP just takes ∇_q Φ with a unit seed. y is a length-1 dummy state used
# to size AD buffers in vecjacobian! backends; λ is the unit seed [1].
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

# Override getprob: OptimizationSolution has no `prob` field; its cache plays the role.
getprob(S::OptimizationAdjointSensitivityFunction) = S.sol.cache

# Wrapper for the KKT residual closure. Named struct so we can declare the SciMLFunction
# traits that `_vecjacobian!` queries
struct OptimizationKKTResidual{F}
    f::F
end
(o::OptimizationKKTResidual)(args...) = o.f(args...)
SciMLBase.has_paramjac(::OptimizationKKTResidual) = false
SciMLBase.has_jac(::OptimizationKKTResidual) = false
SciMLBase.has_vjp(::OptimizationKKTResidual) = false
SciMLBase.has_vjp_p(::OptimizationKKTResidual) = false
SciMLBase.unwrapped_f(o::OptimizationKKTResidual) = o.f

# Evaluate OptimizationFunction auxiliary fields (grad, hess, cons_j, lag_h).
# Dispatched on:
#   Val{iip}   — from OptimizationFunction{iip}: true = in-place (leading buffer), false = oop
#   Val{has_p} — true = AbstractOptimizationProblem (p explicit), false = OptimizationCache (p baked in)
function _opt_eval_vec(fn, n, x, p, ::Val{true}, ::Val{true})
    out = zeros(eltype(x), n); fn(out, x, p)
    return out
end
function _opt_eval_vec(fn, n, x, _, ::Val{true}, ::Val{false})
    out = zeros(eltype(x), n); fn(out, x)
    return out
end
_opt_eval_vec(fn, _, x, p, ::Val{false}, ::Val{true}) = fn(x, p)
_opt_eval_vec(fn, _, x, _, ::Val{false}, ::Val{false}) = fn(x)

function _opt_eval_mat(fn, m, n, x, p, ::Val{true}, ::Val{true})
    out = zeros(eltype(x), m, n); fn(out, x, p)
    return out
end
function _opt_eval_mat(fn, m, n, x, _, ::Val{true}, ::Val{false})
    out = zeros(eltype(x), m, n); fn(out, x)
    return out
end
_opt_eval_mat(fn, _, _, x, p, ::Val{false}, ::Val{true}) = fn(x, p)
_opt_eval_mat(fn, _, _, x, _, ::Val{false}, ::Val{false}) = fn(x)

function _opt_eval_lag_h(fn, n, x, σ, μ, p, ::Val{true}, ::Val{true})
    H = zeros(eltype(x), n, n); fn(H, x, σ, μ, p)
    return H
end
function _opt_eval_lag_h(fn, n, x, σ, μ, _, ::Val{true}, ::Val{false})
    H = zeros(eltype(x), n, n); fn(H, x, σ, μ)
    return H
end
_opt_eval_lag_h(fn, _, x, σ, μ, p, ::Val{false}, ::Val{true}) = fn(x, σ, μ, p)
_opt_eval_lag_h(fn, _, x, σ, μ, _, ::Val{false}, ::Val{false}) = fn(x, σ, μ)

# ---- Forward-mode AD for the adjoint's own derivatives ----
# The OptimizationAdjoint computes two derivatives itself, both in forward mode:
#   * the Lagrangian gradient ∇_x L — the stationarity rows of the KKT residual `F`, which
#     the outer VJP then differentiates w.r.t. the parameters; and
#   * its Jacobian w.r.t. x, i.e. the Lagrangian Hessian `Lxx` — only as a fallback, when
#     the OptimizationFunction exposes no second-order info (no `lag_h`/`hess`).
# These deliberately do NOT use the stored `grad`/`cons_j`/`hess` (their DI preparation is
# frozen at the solve's Float64 types and rejects the dual inputs AD introduces); they
# differentiate the raw Lagrangian `L` instead. The backend is the optimization problem's
# own ADType by default, or an override passed to `OptimizationAdjoint(autodiff=...)`.
# `_opt_validate_fwd` rejects reverse-mode backends (both derivatives are taken in forward
# mode — the inner gradient must also nest cleanly inside the outer reverse-mode VJP) and
# otherwise returns the backend unchanged; `_opt_grad`/`_opt_hess` then dispatch on it.
function _opt_validate_fwd(adtype::Union{AutoReverseDiff, AutoZygote, AutoTracker, AutoMooncake})
    throw(
        ArgumentError(
            "OptimizationAdjoint differentiates the KKT optimality conditions in forward " *
            "mode, but a reverse-mode AD backend ($(nameof(typeof(adtype)))) was selected " *
            "(via the OptimizationFunction's `adtype` or `OptimizationAdjoint(autodiff=...)`). " *
            "Pass a forward-mode backend instead: AutoForwardDiff(), AutoFiniteDiff(), or " *
            "AutoEnzyme()."
        )
    )
end
_opt_validate_fwd(adtype) = adtype

# Forward-mode gradient of scalar `f` at `x`. Enzyme is run in forward mode; FiniteDiff maps
# to itself; AutoForwardDiff and any other backend use ForwardDiff.
_opt_grad(::AutoFiniteDiff, f, x) = FiniteDiff.finite_difference_gradient(f, x)
# Enzyme is run in forward mode with two annotations the Lagrangian closure requires:
#   * `Enzyme.Const(f)` — the closure captures parameters/multipliers/the OptimizationFunction,
#     which Enzyme can't prove read-only; we differentiate only the explicit `x`.
#   * `set_runtime_activity` — Enzyme's static activity analysis can't classify active-vs-const
#     through that opaque capture, so we opt into runtime activity (also needed for the nested
#     residual case, where `x` is differentiated under the outer VJP).
# `Enzyme.gradient(Forward, ...)` returns a 1-tuple holding a TupleArray; collect to a plain
# Vector so it composes (vcat into the residual, and re-differentiation for Lxx).
const _OPT_ENZYME_FWD = Enzyme.set_runtime_activity(Enzyme.Forward)
_opt_grad(::AutoEnzyme, f, x) = collect(only(Enzyme.gradient(_OPT_ENZYME_FWD, Enzyme.Const(f), x)))
_opt_grad(_, f, x) = ForwardDiff.gradient(f, x)

# Hessian of scalar `f` at `x` (used for Lxx = ∇²_x L when no second-order info is stored).
# ForwardDiff and FiniteDiff use their native second-order routines (FiniteDiff's stencil is
# more accurate than differencing a finite-difference gradient). Enzyme has no native Hessian,
# so there we differentiate the forward-mode gradient once more — `jacobian(Const(∇f), x)` —
# with the same `Const`/runtime-activity handling `_opt_grad` needs.
_opt_hess(::AutoFiniteDiff, f, x) = FiniteDiff.finite_difference_hessian(f, x)
_opt_hess(m::AutoEnzyme, f, x) = only(Enzyme.jacobian(_OPT_ENZYME_FWD, Enzyme.Const(z -> _opt_grad(m, f, z)), x))
_opt_hess(_, f, x) = ForwardDiff.hessian(f, x)

# Directional derivative d/dε f(x + ε v)|_0 = v'∇f(x), in forward mode — the Pearlmutter-style
# contraction that yields v'∇f directly (a single forward seed) without forming the full n-wide
# gradient. Used for the stationarity term λx'∇_x L of the scalar KKT residual. Enzyme falls
# back to the validated full-gradient path then contracts (sum(.*), BLAS-free), since its
# forward JVP API is finickier and the gradient route is already exercised.
_opt_dir(::AutoFiniteDiff, f, x, v) =
    FiniteDiff.finite_difference_derivative(ε -> f(x .+ ε .* v), zero(eltype(x)))
_opt_dir(m::AutoEnzyme, f, x, v) = sum(v .* _opt_grad(m, f, x))
_opt_dir(_, f, x, v) = ForwardDiff.derivative(ε -> f(x .+ ε .* v), zero(eltype(x)))

# Constraint Hessians: `cons_h` writes a length-`m` vector of `n×n` matrices, `H[i] = ∇²cᵢ`.
# m = n_cons, n = n_x.
function _opt_eval_cons_h(fn, m, n, x, p, ::Val{true}, ::Val{true})
    H = [zeros(eltype(x), n, n) for _ in 1:m]; fn(H, x, p)
    return H
end
function _opt_eval_cons_h(fn, m, n, x, _, ::Val{true}, ::Val{false})
    H = [zeros(eltype(x), n, n) for _ in 1:m]; fn(H, x)
    return H
end
_opt_eval_cons_h(fn, _, _, x, p, ::Val{false}, ::Val{true}) = fn(x, p)
_opt_eval_cons_h(fn, _, _, x, _, ::Val{false}, ::Val{false}) = fn(x)

# Thrown when the OptimizationFunction does not carry a derivative `OptimizationAdjoint`
# needs (`grad`, `cons_j`, or the second-order info for the Lagrangian Hessian). Rather
# than silently recomputing it via AD (duplicating the derivative machinery the
# Optimization stack already provides), we require it to be supplied.
#   missing :: :grad | :cons_jac | :hessian
#   has_hess / has_cons_h are only meaningful for :hessian (which partial info is present).
struct OptimizationAdjointMissingDerivativeError <: Exception
    missing::Symbol
    has_cons::Bool
    has_hess::Bool
    has_cons_h::Bool
end
function OptimizationAdjointMissingDerivativeError(missing::Symbol, has_cons::Bool)
    return OptimizationAdjointMissingDerivativeError(missing, has_cons, false, false)
end

function Base.showerror(io::IO, e::OptimizationAdjointMissingDerivativeError)
    if e.missing === :grad
        print(io,
            "OptimizationAdjoint requires the objective gradient `grad`, but the " *
            "OptimizationFunction does not provide it.\n")
    elseif e.missing === :cons_jac
        print(io,
            "OptimizationAdjoint requires the constraint Jacobian `cons_j`, but the " *
            "OptimizationFunction does not provide it.\n")
    else # :hessian
        print(io,
            "OptimizationAdjoint requires second-order derivative information to assemble " *
            "the Lagrangian Hessian, but the OptimizationFunction does not provide it.\n")
        if e.has_cons
            print(io,
                "For a constrained problem it needs either `lag_h` (the Lagrangian Hessian), " *
                "or both `hess` (the objective Hessian) and `cons_h` (the constraint Hessians).\n")
            e.has_hess && !e.has_cons_h &&
                print(io, "`hess` is present but `cons_h` is missing.\n")
            !e.has_hess && e.has_cons_h &&
                print(io, "`cons_h` is present but `hess` is missing.\n")
        else
            print(io,
                "For an unconstrained problem it needs `hess` (the objective Hessian) " *
                "or `lag_h`.\n")
        end
    end
    print(io,
        "Provide the required derivatives on the `OptimizationFunction` — either explicitly, " *
        "or by constructing it with an AD backend (e.g. `OptimizationFunction(f, AutoForwardDiff())`). " *
        "Note that an AD backend alone is not sufficient: the relevant derivatives are only " *
        "generated when the optimizer that produced the solution requested them, so use an " *
        "optimizer that requests them (or pass them directly).")
    if !e.has_cons
        print(io,
            "\nAlternatively, for unconstrained problems use `UnconstrainedOptimizationAdjoint`, " *
            "which requires none of these.")
    end
end

function OptimizationAdjointSensitivityFunction(
        prob,
        opt_sol,
        sensealg::OptimizationAdjoint{CS, AD, FDT, VJP, LS, LK, AT},
        p,
        Δu
    ) where {CS, AD, FDT, VJP, LS, LK, AT}
    x_star = opt_sol.u

    lcons = prob.lcons
    ucons = prob.ucons
    has_cons = lcons !== nothing && ucons !== nothing

    # Wrap in-place cons!(res, x, p) into an out-of-place helper.
    # promote_type handles ForwardDiff Dual propagation when either x or q contains duals.
    # The KKT residual differentiates the constraints w.r.t. the parameters, so we need the
    # three-arg `cons(res, x, p)` form. OptimizationBase ≥ 5.1.2 (SciML/Optimization.jl#1184,
    # the AugLag rewrite) builds `prob.f.cons` as `(res, x, p_call = p) -> f.cons(res, x, p_call)`,
    # which forwards straight to the raw user constraint and is therefore safe to differentiate
    # through w.r.t. p. Earlier versions baked `p` in as a 2-arg `(res, x)` closure and required
    # reaching inside it to recover the original 3-arg function; that path is no longer supported.
    # Single-assignment for n_cons and _cons3: both are captured by closures (g, h_I, eval_cons),
    # so assigning them in separate if/else arms would force Julia to box them.
    n_cons = has_cons ? length(lcons) : 0
    _cons3 = has_cons ? prob.f.cons : nothing
    eval_cons = let _cons3 = _cons3, n_cons = n_cons, x_star = x_star
        function (x, q)
            _cons3 === nothing && return eltype(x_star)[]
            # promote_op gives the inferred result eltype of `+(eltype(x), eltype(q))`.
            # Preferred over promote_type because ForwardDiff's @define_binary_dual_op
            # bypasses promote_type — e.g. Dual + TrackedReal returns Dual{Tag, TrackedReal, N}
            # which promote_type would not predict.
            T = Base.promote_op(+, eltype(x), eltype(q))
            res = Vector{T}(undef, n_cons)
            _cons3(res, x, q)
            return res
        end
    end

    # Classify constraints: equality where lcons[i] == ucons[i]
    eq_idx = has_cons ? findall(i -> lcons[i] == ucons[i], eachindex(lcons)) : Int[]
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
    n_x = length(x_star)

    lb = prob.lb
    ub = prob.ub
    active_lb_var = lb !== nothing ? findall(i -> abs(x_star[i] - lb[i]) <= atol, 1:n_x) : Int[]
    active_ub_var = ub !== nothing ? findall(i -> abs(x_star[i] - ub[i]) <= atol, 1:n_x) : Int[]

    opt_f = prob.f
    iip_val = Val{SciMLBase.isinplace(opt_f)}()
    has_p_val = Val{prob isa SciMLBase.AbstractOptimizationProblem}()

    # Forward-mode backend for the adjoint's own derivatives (inner Lagrangian gradient and,
    # as a fallback, Lxx). Defaults to the optimization problem's ADType; overridden when the
    # user passes `OptimizationAdjoint(autodiff=...)`. Errors on reverse-mode backends.
    fwd_mode = _opt_validate_fwd(sensealg.autodiff === nothing ? opt_f.adtype : sensealg.autodiff)

    # ---- ∇f at x_star: require the stored gradient ----
    ∇f = if opt_f.grad !== nothing
        _opt_eval_vec(opt_f.grad, n_x, x_star, p, iip_val, has_p_val)
    else
        throw(OptimizationAdjointMissingDerivativeError(:grad, has_cons))
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
        throw(OptimizationAdjointMissingDerivativeError(:cons_jac, has_cons))
    end

    # Active ineq lower bound: h_lb(x,p) = lcons[i] - cons(x,p)[i]  (= 0 when active)
    # Active ineq upper bound: h_ub(x,p) = cons(x,p)[i] - ucons[i]  (= 0 when active)
    # Variable bound active lower: h_lb_var = lb[i] - x[i]  (∂/∂x = -eᵢ, ∂/∂p = 0)
    # Variable bound active upper: h_ub_var = x[i] - ub[i]  (∂/∂x = +eᵢ, ∂/∂p = 0)
    #
    # Active-set iteration: KKT requires inequality multipliers ≥ 0 at any optimum.
    # Build Jxhι and recover multipliers; if any are negative, the offending constraints
    # were spuriously flagged by proximity detection. Drop them and re-solve. Dropped
    # indices never come back, so this terminates in ≤ |initial active set| iterations.
    # The iteration cap is defense-in-depth; the mathematical bound is the same.
    mtol = sqrt(eps(eltype(x_star)))
    max_iters = length(active_lb) + length(active_ub) +
                length(active_lb_var) + length(active_ub_var) + 1
    local h_I, n_act, n_bound, n_act_total, y_star, zI_star, Jxhι
    has_negatives = true
    iter = 0
    while has_negatives && iter < max_iters
        iter += 1
        n_act = length(active_lb) + length(active_ub)
        n_bound = length(active_lb_var) + length(active_ub_var)

        h_I = let active_lb = active_lb, active_ub = active_ub,
                lcons = lcons, ucons = ucons, eval_cons = eval_cons, x_star = x_star
            (x, q) -> begin
                c = eval_cons(x, q)
                vcat(
                    isempty(active_lb) ? eltype(x_star)[] : lcons[active_lb] .- c[active_lb],
                    isempty(active_ub) ? eltype(x_star)[] : c[active_ub] .- ucons[active_ub]
                )
            end
        end

        Jxhι_cons = if n_act == 0
            zeros(eltype(x_star), 0, n_x)
        elseif J_full !== nothing
            vcat(
                isempty(active_lb) ? zeros(eltype(x_star), 0, n_x) : -J_full[active_lb, :],
                isempty(active_ub) ? zeros(eltype(x_star), 0, n_x) : J_full[active_ub, :]
            )
        else
            throw(OptimizationAdjointMissingDerivativeError(:cons_jac, has_cons))
        end

        Jx_bound = zeros(eltype(x_star), n_bound, n_x)
        for (j, i) in enumerate(active_lb_var)
            Jx_bound[j, i] = -one(eltype(x_star))
        end
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
        y_star = n_eq > 0 ? dual_vars[1:n_eq] : eltype(x_star)[]
        zI_star = n_act > 0 ? dual_vars[(n_eq + 1):(n_eq + n_act)] : eltype(x_star)[]
        z_bound_star = n_bound > 0 ? dual_vars[(n_eq + n_act + 1):end] : eltype(x_star)[]

        neg_in_zI = n_act > 0 && any(<(-mtol), zI_star)
        neg_in_bound = n_bound > 0 && any(<(-mtol), z_bound_star)
        has_negatives = neg_in_zI || neg_in_bound

        if has_negatives
            # Filter offenders out of each active set; they never re-enter.
            # Use non-capturing broadcast masks rather than `findall(j -> zI_star[j]...)`:
            # a capturing closure created in this loop forces zI_star/z_bound_star to be boxed
            # in the enclosing frame (zI_star is also captured by L below), defeating the let.
            n_lb = length(active_lb)
            n_ub = length(active_ub)
            n_lb_var = length(active_lb_var)
            n_ub_var = length(active_ub_var)
            active_lb = active_lb[findall(@view(zI_star[1:n_lb]) .>= -mtol)]
            active_ub = active_ub[findall(@view(zI_star[(n_lb + 1):(n_lb + n_ub)]) .>= -mtol)]
            active_lb_var = active_lb_var[findall(@view(z_bound_star[1:n_lb_var]) .>= -mtol)]
            active_ub_var = active_ub_var[findall(@view(z_bound_star[(n_lb_var + 1):(n_lb_var + n_ub_var)]) .>= -mtol)]
        end
    end

    # Lagrangian with fixed multipliers (differentiated for both the residual stationarity
    # rows and the Lxx fallback). Uses `sum(μ .* c)` rather than `dot(μ, c)`: `dot` lowers to
    # a BLAS call on Float64 vectors, which Enzyme's forward mode cannot differentiate under
    # runtime activity. The multiplier vectors are tiny, so the broadcast costs nothing, and
    # ForwardDiff/FiniteDiff are unaffected.
    L = let prob = prob, y_star = y_star, g = g, n_eq = n_eq,
            zI_star = zI_star, h_I = h_I, n_act = n_act
        function (x, q)
            val = prob.f(x, q)
            n_eq > 0 && (val += sum(y_star .* g(x, q)))
            n_act > 0 && (val += sum(zI_star .* h_I(x, q)))
            return val
        end
    end

    # ---- Lagrangian Hessian w.r.t. x ----
    # ∇²L = σ*∇²f + Σ μᵢ*∇²cᵢ with σ = 1. Mapping from our dual vars to the full μ vector
    # (chosen to match `lag_h`'s Hessian-of-(σ*f + Σ μᵢ*consᵢ) convention):
    #   μ[eq_idx[j]]    =  y_star[j]               (g = cons[eq] - lcons, same sign as cons)
    #   μ[active_lb[j]] = -zI_star[j]              (h_lb = lcons - cons  → -cons contribution)
    #   μ[active_ub[j]] =  zI_star[n_lb + j]       (h_ub = cons - ucons  → +cons contribution)
    mu_full = zeros(eltype(x_star), n_cons)
    for (j, i) in enumerate(eq_idx)
        mu_full[i] = y_star[j]
    end
    for (j, i) in enumerate(active_lb)
        mu_full[i] -= zI_star[j]
    end
    for (j, i) in enumerate(active_ub)
        mu_full[i] += zI_star[length(active_lb) + j]
    end

    # Lagrangian Hessian ∇²_x L, in order of preference:
    #   1. `lag_h`  — the combined Lagrangian Hessian, used directly.
    #   2. `hess` (+ `cons_h` when constrained) — assemble ∇²L = ∇²f + Σ μᵢ ∇²cᵢ.
    #      This covers solvers (e.g. IPNewton) that populate `hess`/`cons_h` but not `lag_h`.
    #   3. AD fallback — forward-mode over the Lagrangian gradient. Covers solvers that
    #      expose no second-order info (e.g. SLSQP: only `grad` + `cons_j`). See below.
    Lxx = if opt_f.lag_h !== nothing
        _opt_eval_lag_h(opt_f.lag_h, n_x, x_star, one(eltype(x_star)), mu_full, p, iip_val, has_p_val)
    elseif opt_f.hess !== nothing && (!has_cons || opt_f.cons_h !== nothing)
        H = _opt_eval_mat(opt_f.hess, n_x, n_x, x_star, p, iip_val, has_p_val)
        if has_cons && opt_f.cons_h !== nothing
            cons_hessians = _opt_eval_cons_h(
                opt_f.cons_h, n_cons, n_x, x_star, p, iip_val, has_p_val
            )
            for i in 1:n_cons
                iszero(mu_full[i]) && continue
                # non-mutating: `hess` may return Symmetric/StaticArray/immutable
                H = H + mu_full[i] * cons_hessians[i]
            end
        end
        H
    else
        # No second-order info: take the Hessian of the *raw* Lagrangian `L` w.r.t. x (its
        # constraint terms route through the prep-free `prob.f.cons`), NOT the stored
        # `grad`/`cons_j` — their DI preparation is frozen at Float64 and a forward pass
        # introduces dual `x`, which it would reject (the same preparation wall that blocks
        # reusing them in the p-residual). The first-order `cons_j` is still reused for the
        # Jacobian blocks Jxg/Jxhι; only the second-order term is AD'd here. `fwd_mode` picks
        # the forward-mode backend; evaluated at the real Float64 (x*, p) — no nesting with
        # the outer VJP.
        let L = L, p = p, fwd_mode = fwd_mode
            _opt_hess(fwd_mode, z -> L(z, p), x_star)
        end
    end

    N = n_x + n_eq + n_act_total
    KKT = zeros(eltype(x_star), N, N)
    KKT[1:n_x, 1:n_x] = Lxx
    if n_eq > 0
        KKT[1:n_x, (n_x + 1):(n_x + n_eq)] = Jxg'
        KKT[(n_x + 1):(n_x + n_eq), 1:n_x] = Jxg
    end
    if n_act_total > 0
        KKT[1:n_x, (n_x + n_eq + 1):N] = Jxhι'
        KKT[(n_x + n_eq + 1):N, 1:n_x] = Jxhι
    end

    # KKT is symmetric, so KKT' = KKT. Solve KKT * λ_full = [Δu; 0; ...; 0] once for all parameters.
    rhs_adj = vcat(Δu, zeros(eltype(x_star), n_eq + n_act_total))
    λ_full = solve(
        LinearProblem(KKT, rhs_adj), sensealg.linsolve;
        sensealg.linsolve_kwargs...
    ).u

    
    tunables, repack = if p === nothing || p isa SciMLBase.NullParameters
        p, identity
    elseif isscimlstructure(p)
        t, r, _ = canonicalize(Tunable(), p)
        t, r
    else
        p, identity
    end

    autojacvec = sensealg.autojacvec

    # Scalar KKT residual via λ-contraction. The cotangent λ is already known (constant) from
    # the KKT solve, so instead of forming the vector residual F = [∇_xL; g; h_I] and VJP'ing
    # it with λ, we build the scalar Φ(q) = λ·F(x*, q) and let the outer pass take ∇_q Φ (a VJP
    # of a scalar with seed [1]). Identical result, but the stationarity term λx'∇_xL is a
    # single forward directional derivative (Pearlmutter) rather than the full n_x-wide gradient
    # — much less for the outer AD to differentiate through. `L` routes its constraint terms
    # through the prep-free `prob.f.cons`, so it is differentiable w.r.t. `q` (unlike the stored
    # `grad`/`cons_j`, frozen at the solve's Float64 types). Constraint terms use sum(.*) not
    # `dot` to stay BLAS-free (Enzyme forward can't differentiate BLAS `dot` under runtime
    # activity). Variable-bound rows are dropped: ∂(lb - x)/∂p = 0.
    #
    # Scalar output: y is the length-1 dummy state backends use to size buffers.
    y = zeros(eltype(x_star), 1)
    λx = λ_full[1:n_x]
    λy = n_eq  > 0 ? λ_full[(n_x + 1):(n_x + n_eq)]                : eltype(x_star)[]
    λz = n_act > 0 ? λ_full[(n_x + n_eq + 1):(n_x + n_eq + n_act)] : eltype(x_star)[]
    f_F = OptimizationKKTResidual(
        let L = L, g = g, h_I = h_I, x_star = x_star, λx = λx, λy = λy, λz = λz,
                fwd_mode = fwd_mode, n_eq = n_eq, n_act = n_act
            function (_, q_full, _)
                val = _opt_dir(fwd_mode, x -> L(x, q_full), x_star, λx)
                n_eq  > 0 && (val += sum(λy .* g(x_star, q_full)))
                n_act > 0 && (val += sum(λz .* h_I(x_star, q_full)))
                return [val]
            end
        end
    )

    # Scalar cotangent: the contraction with the KKT λ is folded into Φ above, so the outer
    # VJP just needs ∇_q Φ — i.e. seed with 1.
    λ = [one(eltype(x_star))]

    # Build pf and paramjac_config via the same adjointdiffcache machinery used by
    # SteadyStateAdjoint. f_F is OOP, scalar output, no time argument.
    # For Bool dispatch: pf = ParamGradientWrapper(f_F, nothing, y), pJ = 1 × n_p matrix.
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
        _config = get_paramjac_config(
            autojacvec, p, f_F, y, tunables, nothing;
            numindvar = 1, alg = nothing
        )
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
        _pJ = zeros(eltype(x_star), 1, length(tunables))
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
KKT symmetry), then returns `dp = -∇_q Φ` via a single `vecjacobian!` call. Rather than
VJP'ing the vector KKT residual `F(x*, q) = [∇_x L; g; h_I]` against `λ_full`, the (constant)
contraction is folded in up front as the scalar `Φ(q) = λ_full · F(x*, q)`, so the outer pass
only needs its gradient `∇_q Φ` (a scalar VJP, unit seed). Variable-bound rows are dropped
since `∂(lb-x)/∂p = 0`.

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
