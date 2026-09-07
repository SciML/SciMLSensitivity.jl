# Adjoint sensitivity analysis for fully implicit DAEs (`DAEProblem`):
#
#     F(du, u, p, t) = 0
#
# following the augmented adjoint DAE formulation of Cao, Li, Petzold & Serban,
# "Adjoint sensitivity analysis for differential-algebraic equations: The adjoint
# DAE system and its numerical solution", SIAM Journal on Scientific Computing
# 24(3), 2003.
#
# For a cost functional G(p) = ∫ g(u, p, t) dt (discrete contributions are handled
# as Dirac deltas via callbacks), the adjoint λ(t) satisfies, in forward time,
#
#     d/dt(F_du' λ) - F_u' λ + g_u' = 0
#
# with dG/du₀ = (F_du' λ)(t₀) and dG/dp = ∫ (g_p' - F_p' λ) dt. To avoid the total
# time derivative of F_du along the trajectory, introduce the augmented variable
#
#     w := F_du' λ
#
# which turns the adjoint into the semi-explicit system
#
#     ẇ = F_u' λ - g_u',        0 = w - F_du' λ.
#
# As written, this system is index-2 even when the original DAE is index-1 (λ
# components paired with algebraic variables appear only under one differentiation
# of the constraint). We therefore index-reduce structurally, using the fact that
# w_j ≡ 0 for every algebraic variable j (zero column of F_du): for those rows the
# differential equation ẇ_j = 0 = [F_u' λ - g_u']_j is itself the algebraic
# equation determining the remaining adjoint components. The augmented adjoint
# state is z = [w; λ] — with the parameter quadrature appended for
# `InterpolatingAdjoint`, z = [w; λ; grad] — with
#
#     row j ∈ differential vars:  ẇ_j    = [F_u' λ]_j - g_u[j]
#     row j ∈ algebraic vars:     0      = w_j
#     row n+j, j differential:    0      = w_j - [F_du' λ]_j
#     row n+j, j algebraic:       0      = [F_u' λ]_j - g_u[j]
#     grad rows:                  grad'  = F_p' λ - g_p'
#
# solved backwards from z(T) = 0 as a singular-mass-matrix ODEProblem (for
# mass-matrix-capable ODE algorithms) or as a fully implicit DAEProblem (for DAE
# algorithms). For the special case F = M u̇ - f(u, p, t) this reduces exactly to
# the transposed mass-matrix adjoint used by `ODEAdjointProblem`. Note no sign
# flips relative to the ODE adjoint code appear here: the vjps of the residual F
# carry them. For `QuadratureAdjoint`/`GaussAdjoint` the parameter gradient is
# instead accumulated from the p-vjp of the residual, -λ' F_p + g_p, along the
# reverse solve (see `AdjointSensitivityIntegrand` and `GaussIntegrand`).
#
# Discrete cost contributions jump `w` at the data points. When the cost touches
# algebraic variables (index-1) or when the DAE is index-2 Hessenberg, the raw jump
# gᵤ is inconsistent with the adjoint's (hidden) constraints and must be corrected;
# this shares the `ReverseLossCallback` machinery with the semi-explicit
# mass-matrix code: a multiplier Δλa supported on the algebraic equations produces
# the consistent jump
#
#     w[diffvars] += gᵤ[diffvars] + dhdd' Δλa,      dhdd = F_u[algeqs, diffvars]
#
# together with the parameter-gradient correction dp += Δλa' ∂F_algeqs/∂p that is
# accumulated after the reverse solve (the `rcb.Δλas` mechanism). The multiplier is
# index-dependent:
#
#     index-1:            Δλa = -dhda' \ gᵤ[algvars],   dhda = F_u[algeqs, algvars]
#     index-2 Hessenberg: Δλa = -(N dhdd') \ (N gᵤ[diffvars]),
#                         N = (F_du[diffeqs, diffvars] \ F_u[diffeqs, algvars])'
#
# where the index-2 formula enforces the adjoint hidden constraint (post-jump
# w[diffvars] must annihilate range(F_u[diffeqs, algvars]) through F_du) while
# leaving the pairing with admissible state perturbations (tangent to the
# constraint manifold) unchanged. Discrete cost on the algebraic variables of an
# index-2 DAE would require derivatives of delta distributions and is rejected.

# `unwrapped_f` does not descend into AbstractSciMLFunctions; the Enzyme vjp
# path needs the raw residual (a `DAEFunction` whose fields are all singletons
# is a ghost type that Enzyme cannot wrap in `Duplicated`).
dae_unwrapped_f(f) = unwrapped_f(f)
dae_unwrapped_f(f::DAEFunction) = unwrapped_f(f.f)

function dae_du_jacobian(f, isinplace, dy, y, p, t)
    return if isinplace
        ForwardDiff.jacobian(dy) do du
            res = similar(du, promote_type(eltype(du), eltype(y)), length(y))
            res .= false
            f(res, du, y, p, t)
            res
        end
    else
        ForwardDiff.jacobian(du -> vec(f(du, y, p, t)), dy)
    end
end

function dae_u_jacobian(f, isinplace, dy, y, p, t)
    return if isinplace
        ForwardDiff.jacobian(y) do u
            res = similar(u, promote_type(eltype(u), eltype(dy)), length(u))
            res .= false
            f(res, dy, u, p, t)
            res
        end
    else
        ForwardDiff.jacobian(u -> vec(f(dy, u, p, t)), y)
    end
end

function dae_p_jacobian(f, isinplace, dy, y, p, t, tunables, repack)
    return if isinplace
        ForwardDiff.jacobian(tunables) do _tunables
            res = similar(_tunables, length(y))
            res .= false
            f(res, dy, y, repack(_tunables), t)
            res
        end
    else
        ForwardDiff.jacobian(_tunables -> vec(f(dy, y, repack(_tunables), t)), tunables)
    end
end

const DAE_ADJOINT_UNBALANCED_MESSAGE = """
The number of algebraic equations (rows of ∂F/∂(du) that are identically zero)
does not match the number of algebraic variables (from `differential_vars`, or
columns of ∂F/∂(du) that are identically zero). Adjoint sensitivity analysis for
`DAEProblem` requires a square algebraic subsystem. If `differential_vars` was not
provided to the `DAEProblem`, provide it explicitly so that the algebraic
variables can be identified structurally.
"""

const DAE_ADJOINT_HIGH_INDEX_MESSAGE = """
The algebraic subsystem of the DAE could not be classified as index-1 (∂F_alg/∂u_alg
nonsingular) or Hessenberg index-2 (∂F_alg/∂u_alg ≡ 0 with
∂F_alg/∂u_diff * (∂F_diff/∂(du)_diff)⁻¹ * ∂F_diff/∂u_alg nonsingular) at the
terminal time of the forward solution. Mixed index-1/index-2 systems and DAEs of
index 3 or higher are not currently supported by the `DAEProblem` adjoint.
Consider index-reducing the system (e.g. via ModelingToolkit's `structural_simplify`)
before differentiating it.
"""

# Classify the DAE at (dy, y, p, t). The zero rows/columns of ∂F/∂(du) determine the
# algebraic equations/variables; `differential_vars` from the DAEProblem takes
# precedence for the variables when provided. Returns
# (diffvar_idxs, algevar_idxs, diffeq_idxs, algeeq_idxs, index2).
function dae_adjoint_structure(f, isinplace, dy, y, p, t, differential_vars)
    n = length(y)
    J_du = dae_du_jacobian(f, isinplace, dy, y, p, t)
    algeeq_idxs = findall(i -> all(iszero, @view(J_du[i, :])), 1:n)
    algevar_idxs = if differential_vars === nothing
        findall(j -> all(iszero, @view(J_du[:, j])), 1:n)
    else
        findall(!, collect(differential_vars))
    end
    diffeq_idxs = setdiff(1:n, algeeq_idxs)
    diffvar_idxs = setdiff(1:n, algevar_idxs)
    length(algeeq_idxs) == length(algevar_idxs) ||
        error(DAE_ADJOINT_UNBALANCED_MESSAGE)

    index2 = false
    if !isempty(algevar_idxs)
        J_u = dae_u_jacobian(f, isinplace, dy, y, p, t)
        dhda = J_u[algeeq_idxs, algevar_idxs]
        if !issuccess(lu(dhda, check = false))
            all(iszero, dhda) || error(DAE_ADJOINT_HIGH_INDEX_MESSAGE)
            Fdd = lu(J_du[diffeq_idxs, diffvar_idxs], check = false)
            issuccess(Fdd) || error(DAE_ADJOINT_HIGH_INDEX_MESSAGE)
            N = (Fdd \ J_u[diffeq_idxs, algevar_idxs])'
            NB = N * J_u[algeeq_idxs, diffvar_idxs]'
            issuccess(lu(NB, check = false)) || error(DAE_ADJOINT_HIGH_INDEX_MESSAGE)
            index2 = true
        end
    end
    return diffvar_idxs, algevar_idxs, diffeq_idxs, algeeq_idxs, index2
end

struct DAEAdjointSensitivityFunction{
        C <: AdjointDiffCache, Alg <: AbstractAdjointSensitivityAlgorithm,
        uType, duType, SType, pType, fType,
    } <: SensitivityFunction
    diffcache::C
    sensealg::Alg
    discrete::Bool
    y::uType
    dy::duType
    sol::SType
    prob::pType
    f::fType
end

function DAEAdjointSensitivityFunction(
        g, sensealg, discrete, sol, dgdu, dgdp, f, alg;
        quad = false
    )
    diffcache, y = adjointdiffcache(
        g, sensealg, discrete, sol, dgdu, dgdp, f, alg; quad
    )
    dy = similar(y)
    sol(dy, sol.prob.tspan[2], Val{1})
    return DAEAdjointSensitivityFunction(
        diffcache, sensealg, discrete, y, dy, sol, sol.prob, f
    )
end

# z = [w; λ] (+ grad for the interpolating adjoint), solved backwards. `dz` is used
# as scratch: the u-vjp is written into the w rows and the du-vjp into the λ rows,
# then both are rearranged in place into the residual layout derived at the top of
# this file.
function (S::DAEAdjointSensitivityFunction)(dz, z, p, t)
    (; y, dy, sol, discrete, diffcache) = S
    (; diffvar_idxs, algevar_idxs) = diffcache

    n = length(y)
    if t isa ForwardDiff.Dual && eltype(y) <: AbstractFloat
        _y = sol(t, continuity = :right)
        _dy = sol(t, Val{1}, continuity = :right)
    else
        sol(y, t, continuity = :right)
        sol(dy, t, Val{1}, continuity = :right)
        _y = y
        _dy = dy
    end

    w = @view z[1:n]
    λ = @view z[(n + 1):(2n)]
    vjp_u = @view dz[1:n]
    vjp_du = @view dz[(n + 1):(2n)]
    dgrad = length(dz) > 2n ? @view(dz[(2n + 1):length(dz)]) : nothing

    vecjacobian!(vjp_u, _y, λ, p, t, S; dgrad, du = _dy, ddu = vjp_du)

    # Continuous cost enters both the ẇ rows and the algebraic λ rows through the
    # same g_u subtraction, so apply it before the rows are rearranged.
    discrete || accumulate_cost!(vjp_u, _y, p, t, S, dgrad)

    @inbounds for j in algevar_idxs
        dz[n + j] = dz[j] # 0 = [F_u' λ]_j - g_u[j]
        dz[j] = w[j]      # 0 = w_j
    end
    @inbounds for j in diffvar_idxs
        dz[n + j] = w[j] - dz[n + j] # 0 = w_j - [F_du' λ]_j
    end
    return nothing
end

# Fully implicit residual form of the augmented adjoint, M ż - G(z), so the
# adjoint can be solved as a `DAEProblem` by the same DAE algorithm as the forward
# pass (e.g. `DFBDF`). `mmdiag` is the diagonal of the augmented mass matrix.
struct DAEAdjointResidual{S <: DAEAdjointSensitivityFunction, M} <:
    SensitivityFunction
    S::S
    mmdiag::M
end

function (R::DAEAdjointResidual)(res, dz, z, p, t)
    R.S(res, z, p, t)
    @. res = R.mmdiag * dz - res
    return nothing
end

@doc doc"""
```julia
DAEAdjointProblem(sol, sensealg, alg, t = nothing,
                  dgdu_discrete = nothing, dgdp_discrete = nothing,
                  dgdu_continuous = nothing, dgdp_continuous = nothing,
                  g = nothing; kwargs...)
```

Constructs the adjoint problem for a fully implicit `DAEProblem` solution `sol`
(`F(du, u, p, t) = 0`), using the augmented adjoint DAE formulation of Cao, Li,
Petzold & Serban (SIAM Journal on Scientific Computing 24(3), 2003). The augmented
adjoint state is `z = [w; λ]` with `w = (∂F/∂(du))' λ` (plus the parameter-gradient
quadrature block for `InterpolatingAdjoint`). The returned problem is a
singular-mass-matrix `ODEProblem` when `alg` is a mass-matrix-capable stiff ODE
solver (e.g. `FBDF`, `Rodas5P`), or a fully implicit `DAEProblem` when `alg` is a
DAE solver (e.g. `DFBDF`), solved backwards in time. When `alg` is `nothing` the
algorithm of the forward solution decides, so a forward solve with the default DAE
algorithm gets a `DAEProblem`.

The arguments mirror `ODEAdjointProblem`, with methods for `InterpolatingAdjoint`,
`QuadratureAdjoint`, and `GaussAdjoint`/`GaussKronrodAdjoint` (which mirrors the
Gauss `ODEAdjointProblem` signature and returns the jump callback separately).
Index-1 DAEs are fully supported, including discrete and continuous cost
contributions on the algebraic variables. Hessenberg index-2 DAEs are supported
by `InterpolatingAdjoint` with cost contributions on the differential variables
(the consistent boundary jumps are obtained by projecting onto the adjoint
constraint manifold); the quadrature-style methods reject them, since the
impulsive index-2 adjoint multiplier at the cost times is only captured by the
state-augmented gradient accumulation.
Higher-index and mixed index-1/index-2 systems are rejected; index-reduce such
systems first. The forward solution must be dense (checkpointing is not currently
supported) and its interpolant must support first-derivative evaluation
`sol(t, Val{1})`, which holds for the native Julia DAE solvers such as `DFBDF`.
"""
@noinline function DAEAdjointProblem(
        sol, sensealg::InterpolatingAdjoint, alg,
        t = nothing,
        dgdu_discrete::DG1 = nothing,
        dgdp_discrete::DG2 = nothing,
        dgdu_continuous::DG3 = nothing,
        dgdp_continuous::DG4 = nothing,
        g::G = nothing,
        ::Val{RetCB} = Val(false);
        checkpoints = current_time(sol),
        callback = CallbackSet(), no_start = false,
        reltol = nothing, abstol = nothing,
        kwargs...
    ) where {DG1, DG2, DG3, DG4, G, RetCB}
    ischeckpointing(sensealg, sol) &&
        error("Checkpointing is not currently supported for DAEProblem adjoints. Use a dense forward solution with `InterpolatingAdjoint(checkpointing = false)`.")
    adj_prob, cb, rcb = _dae_adjoint_problem(
        sol, sensealg, alg, t, dgdu_discrete, dgdp_discrete,
        dgdu_continuous, dgdp_continuous, g;
        withgrad = true, attach_callback = true, callback, no_start
    )
    if RetCB
        return adj_prob, rcb
    else
        return adj_prob
    end
end

@noinline function DAEAdjointProblem(
        sol, sensealg::QuadratureAdjoint, alg,
        t = nothing,
        dgdu_discrete::DG1 = nothing,
        dgdp_discrete::DG2 = nothing,
        dgdu_continuous::DG3 = nothing,
        dgdp_continuous::DG4 = nothing,
        g::G = nothing,
        ::Val{RetCB} = Val(false); no_start = false,
        callback = CallbackSet()
    ) where {DG1, DG2, DG3, DG4, G, RetCB}
    adj_prob, cb, rcb = _dae_adjoint_problem(
        sol, sensealg, alg, t, dgdu_discrete, dgdp_discrete,
        dgdu_continuous, dgdp_continuous, g;
        withgrad = false, attach_callback = true, callback, no_start
    )
    if RetCB
        return adj_prob, rcb
    else
        return adj_prob
    end
end

@noinline function DAEAdjointProblem(
        sol, sensealg::AbstractGAdjoint, alg,
        GaussInt::GaussIntegrand, integrating_cb,
        t = nothing,
        dgdu_discrete::DG1 = nothing,
        dgdp_discrete::DG2 = nothing,
        dgdu_continuous::DG3 = nothing,
        dgdp_continuous::DG4 = nothing,
        g::G = nothing,
        ::Val{RetCB} = Val(false);
        checkpoints = current_time(sol),
        callback = CallbackSet(), no_start = false,
        reltol = nothing, abstol = nothing, kwargs...
    ) where {DG1, DG2, DG3, DG4, G, RetCB}
    ischeckpointing(sensealg, sol) &&
        error("Checkpointing is not currently supported for DAEProblem adjoints.")
    # The Gauss integrating callback is combined with the jump callback by the
    # caller, so do not attach the jump callback to the problem.
    adj_prob, cb, rcb = _dae_adjoint_problem(
        sol, sensealg, alg, t, dgdu_discrete, dgdp_discrete,
        dgdu_continuous, dgdp_continuous, g;
        withgrad = false, attach_callback = !RetCB, callback, no_start
    )
    return adj_prob, cb, rcb
end

function _dae_adjoint_problem(
        sol, sensealg, alg, t, dgdu_discrete, dgdp_discrete,
        dgdu_continuous, dgdp_continuous, g;
        withgrad, attach_callback, callback, no_start
    )
    dgdu_discrete === nothing && dgdu_continuous === nothing && g === nothing &&
        error("Either `dgdu_discrete`, `dgdu_continuous`, or `g` must be specified.")
    t !== nothing && dgdu_discrete === nothing && dgdp_discrete === nothing &&
        error("It looks like you're using the direct `adjoint_sensitivities` interface
               with a discrete cost function but no specified `dgdu_discrete` or `dgdp_discrete`.
               Please use the higher level `solve` interface or specify these two contributions.")
    if callback !== nothing &&
            !(
            callback isa CallbackSet && isempty(callback.continuous_callbacks) &&
                isempty(callback.discrete_callbacks)
        )
        error("Callbacks are not currently supported for DAEProblem adjoints.")
    end
    # No algorithm means the forward solve used the default DAE algorithm; keep the adjoint in
    # residual form for it. The mass-matrix form under the default ODE algorithm rebuilds its
    # Jacobian at every step and does not finish on moderately sized problems.
    alg === nothing && (alg = sol.alg)

    (; tspan) = sol.prob
    p = parameter_values(sol.prob)
    u0 = state_values(sol.prob)

    if p === nothing || p isa SciMLBase.NullParameters
        tunables, repack = p, identity
    else
        tunables, repack, _ = canonicalize(Tunable(), p)
    end

    terminated = false
    if hasfield(typeof(sol), :retcode)
        if sol.retcode == ReturnCode.Terminated
            tspan = (tspan[1], last(current_time(sol)))
            terminated = true
        end
    end
    tspan = reverse(tspan)

    discrete = (
        t !== nothing &&
            (
            dgdu_continuous === nothing && dgdp_continuous === nothing ||
                g !== nothing
        )
    )

    numstates = length(u0)
    numparams = p === nothing || p === SciMLBase.NullParameters() ? 0 : length(tunables)

    λ = if withgrad
        p === nothing || p === SciMLBase.NullParameters() ? similar(u0) :
            one(eltype(u0)) .* similar(tunables, numstates + numparams)
    else
        similar(u0, numstates)
    end
    λ .= false

    sense = DAEAdjointSensitivityFunction(
        g, sensealg, discrete, sol,
        dgdu_continuous, dgdp_continuous,
        sol.prob.f, alg; quad = !withgrad
    )

    if !withgrad && sense.diffcache.index2
        error(
            "`$(nameof(typeof(sensealg)))` does not support Hessenberg index-2 DAEs: " *
                "the index-2 adjoint multiplier is impulsive at discrete cost times " *
                "and its parameter-gradient contribution is concentrated at the " *
                "quadrature interval endpoints, where it is not captured by a " *
                "quadrature over the adjoint interpolant. Use `InterpolatingAdjoint`, " *
                "whose state-augmented gradient accumulation integrates it correctly."
        )
    end

    init_cb = (discrete || dgdu_discrete !== nothing)
    cb, rcb, _ = generate_callbacks(
        sense, dgdu_discrete, dgdp_discrete,
        λ, t, tspan[2], callback, init_cb, terminated, no_start
    )

    len = withgrad ? 2 * numstates + numparams : 2 * numstates
    z0 = similar(λ, len)
    z0 .= false

    mmdiag = zeros(eltype(z0), len)
    mmdiag[sense.diffcache.diffvar_idxs] .= one(eltype(z0))
    withgrad && (mmdiag[(2 * numstates + 1):len] .= one(eltype(z0)))

    # The algebraic λ rows must be (re-)solved consistently at the terminal time and
    # after every discrete-cost jump. `BrownFullBasicInit` does exactly that (it
    # modifies only the algebraic variables), and its initialization system is the
    # adjoint consistency condition of Cao et al. For Hessenberg index-2 DAEs that
    # system is singular (the index-2 adjoint multiplier only appears under
    # differentiation). The natural next choice, `ShampineCollocationInit` (a BDF1
    # collocation consistency solve), is *not* usable here: it modifies all
    # variables, including the differential adjoint components `w`/`grad` whose
    # terminal (and post-jump) values are prescribed boundary conditions
    # (`w(T) = 0`, and the jump sets them), so it corrupts the boundary data and
    # returns the wrong gradient. Instead use `NoInit`: `z(T) = 0` is already
    # consistent for the homogeneous augmented system, and after each jump the
    # `derivative_discontinuity!` in the callback lets the BDF discretization
    # resolve the index-2 multiplier from the following step (the terminal `λ`
    # value itself is never read by the gradient extraction).
    initializealg = sense.diffcache.index2 ? SciMLBase.NoInit() :
        BrownFullBasicInit()

    adj_prob = if alg isa SciMLBase.AbstractDAEAlgorithm
        # Solve the augmented adjoint in fully implicit residual form with the
        # user's DAE algorithm.
        resid = DAEAdjointResidual(sense, mmdiag)
        diffvars_aug = mmdiag .== one(eltype(z0))
        dz0 = zero(z0)
        daefun = DAEFunction{true, true}(resid)
        # Automatic initial-dt selection does not support backwards-in-time
        # DAEProblems, so seed the reverse solve with the forward solution's
        # last step size.
        fwd_t = current_time(sol)
        dt0 = length(fwd_t) > 1 ?
            sign(tspan[2] - tspan[1]) * abs(fwd_t[end] - fwd_t[end - 1]) :
            (tspan[2] - tspan[1]) / 1000
        if attach_callback
            DAEProblem(
                daefun, dz0, z0, tspan, p;
                differential_vars = diffvars_aug, callback = cb, initializealg,
                dt = dt0
            )
        else
            DAEProblem(
                daefun, dz0, z0, tspan, p;
                differential_vars = diffvars_aug, initializealg, dt = dt0
            )
        end
    else
        odefun = ODEFunction{true, true}(sense, mass_matrix = Diagonal(mmdiag))
        if attach_callback
            ODEProblem(odefun, z0, tspan, p; callback = cb, initializealg)
        else
            ODEProblem(odefun, z0, tspan, p; initializealg)
        end
    end
    return adj_prob, cb, rcb
end

function DAEAdjointProblem(sol, sensealg::AbstractAdjointSensitivityAlgorithm, args...; kwargs...)
    error(
        """
        `$(nameof(typeof(sensealg)))` is not currently supported for adjoint sensitivity
        analysis of fully implicit `DAEProblem`s. Supported methods are
        `InterpolatingAdjoint`, `QuadratureAdjoint`, `GaussAdjoint`, and
        `GaussKronrodAdjoint`, e.g.

            adjoint_sensitivities(sol, alg; sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP()), ...)

        Alternatively, express the DAE in mass-matrix form as an `ODEProblem`
        (e.g. via ModelingToolkit), for which all continuous adjoint methods are
        supported.
        """
    )
end

# Shared parameter-vjp of the DAE residual for the Quadrature/Gauss integrands:
# out = (∂F/∂p)' λ at the interpolated (du(t), u(t)). `S` is an
# `AdjointSensitivityIntegrand` or a `GaussIntegrand` whose `dy` field is the
# derivative interpolation buffer (filled here).
function dae_vec_pjac!(out, λ, y, t, S)
    (; paramjac_config, sensealg, sol, tunables, repack, dy) = S
    f = S.unwrappedf
    isinplace = DiffEqBase.isinplace(sol.prob)
    sol(dy, t, Val{1})

    if sensealg.autojacvec isa ReverseDiffVJP
        tape = paramjac_config
        tdu, tu, tp, tt = ReverseDiff.input_hook(tape)
        output = ReverseDiff.output_hook(tape)
        ReverseDiff.unseed!(tdu)
        ReverseDiff.unseed!(tu)
        ReverseDiff.unseed!(tp)
        ReverseDiff.unseed!(tt)
        ReverseDiff.value!(tdu, dy)
        ReverseDiff.value!(tu, y)
        ReverseDiff.value!(tp, tunables)
        ReverseDiff.value!(tt, [t])
        ReverseDiff.forward_pass!(tape)
        ReverseDiff.increment_deriv!(output, λ)
        ReverseDiff.reverse_pass!(tape)
        copyto!(vec(out), ReverseDiff.deriv(tp))
    elseif sensealg.autojacvec isa ZygoteVJP
        isinplace &&
            error("`ZygoteVJP` requires an out-of-place DAE residual `f(du, u, p, t)`. Use `ReverseDiffVJP()` for in-place residuals.")
        _, back = Zygote.pullback(tunables) do _tunables
            vec(f(dy, y, repack(_tunables), t))
        end
        tmp = back(λ)
        if tmp[1] === nothing
            recursive_copyto!(out, 0)
        else
            recursive_copyto!(out, tmp[1])
        end
    elseif sensealg.autojacvec isa EnzymeVJP
        tunables isa AbstractArray ||
            error("`EnzymeVJP` currently requires `AbstractArray` parameters for DAEProblem adjoints. Use `ReverseDiffVJP()` for structured parameters.")
        if isinplace
            res_primal, res_shadow, f_shadow = paramjac_config
            vec(res_shadow) .= vec(λ)
            Enzyme.remake_zero!(res_primal)
            Enzyme.remake_zero!(out)
            vf = SciMLBase.Void(f)
            fdup = if Base.issingletontype(typeof(vf))
                Enzyme.Const(vf)
            else
                Enzyme.remake_zero!(f_shadow)
                Enzyme.Duplicated(vf, f_shadow)
            end
            Enzyme.autodiff(
                sensealg.autojacvec.mode,
                fdup, Enzyme.Const,
                Enzyme.Duplicated(res_primal, res_shadow),
                Enzyme.Const(dy), Enzyme.Const(y),
                Enzyme.Duplicated(tunables, out), Enzyme.Const(t)
            )
        else
            Enzyme.remake_zero!(out)
            Enzyme.autodiff(
                sensealg.autojacvec.mode, Enzyme.Const(_enzyme_dae_vecpjac_dot),
                Enzyme.Active, Enzyme.Const(f), Enzyme.Const(repack),
                Enzyme.Const(dy), Enzyme.Const(y),
                Enzyme.Duplicated(tunables, out),
                Enzyme.Const(t), Enzyme.Const(λ)
            )
        end
    elseif sensealg.autojacvec isa Bool && !sensealg.autojacvec
        pJ = dae_p_jacobian(f, isinplace, dy, y, p_for_jacobian(S), t, tunables, repack)
        mul!(vec(out), pJ', λ)
    else
        error(
            "$(nameof(typeof(sensealg.autojacvec))) is not currently supported for DAEProblem adjoints with `$(nameof(typeof(sensealg)))`. " *
                "Use `ReverseDiffVJP()`, `EnzymeVJP()`, `ZygoteVJP()`, or `autojacvec = false`."
        )
    end
    return out
end

p_for_jacobian(S) = S.p

function _enzyme_dae_vecpjac_dot(f, repack, du, y, tunables, t, λ)
    return dot(vec(f(du, y, repack(tunables), t)), vec(λ))
end

# Builds the parameter-vjp configuration for the Quadrature/Gauss integrands over
# a DAE residual. Returns (pf, pJ, paramjac_config).
function dae_integrand_paramjac_config(sensealg, prob, unwrappedf, y, dy, tunables, repack, T)
    return if sensealg.autojacvec isa ReverseDiffVJP
        tape = if DiffEqBase.isinplace(prob)
            ReverseDiff.GradientTape((dy, y, tunables, [T])) do du, u, tunables, t
                res = similar(tunables, size(u))
                res .= false
                unwrappedf(res, du, u, repack(tunables), first(t))
                return vec(res)
            end
        else
            ReverseDiff.GradientTape((dy, y, tunables, [T])) do du, u, tunables, t
                vec(unwrappedf(du, u, repack(tunables), first(t)))
            end
        end
        paramjac_config = compile_tape(sensealg.autojacvec) ? ReverseDiff.compile(tape) :
            tape
        nothing, nothing, paramjac_config
    elseif sensealg.autojacvec isa EnzymeVJP
        nothing, nothing, (zero(y), zero(y), Enzyme.make_zero(SciMLBase.Void(unwrappedf)))
    else
        # ZygoteVJP and `autojacvec = false` need no cached configuration; other
        # backends error inside `dae_vec_pjac!`.
        nothing, nothing, nothing
    end
end
