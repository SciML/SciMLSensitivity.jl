struct NILSASSensitivityFunction{iip, NILSS, ASF, Mtype} <:
       AbstractODEFunction{iip}
    nilss::NILSS
    S::ASF # Adjoint sensitivity function
    M::Mtype
    discrete::Bool
end

struct QuadratureCache{A1, A2, A3, A4, A5}
    dwv::A1
    dwf::A1
    dwfs::A2
    dvf::A3
    dvfs::A4
    dJs::A4
    C::A5
    R::A5
    b::A1
end

function QuadratureCache(u0, M, nseg, numparams)
    dwv = Array{eltype(u0)}(undef, M, nseg)
    dwf = Array{eltype(u0)}(undef, M, nseg)
    dwfs = Array{eltype(u0)}(undef, numparams * M, nseg)
    dvf = Array{eltype(u0)}(undef, 1, nseg)
    dvfs = Array{eltype(u0)}(undef, numparams, nseg)
    dJs = Array{eltype(u0)}(undef, numparams, nseg)
    C = Array{eltype(u0)}(undef, M, M, nseg)
    R = Array{eltype(u0)}(undef, M, M, nseg)
    b = Array{eltype(u0)}(undef, M, nseg)

    QuadratureCache{typeof(dwv), typeof(dwfs), typeof(dvf), typeof(dvfs), typeof(C)}(dwv,
        dwf,
        dwfs,
        dvf,
        dvfs,
        dJs, C,
        R, b)
end

struct NILSASProblem{A, NILSS, Aprob, Qcache, solType, z0Type, tType, G, T, DG1, DG2}
    sensealg::A
    nilss::NILSS # diffcache
    adjoint_prob::Aprob
    quadcache::Qcache
    sol::solType
    z0::z0Type
    t::tType
    g::G
    T_seg::T
    dtsave::T
    dgdu_discrete::DG1
    dgdp_discrete::DG2
end

function NILSASProblem(sol, sensealg::NILSAS, alg;
        t = nothing, dgdu_discrete = nothing, dgdp_discrete = nothing,
        dgdu_continuous = nothing, dgdp_continuous = nothing, g = sensealg.g,
        kwargs...)
    (; tspan, f) = sol.prob
    p = parameter_values(sol.prob)
    u0 = state_values(sol.prob)
    tunables, repack, aliases = canonicalize(Tunable(), p)
    (; nseg, nstep, rng, adjoint_sensealg, M) = sensealg  #number of segments on time interval, number of steps saved on each segment

    numindvar = length(u0)
    numparams = length(tunables)

    # some shadowing sensealgs require knowledge of g
    check_for_g(sensealg, g)

    # sensealg choice
    adjoint_sensealg === nothing &&
        (adjoint_sensealg = automatic_sensealg_choice(sol.prob, u0, tunables, false, repack))

    p === nothing &&
        error("You must have parameters to use parameter sensitivity calculations!")
    !(u0 isa AbstractVector) && error("`u` has to be an AbstractVector.")

    nstep <= 1 &&
        error("At least the start and the end point of each segment must be stored. Please use `nstep >=2`.")

    !(u0 isa AbstractVector) && error("`u` has to be an AbstractVector.")

    # segmentation: determine length of segmentation and spacing between saved points
    T_seg = (tspan[2] - tspan[1]) / nseg # length of each segment
    dtsave = T_seg / (nstep - 1)

    # check that it's either discrete or continuous
    if t !== nothing
        @assert dgdu_discrete !== nothing || dgdp_discrete !== nothing || g !== nothing
        error("`NILSAS` can currently not be used with discrete cost functions.")
    else
        @assert dgdu_continuous !== nothing || dgdp_continuous !== nothing || g !== nothing
    end

    # homogeneous + inhomogeneous adjoint sensitivity problem
    # assign initial values to y, vstar, w
    y = copy(last(state_values(sol)))
    z0 = terminate_conditions(adjoint_sensealg, rng, f, y, tunables, tspan[2], numindvar,
        numparams, M)
    nilss = NILSSSensitivityFunction(sensealg, f, u0, tunables, tspan, g, dgdu_continuous,
        dgdp_continuous)
    tspan = (tspan[2] - T_seg, tspan[2])
    checkpoints = tspan[1]:dtsave:tspan[2]

    adjoint_prob = ODEAdjointProblem(sol, adjoint_sensealg, alg, t, dgdu_discrete,
        dgdp_discrete,
        dgdu_continuous, dgdp_continuous,
        g;
        checkpoints = checkpoints,
        z0 = z0, M = M, nilss = nilss, tspan = tspan,
        kwargs...)

    # pre-allocate variables for integration Eq.(23) in NILSAS paper.
    quadcache = QuadratureCache(u0, M, nseg, numparams)

    NILSASProblem{typeof(sensealg), typeof(nilss), typeof(adjoint_prob), typeof(quadcache),
        typeof(sol), typeof(z0), typeof(t), typeof(g), typeof(T_seg),
        typeof(dgdu_discrete), typeof(dgdp_discrete)}(sensealg,
        nilss,
        adjoint_prob,
        quadcache,
        sol,
        deepcopy(z0),
        t,
        g, T_seg,
        dtsave, dgdu_discrete,
        dgdp_discrete)
end

function terminate_conditions(alg::BacksolveAdjoint, rng, f, y, p, t, numindvar, numparams,
        M)
    if isinplace(f)
        f_unit = zero(y)
        f(f_unit, y, p, t)
        normalize!(f_unit)
    else
        f_unit = f(y, p, t)
        normalize!(f_unit)
    end

    if M > 1
        W = rand(rng, numindvar, M - 1)
        W .-= (f_unit' * W) .* f_unit
        w, _ = qr(W)
        _w = @view w[:, 1:(M - 1)]
        W = hcat(_w, f_unit)
    else
        W = f_unit
    end
    vst = zeros(numindvar)

    # quadratures
    C = zeros(M, M)
    dwv = zeros(M)
    dwf = zeros(M)
    dvf = zeros(1)
    dJs = zeros(numparams)

    return ArrayPartition([vst; vec(W)], zeros(numparams * (M + 1)), y, C, dwv, dwf, dvf,
        dJs)
end

function split_states(du, u, t, NS::NILSASSensitivityFunction, j; update = true)
    (; nilss, S) = NS
    (; numindvar, numparams) = nilss

    indx1 = (j - 1) * (numindvar) + 1
    indx2 = indx1 + (numindvar - 1)
    indx3 = (j - 1) * (numparams) + 1
    indx4 = indx3 + (numparams - 1)

    λ = @view u.x[1][indx1:indx2]
    grad = @view u.x[2][indx3:indx4]
    _y = u.x[3]

    # like ODE/Drift term and scalar noise
    dλ = @view du.x[1][indx1:indx2]
    dgrad = @view du.x[2][indx3:indx4]
    dy = du.x[3]

    λ, grad, _y, dλ, dgrad, dy
end

function split_quadratures(du, u, t, NS::NILSASSensitivityFunction; update = true)
    (; nilss, S) = NS
    (; numindvar, numparams) = nilss

    C = u.x[4]
    dwv = u.x[5]
    dwf = u.x[6]
    dvf = u.x[7]
    dJs = u.x[8]

    dC = du.x[4]
    ddwv = du.x[5]
    ddwf = du.x[6]
    ddvf = du.x[7]
    ddJs = du.x[8]

    dC, ddwv, ddwf, ddvf, ddJs, C, dwv, dwf, dvf, dJs
end

function (NS::NILSASSensitivityFunction)(du, u, p, t)
    (; nilss, S, M) = NS
    (; f, dg_val, pgpu, pgpu_config, pgpp, pgpp_config, numparams, numindvar, alg) = nilss
    (; y, discrete) = S

    λ, _, _y, dλ, dgrad, dy = split_states(du, u, t, NS, 1)
    copyto!(vec(y), _y)

    #  compute gradient of objective wrt. state
    if !discrete
        accumulate_cost!(y, p, t, nilss)
    end

    # loop over all adjoint states
    for j in 1:(M + 1)
        λ, _, _, dλ, dgrad, dy = split_states(du, u, t, NS, j)
        vecjacobian!(dλ, y, λ, p, t, S, dgrad = dgrad, dy = dy)
        dλ .*= -1
        dgrad .*= -1

        if j == 1
            # j = 1 is the inhomogeneous adjoint solution
            if !discrete
                if dg_val isa Tuple
                    dλ .-= vec(dg_val[1])
                else
                    dλ .-= vec(dg_val)
                end
            end
        end
    end

    # quadratures
    dC, ddwv, ddwf, ddvf, ddJs, _, _, _, _, _ = split_quadratures(du, u, t, NS)
    # j = 1 is the inhomogeneous adjoint solution
    λv, _, _, _, _, dy = split_states(du, u, t, NS, 1)
    ddvf .= -dot(λv, dy)
    for j in 1:M
        λ, _, _, _, _, _ = split_states(du, u, t, NS, j + 1)
        ddwf[j] = -dot(λ, dy)
        ddwv[j] = -dot(λ, λv)
        for i in (j + 1):M
            λ2, _, _, _, _, _ = split_states(du, u, t, NS, i + 1)
            dC[j, i] = -dot(λ, λ2)
            dC[i, j] = dC[j, i]
        end
        dC[j, j] = -dot(λ, λ)
    end

    if dg_val isa Tuple && !discrete
        ddJs .= -vec(dg_val[2])
    end

    return nothing
end

function accumulate_cost!(y, p, t, nilss::NILSSSensitivityFunction)
    (; dgdu, dgdp, dg_val, pgpu, pgpu_config, pgpp, pgpp_config, alg) = nilss

    if dgdu === nothing
        if dg_val isa Tuple
            gradient!(dg_val[1], pgpu, y, alg, pgpu_config)
            gradient!(dg_val[2], pgpp, y, alg, pgpp_config)
        else
            gradient!(dg_val, pgpu, y, alg, pgpu_config)
        end
    else
        if dg_val isa Tuple
            dgdu(dg_val[1], y, p, t)
            dgdp(dg_val[2], y, p, t)
        else
            dgdu(dg_val, y, p, t)
        end
    end

    return nothing
end

function adjoint_sense(prob::NILSASProblem, nilsas::NILSAS, alg; kwargs...)
    (; M, nseg, nstep, adjoint_sensealg) = nilsas
    (; sol, nilss, z0, t, dgdu_discrete, dgdp_discrete, g, T_seg, dtsave, adjoint_prob) = prob
    (; u0, tspan) = adjoint_prob
    (; dgdu, dgdp) = nilss

    copyto!(z0, u0)

    @assert haskey(adjoint_prob.kwargs, :callback)
    # get loss callback
    cb = adjoint_prob.kwargs[:callback]

    # adjoint sensitivities on segments
    for iseg in nseg:-1:1
        t1 = tspan[1] - (nseg - iseg + 1) * T_seg
        t2 = tspan[1] - (nseg - iseg) * T_seg
        checkpoints = t1:dtsave:t2
        _prob = ODEAdjointProblem(sol, adjoint_sensealg, alg, t, dgdu_discrete,
            dgdp_discrete,
            dgdu, dgdp, g;
            checkpoints = checkpoints, z0 = z0, M = M, nilss = nilss,
            tspan = (t1, t2), kwargs...)
        _sol = solve(_prob, alg; save_everystep = false, save_start = false,
            saveat = eltype(state_values(sol.prob))[],
            dt = dtsave,
            tstops = checkpoints,
            callback = cb,
            kwargs...)

        # renormalize at interfaces and store quadratures
        # update sense problem
        renormalize!(prob, _sol, z0, iseg)
    end
    return nothing
end

function renormalize!(prob::NILSASProblem, sol, z0, iseg)
    (; quadcache, nilss, sensealg) = prob
    (; M) = sensealg
    (; numparams, numindvar) = nilss
    (; R, b) = quadcache

    x = state_values(sol)[end].x
    # vstar_right (inhomogeneous adjoint on the rhs of the interface)
    vstar = @view x[1][1:numindvar]
    # homogeneous adjoint on the rhs of the interface
    W = @view x[1][(numindvar + 1):end]
    W = reshape(W, numindvar, M)

    Q_, R_ = qr(W)
    Q = @view Q_[:, 1:M]
    b_ = (Q' * vstar)

    # store R and b to solve NILSAS problem
    copyto!((@view R[:, :, iseg]), R_)
    copyto!((@view b[:, iseg]), b_)

    # store quadrature values
    store_quad(quadcache, x, numparams, iseg)

    # reset z0
    reset!(z0, numindvar, vstar, b_, Q)

    return nothing
end

function store_quad(quadcache, x, numparams, iseg)
    (; dwv, dwf, dwfs, dvf, dvfs, dJs, C) = quadcache

    grad_vfs = @view x[2][1:numparams]
    copyto!((@view dvfs[:, iseg]), grad_vfs)

    grad_wfs = @view x[2][(numparams + 1):end]
    copyto!((@view dwfs[:, iseg]), grad_wfs)

    # C_i = x[4]
    copyto!((@view C[:, :, iseg]), x[4])
    # dwv_i = x[5]
    copyto!((@view dwv[:, iseg]), x[5])
    # dwf_i = x[6]
    copyto!((@view dwf[:, iseg]), x[6])
    # dvf_i = x[7]
    copyto!((@view dvf[:, iseg]), x[7])
    # dJs_i = x[8]
    copyto!((@view dJs[:, iseg]), x[8])
    return nothing
end

function reset!(z0, numindvar, vstar, b, Q)
    # modify z0
    x0 = z0.x

    # vstar_left
    v = @view x0[1][1:numindvar]
    v .= vstar - vec(b' * Q')

    # W_left (homogeneous adjoint on lhs of the interface)
    w = @view x0[1][(numindvar + 1):end]
    w .= vec(Q)

    # reset all other values t0 zero
    x0[2] .*= false
    x0[4] .*= false
    x0[5] .*= false
    x0[6] .*= false
    x0[7] .*= false
    x0[8] .*= false
    return nothing
end

function nilsas_min(cache::QuadratureCache)
    (; dwv, dwf, dvf, C, R, b) = cache

    # Construct Schur complement of the Lagrange multiplier method of the NILSAS problem.

    # see description in Appendix A of Nilsas paper.
    # M= # unstable CLVs, K = # segments
    M, K = size(dwv)

    # construct Cinv
    # Cinv is a block diagonal matrix
    Cinv = zeros(eltype(C), M * K, M * K)

    for i in 1:K
        Ci = @view C[:, :, i]
        _Cinv = @view Cinv[((i - 1) * M + 1):(i * M), ((i - 1) * M + 1):(i * M)]
        Ciinv = inv(Ci)
        copyto!(_Cinv, Ciinv)
    end

    # construct B, also very sparse if K >> M
    B = zeros(eltype(C), M * K - M + 1, M * K)

    for i in 1:K
        if i < K
            # off diagonal Rs
            _B = @view B[((i - 1) * M + 1):(i * M), (i * M + 1):((i + 1) * M)]
            _R = @view R[:, :, i + 1]
            copyto!(_B, _R)
            _B .*= -1

            # diagonal ones
            for j in 1:M
                B[(i - 1) * M + j, (i - 1) * M + j] = one(eltype(R))
            end
        end
        # last row
        dwfi = dwf[:, i]
        _B = @view B[end, ((i - 1) * M + 1):(i * M)]
        copyto!(_B, dwfi)
    end

    # construct d
    d = vec(dwv)

    # construct b
    _b = [b[(M + 1):end]; -sum(dvf)]

    # compute Lagrange multiplier
    λ = (-B * Cinv * B') \ (B * Cinv * d + _b)

    # return a
    return reshape(-Cinv * (B' * λ + d), M, K)
end

function shadow_adjoint(prob::NILSASProblem, alg; sensealg = prob.sensealg, kwargs...)
    shadow_adjoint(prob, sensealg, alg; kwargs...)
end

function shadow_adjoint(prob::NILSASProblem, sensealg::NILSAS, alg; kwargs...)

    # compute adjoint sensitivities
    adjoint_sense(prob, sensealg, alg; kwargs...)

    # compute NILSAS problem on multiple segments
    a = nilsas_min(prob.quadcache)

    # compute gradient, Eq. (28) -- second part to avoid explicit construction of vbar
    (; M, nseg) = sensealg
    (; dvfs, dJs, dwfs) = prob.quadcache

    res = vec(sum(dvfs, dims = 2)) + vec(sum(dJs, dims = 2))
    NP = length(res) # number of parameters

    # loop over segments
    for (i, ai) in enumerate(eachcol(a))
        dwfsi = @view dwfs[:, i]
        dwfsi = reshape(dwfsi, NP, M)
        res .+= dwfsi * ai
    end

    return res / (nseg * prob.T_seg)
end

function check_for_g(sensealg::NILSAS, g)
    (g === nothing && error("To use NILSAS, g must be passed as a kwarg to `NILSAS(g=g)`."))
end
