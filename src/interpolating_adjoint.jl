struct ODEInterpolatingAdjointSensitivityFunction{C <: AdjointDiffCache,
    Alg <: InterpolatingAdjoint,
    uType, SType, CPS, pType,
    fType <:
    DiffEqBase.AbstractDiffEqFunction} <:
       SensitivityFunction
    diffcache::C
    sensealg::Alg
    discrete::Bool
    y::uType
    sol::SType
    checkpoint_sol::CPS
    prob::pType
    f::fType
    noiseterm::Bool
end

TruncatedStacktraces.@truncate_stacktrace ODEInterpolatingAdjointSensitivityFunction

mutable struct CheckpointSolution{S, I, T, T2}
    cpsol::S # solution in a checkpoint interval
    intervals::I # checkpoint intervals
    cursor::Int # sol.prob.tspan = intervals[cursor]
    tols::T
    tstops::T2 # for callbacks
end

function ODEInterpolatingAdjointSensitivityFunction(g, sensealg, discrete, sol, dgdu, dgdp,
        f, alg,
        checkpoints, tols, tstops = nothing;
        noiseterm = false,
        tspan = reverse(sol.prob.tspan))
    checkpointing = ischeckpointing(sensealg, sol)
    (checkpointing && checkpoints === nothing) &&
        error("checkpoints must be passed when checkpointing is enabled.")

    checkpoint_sol = if checkpointing
        intervals = map(tuple, @view(checkpoints[1:(end - 1)]), @view(checkpoints[2:end]))
        interval_end = intervals[end][end]
        tspan[1] > interval_end && push!(intervals, (interval_end, tspan[1]))
        cursor = lastindex(intervals)
        interval = intervals[cursor]

        if sol.prob isa Union{SDEProblem, RODEProblem}
            # replicated noise
            _sol = deepcopy(sol)
            idx1 = searchsortedfirst(_sol.W.t, interval[1] - 1000eps(interval[1]))
            if sol.W isa DiffEqNoiseProcess.NoiseProcess
                sol.W.save_everystep = false
                _sol.W.save_everystep = false
                forwardnoise = DiffEqNoiseProcess.NoiseWrapper(_sol.W, indx = idx1)
            elseif sol.W isa DiffEqNoiseProcess.NoiseGrid
                #idx2 = searchsortedfirst(_sol.W.t, interval[2]+1000eps(interval[1]))
                forwardnoise = DiffEqNoiseProcess.NoiseGrid(_sol.W.t[idx1:end],
                    _sol.W.W[idx1:end])
            else
                error("NoiseProcess type not implemented.")
            end
            dt = choose_dt((_sol.W.t[idx1] - _sol.W.t[idx1 + 1]), _sol.W.t, interval)

            cpsol = solve(
                remake(sol.prob, tspan = interval, u0 = sol(interval[1]),
                    noise = forwardnoise),
                sol.alg, save_noise = false; dt = dt, tstops = _sol.t[idx1:end],
                tols...)
        else
            if tstops === nothing
                cpsol = solve(remake(sol.prob, tspan = interval, u0 = sol(interval[1])),
                    sol.alg; tols...)
            else
                if maximum(interval[1] .< tstops .< interval[2])
                    # callback might have changed p
                    _p = reset_p(sol.prob.kwargs[:callback], interval)
                    cpsol = solve(
                        remake(sol.prob, tspan = interval, u0 = sol(interval[1])),
                        tstops = tstops,
                        p = _p, sol.alg; tols...)
                else
                    cpsol = solve(
                        remake(sol.prob, tspan = interval, u0 = sol(interval[1])),
                        tstops = tstops, sol.alg; tols...)
                end
            end
        end
        CheckpointSolution(cpsol, intervals, cursor, tols, tstops)
    else
        nothing
    end

    diffcache, y = adjointdiffcache(g, sensealg, discrete, sol, dgdu, dgdp, f, alg;
        quad = false, noiseterm = noiseterm)

    return ODEInterpolatingAdjointSensitivityFunction(diffcache, sensealg,
        discrete, y, sol,
        checkpoint_sol, sol.prob, f,
        noiseterm)
end

function findcursor(intervals, t)
    # equivalent with `findfirst(x->x[1] <= t <= x[2], intervals)`
    lt(x, t) = <(x[2], t)
    return searchsortedfirst(intervals, t, lt = lt)
end

function choose_dt(dt, ts, interval)
    if dt < 1000eps(interval[2])
        if length(ts) > 2
            dt = ts[end - 1] - ts[end - 2]
            if dt < 1000eps(interval[2])
                dt = interval[2] - interval[1]
            end
        else
            dt = interval[2] - interval[1]
        end
    end
    return dt
end

# u = λ'
# add tstop on all the checkpoints
function (S::ODEInterpolatingAdjointSensitivityFunction)(du, u, p, t)
    @unpack sol, checkpoint_sol, discrete, prob, f = S

    λ, grad, y, dλ, dgrad, dy = split_states(du, u, t, S)

    if S.noiseterm
        if length(u) == length(du)
            vecjacobian!(dλ, y, λ, p, t, S, dgrad = dgrad)
        elseif length(u) != length(du) && StochasticDiffEq.is_diagonal_noise(prob) &&
               !isnoisemixing(S.sensealg)
            vecjacobian!(dλ, y, λ, p, t, S)
            jacNoise!(λ, y, p, t, S, dgrad = dgrad)
        else
            jacNoise!(λ, y, p, t, S, dgrad = dgrad, dλ = dλ)
        end
    else
        vecjacobian!(dλ, y, λ, p, t, S, dgrad = dgrad)
    end

    dλ .*= -one(eltype(λ))
    dgrad .*= -one(eltype(dgrad))

    discrete || accumulate_cost!(dλ, y, p, t, S, dgrad)
    return nothing
end

function (S::ODEInterpolatingAdjointSensitivityFunction)(du, u, p, t, W)
    @unpack sol, checkpoint_sol, discrete, prob, f = S

    λ, grad, y, dλ, dgrad, dy = split_states(du, u, t, S)

    vecjacobian!(dλ, y, λ, p, t, S, dgrad = dgrad, W = W)

    dλ .*= -one(eltype(λ))
    dgrad .*= -one(eltype(dgrad))

    discrete || accumulate_cost!(dλ, y, p, t, S, dgrad)
    return nothing
end

function split_states(du, u, t, S::TS;
        update = true) where {TS <:
                              ODEInterpolatingAdjointSensitivityFunction}
    @unpack sol, y, checkpoint_sol, discrete, prob, f = S
    idx = length(y)
    if update
        if checkpoint_sol === nothing
            if t isa ForwardDiff.Dual && eltype(S.y) <: AbstractFloat
                y = sol(t, continuity = :right)
            else
                sol(y, t, continuity = :right)
            end
        else
            intervals = checkpoint_sol.intervals
            interval = intervals[checkpoint_sol.cursor]
            if !(interval[1] <= t <= interval[2])
                cursor′ = findcursor(intervals, t)
                interval = intervals[cursor′]
                cpsol_t = checkpoint_sol.cpsol.t
                if t isa ForwardDiff.Dual && eltype(S.y) <: AbstractFloat
                    y = sol(interval[1])
                else
                    sol(y, interval[1])
                end
                if sol.prob isa Union{SDEProblem, RODEProblem}
                    #idx1 = searchsortedfirst(sol.t, interval[1])
                    _sol = deepcopy(sol)
                    idx1 = searchsortedfirst(_sol.t, interval[1] - 100eps(interval[1]))
                    idx2 = searchsortedfirst(_sol.t, interval[2] + 100eps(interval[2]))
                    idx_noise = searchsortedfirst(_sol.W.t,
                        interval[1] - 100eps(interval[1]))
                    if sol.W isa DiffEqNoiseProcess.NoiseProcess
                        _sol.W.save_everystep = false
                        forwardnoise = DiffEqNoiseProcess.NoiseWrapper(_sol.W,
                            indx = idx_noise)
                    elseif sol.W isa DiffEqNoiseProcess.NoiseGrid
                        forwardnoise = DiffEqNoiseProcess.NoiseGrid(
                            _sol.W.t[idx_noise:end],
                            _sol.W.W[idx_noise:end])
                    else
                        error("NoiseProcess type not implemented.")
                    end
                    prob′ = remake(prob, tspan = intervals[cursor′], u0 = y,
                        noise = forwardnoise)
                    dt = choose_dt(abs(cpsol_t[1] - cpsol_t[2]), cpsol_t, interval)
                    cpsol′ = solve(prob′, sol.alg, save_noise = false; dt = dt,
                        tstops = _sol.t[idx1:idx2], checkpoint_sol.tols...)
                else
                    if checkpoint_sol.tstops === nothing
                        prob′ = remake(prob, tspan = intervals[cursor′], u0 = y)
                        cpsol′ = solve(prob′, sol.alg;
                            dt = abs(cpsol_t[end] - cpsol_t[end - 1]),
                            checkpoint_sol.tols...)
                    else
                        if maximum(interval[1] .< checkpoint_sol.tstops .< interval[2])
                            # callback might have changed p
                            _p = reset_p(prob.kwargs[:callback], interval)
                            prob′ = remake(prob, tspan = intervals[cursor′], u0 = y, p = _p)
                            cpsol′ = solve(prob′, sol.alg;
                                dt = abs(cpsol_t[end] - cpsol_t[end - 1]),
                                tstops = checkpoint_sol.tstops,
                                checkpoint_sol.tols...)
                        else
                            prob′ = remake(prob, tspan = intervals[cursor′], u0 = y)
                            cpsol′ = solve(prob′, sol.alg;
                                dt = abs(cpsol_t[end] - cpsol_t[end - 1]),
                                tstops = checkpoint_sol.tstops,
                                checkpoint_sol.tols...)
                        end
                    end
                end
                checkpoint_sol.cpsol = cpsol′
                checkpoint_sol.cursor = cursor′
            end
            checkpoint_sol.cpsol(y, t, continuity = :right)
        end
    end

    λ = @view u[1:idx]
    grad = @view u[(idx + 1):end]

    if length(u) == length(du)
        dλ = @view du[1:idx]
        dgrad = @view du[(idx + 1):end]

    elseif length(u) != length(du) && StochasticDiffEq.is_diagonal_noise(prob) &&
           !isnoisemixing(S.sensealg)
        idx1 = [length(u) * (i - 1) + i for i in 1:idx] # for diagonal indices of [1:idx,1:idx]

        dλ = @view du[idx1]
        dgrad = @view du[(idx + 1):end, 1:idx]

    elseif du isa AbstractMatrix
        # non-diagonal noise and noise mixing case
        m = prob.noise_rate_prototype === nothing ? idx :
            size(prob.noise_rate_prototype)[2]
        dλ = @view du[1:idx, 1:m]
        dgrad = @view du[(idx + 1):end, 1:m]
    end

    λ, grad, y, dλ, dgrad, nothing
end

# g is either g(t,u,p) or discrete g(t,u,i)
@noinline function ODEAdjointProblem(sol, sensealg::InterpolatingAdjoint, alg,
        t = nothing,
        dgdu_discrete::DG1 = nothing,
        dgdp_discrete::DG2 = nothing,
        dgdu_continuous::DG3 = nothing,
        dgdp_continuous::DG4 = nothing,
        g::G = nothing,
        ::Val{RetCB} = Val(false);
        checkpoints = sol.t,
        callback = CallbackSet(),
        reltol = nothing, abstol = nothing,
        kwargs...) where {DG1, DG2, DG3, DG4, G, RetCB}
    dgdu_discrete === nothing && dgdu_continuous === nothing && g === nothing &&
        error("Either `dgdu_discrete`, `dgdu_continuous`, or `g` must be specified.")
    t !== nothing && dgdu_discrete === nothing && dgdp_discrete === nothing &&
        error("It looks like you're using the direct `adjoint_sensitivities` interface
               with a discrete cost function but no specified `dgdu_discrete` or `dgdp_discrete`.
               Please use the higher level `solve` interface or specify these two contributions.")

    @unpack p, u0, tspan = sol.prob

    ## Force recompile mode until vjps are specialized to handle this!!!
    f = if sol.prob.f isa ODEFunction &&
           sol.prob.f.f isa FunctionWrappersWrappers.FunctionWrappersWrapper
        ODEFunction{isinplace(sol.prob), true}(unwrapped_f(sol.prob.f))
    else
        sol.prob.f
    end

    # check if solution was terminated, then use reduced time span
    terminated = false
    if hasfield(typeof(sol), :retcode)
        if sol.retcode == ReturnCode.Terminated
            tspan = (tspan[1], sol.t[end])
            terminated = true
        end
    end
    tspan = reverse(tspan)

    discrete = (t !== nothing &&
                (dgdu_continuous === nothing && dgdp_continuous === nothing ||
                 g !== nothing))

    # remove duplicates from checkpoints
    if ischeckpointing(sensealg, sol) &&
       (length(unique(checkpoints)) != length(checkpoints))
        _checkpoints, duplicate_iterator_times = separate_nonunique(checkpoints)
        tstops = duplicate_iterator_times[1]
        checkpoints = filter(x -> x ∉ tstops, _checkpoints)
        # check if start is in checkpoints. Otherwise first interval is missed.
        if checkpoints[1] != tspan[2]
            pushfirst!(checkpoints, tspan[2])
        end

        if haskey(kwargs, :tstops)
            (tstops !== kwargs[:tstops]) && unique!(push!(tstops, kwargs[:tstops]...))
        end

        # check if end is in checkpoints.
        if checkpoints[end] != tspan[1]
            push!(checkpoints, tspan[1])
        end
    else
        tstops = nothing
    end

    numstates = length(u0)
    numparams = p === nothing || p === DiffEqBase.NullParameters() ? 0 : length(p)

    len = numstates + numparams

    λ = p === nothing || p === DiffEqBase.NullParameters() ? similar(u0) :
        one(eltype(u0)) .* similar(p, len)
    λ .= false

    sense = ODEInterpolatingAdjointSensitivityFunction(g, sensealg, discrete, sol,
        dgdu_continuous, dgdp_continuous, f,
        alg, checkpoints,
        (reltol = reltol, abstol = abstol),
        tstops, tspan = tspan)

    init_cb = (discrete || dgdu_discrete !== nothing)
    cb, rcb, duplicate_iterator_times = generate_callbacks(sense, dgdu_discrete,
        dgdp_discrete,
        λ, t, tspan[2],
        callback, init_cb, terminated)
    z0 = vec(zero(λ))
    original_mm = sol.prob.f.mass_matrix
    if original_mm === I || original_mm === (I, I)
        mm = I
    else
        adjmm = copy(sol.prob.f.mass_matrix')
        zzz = similar(adjmm, numstates, numparams)
        fill!(zzz, zero(eltype(zzz)))
        # using concrate I is slightly more efficient
        II = Diagonal(I, numparams)
        mm = [adjmm zzz
              copy(zzz') II]
    end

    jac_prototype = sol.prob.f.jac_prototype
    if !sense.discrete || jac_prototype === nothing
        adjoint_jac_prototype = nothing
    else
        _adjoint_jac_prototype = copy(jac_prototype')
        zzz = similar(_adjoint_jac_prototype, numstates, numparams)
        fill!(zzz, zero(eltype(zzz)))
        II = Diagonal(I, numparams)
        adjoint_jac_prototype = [_adjoint_jac_prototype zzz
                                 copy(zzz') II]
    end

    odefun = ODEFunction{true, true}(sense, mass_matrix = mm,
        jac_prototype = adjoint_jac_prototype)
    if RetCB
        return ODEProblem(odefun, z0, tspan, p, callback = cb), rcb
    else
        return ODEProblem(odefun, z0, tspan, p, callback = cb)
    end
end

@noinline function SDEAdjointProblem(sol, sensealg::InterpolatingAdjoint, alg,
        t = nothing,
        dgdu_discrete::DG1 = nothing,
        dgdp_discrete::DG2 = nothing,
        dgdu_continuous::DG3 = nothing,
        dgdp_continuous::DG4 = nothing,
        g::G = nothing;
        checkpoints = sol.t,
        callback = CallbackSet(),
        reltol = nothing, abstol = nothing,
        diffusion_jac = nothing, diffusion_paramjac = nothing,
        kwargs...) where {DG1, DG2, DG3, DG4, G}
    dgdu_discrete === nothing && dgdu_continuous === nothing && g === nothing &&
        error("Either `dgdu_discrete`, `dgdu_continuous`, or `g` must be specified.")
    t !== nothing && dgdu_discrete === nothing && dgdp_discrete === nothing &&
        error("It looks like you're using the direct `adjoint_sensitivities` interface
               with a discrete cost function but no specified `dgdu_discrete` or `dgdp_discrete`.
               Please use the higher level `solve` interface or specify these two contributions.")

    @unpack f, p, u0, tspan = sol.prob

    # check if solution was terminated, then use reduced time span
    terminated = false
    if hasfield(typeof(sol), :retcode)
        if sol.retcode == ReturnCode.Terminated
            tspan = (tspan[1], sol.t[end])
            terminated = true
        end
    end
    tspan = reverse(tspan)

    discrete = (t !== nothing &&
                (dgdu_continuous === nothing && dgdp_continuous === nothing ||
                 g !== nothing))

    # remove duplicates from checkpoints
    if ischeckpointing(sensealg, sol) &&
       (length(unique(checkpoints)) != length(checkpoints))
        _checkpoints, duplicate_iterator_times = separate_nonunique(checkpoints)
        tstops = duplicate_iterator_times[1]
        checkpoints = filter(x -> x ∉ tstops, _checkpoints)
        # check if start is in checkpoints. Otherwise first interval is missed.
        if checkpoints[1] != tspan[2]
            pushfirst!(checkpoints, tspan[2])
        end

        if haskey(kwargs, :tstops)
            (tstops !== kwargs[:tstops]) && unique!(push!(tstops, kwargs[:tstops]...))
        end

        # check if end is in checkpoints.
        if checkpoints[end] != tspan[1]
            push!(checkpoints, tspan[1])
        end
    else
        tstops = nothing
    end

    numstates = length(u0)
    numparams = p === nothing || p === DiffEqBase.NullParameters() ? 0 : length(p)

    len = numstates + numparams

    λ = one(eltype(u0)) .* similar(p, len)
    λ .= false

    sense_drift = ODEInterpolatingAdjointSensitivityFunction(g, sensealg, discrete, sol,
        dgdu_continuous,
        dgdp_continuous, sol.prob.f,
        alg, checkpoints,
        (reltol = reltol,
            abstol = abstol),
        tspan = tspan)

    diffusion_function = ODEFunction{isinplace(sol.prob), true}(sol.prob.g,
        jac = diffusion_jac,
        paramjac = diffusion_paramjac)
    sense_diffusion = ODEInterpolatingAdjointSensitivityFunction(
        g, sensealg, discrete, sol,
        dgdu_continuous,
        dgdp_continuous,
        diffusion_function,
        alg, checkpoints,
        (reltol = reltol,
            abstol = abstol);
        tspan = tspan,
        noiseterm = true)

    init_cb = (discrete || dgdu_discrete !== nothing) # && tspan[1] == t[end]
    cb, _, duplicate_iterator_times = generate_callbacks(sense_drift, dgdu_discrete,
        dgdp_discrete, λ, t,
        tspan[2], callback, init_cb,
        terminated)
    z0 = vec(zero(λ))
    original_mm = sol.prob.f.mass_matrix
    if original_mm === I || original_mm === (I, I)
        mm = I
    else
        adjmm = copy(sol.prob.f.mass_matrix')
        zzz = similar(adjmm, numstates, numparams)
        fill!(zzz, zero(eltype(zzz)))
        # using concrate I is slightly more efficient
        II = Diagonal(I, numparams)
        mm = [adjmm zzz
              copy(zzz') II]
    end

    jac_prototype = sol.prob.f.jac_prototype
    if !sense_drift.discrete || jac_prototype === nothing
        adjoint_jac_prototype = nothing
    else
        _adjoint_jac_prototype = copy(jac_prototype')
        zzz = similar(_adjoint_jac_prototype, numstates, numparams)
        fill!(zzz, zero(eltype(zzz)))
        II = Diagonal(I, numparams)
        adjoint_jac_prototype = [_adjoint_jac_prototype zzz
                                 copy(zzz') II]
    end

    sdefun = SDEFunction(sense_drift, sense_diffusion, mass_matrix = mm,
        jac_prototype = adjoint_jac_prototype)

    # replicated noise
    _sol = deepcopy(sol)
    backwardnoise = reverse(_sol.W)

    if StochasticDiffEq.is_diagonal_noise(sol.prob) && sol.W.u[end] isa Number
        # scalar noise case
        noise_matrix = nothing
    else
        m = sol.prob.noise_rate_prototype === nothing ? numstates :
            size(sol.prob.noise_rate_prototype)[2]
        noise_matrix = similar(z0, length(z0), m)
        noise_matrix .= false
    end

    return SDEProblem(sdefun, sense_diffusion, z0, tspan, p,
        callback = cb,
        noise = backwardnoise,
        noise_rate_prototype = noise_matrix)
end

@noinline function RODEAdjointProblem(sol, sensealg::InterpolatingAdjoint, alg,
        t = nothing,
        dgdu_discrete::DG1 = nothing,
        dgdp_discrete::DG2 = nothing,
        dgdu_continuous::DG3 = nothing,
        dgdp_continuous::DG4 = nothing,
        g::G = nothing;
        checkpoints = sol.t,
        callback = CallbackSet(),
        reltol = nothing, abstol = nothing,
        kwargs...) where {DG1, DG2, DG3, DG4, G}
    dgdu_discrete === nothing && dgdu_continuous === nothing && g === nothing &&
        error("Either `dgdu_discrete`, `dgdu_continuous`, or `g` must be specified.")
    t !== nothing && dgdu_discrete === nothing && dgdp_discrete === nothing &&
        error("It looks like you're using the direct `adjoint_sensitivities` interface
               with a discrete cost function but no specified `dgdu_discrete` or `dgdp_discrete`.
               Please use the higher level `solve` interface or specify these two contributions.")

    @unpack f, p, u0, tspan = sol.prob

    # check if solution was terminated, then use reduced time span
    terminated = false
    if hasfield(typeof(sol), :retcode)
        if sol.retcode == ReturnCode.Terminated
            tspan = (tspan[1], sol.t[end])
            terminated = true
        end
    end
    tspan = reverse(tspan)

    discrete = (t !== nothing &&
                (dgdu_continuous === nothing && dgdp_continuous === nothing ||
                 g !== nothing))

    # remove duplicates from checkpoints
    if ischeckpointing(sensealg, sol) &&
       (length(unique(checkpoints)) != length(checkpoints))
        _checkpoints, duplicate_iterator_times = separate_nonunique(checkpoints)
        tstops = duplicate_iterator_times[1]
        checkpoints = filter(x -> x ∉ tstops, _checkpoints)
        # check if start is in checkpoints. Otherwise first interval is missed.
        if checkpoints[1] != tspan[2]
            pushfirst!(checkpoints, tspan[2])
        end

        if haskey(kwargs, :tstops)
            (tstops !== kwargs[:tstops]) && unique!(push!(tstops, kwargs[:tstops]...))
        end

        # check if end is in checkpoints.
        if checkpoints[end] != tspan[1]
            push!(checkpoints, tspan[1])
        end
    else
        tstops = nothing
    end

    numstates = length(u0)
    numparams = p === nothing || p === DiffEqBase.NullParameters() ? 0 : length(p)

    len = numstates + numparams

    λ = p === nothing || p === DiffEqBase.NullParameters() ? similar(u0) :
        one(eltype(u0)) .* similar(p, len)
    λ .= false

    sense = ODEInterpolatingAdjointSensitivityFunction(g, sensealg, discrete, sol,
        dgdu_continuous, dgdp_continuous, f,
        alg, checkpoints,
        (reltol = reltol, abstol = abstol),
        tstops, tspan = tspan)

    init_cb = (discrete || dgdu_discrete !== nothing) # && tspan[1] == t[end]
    cb, _, duplicate_iterator_times = generate_callbacks(sense, dgdu_discrete,
        dgdp_discrete,
        λ, t, tspan[2],
        callback, init_cb, terminated)
    z0 = vec(zero(λ))
    original_mm = sol.prob.f.mass_matrix
    if original_mm === I || original_mm === (I, I)
        mm = I
    else
        adjmm = copy(sol.prob.f.mass_matrix')
        zzz = similar(adjmm, numstates, numparams)
        fill!(zzz, zero(eltype(zzz)))
        # using concrate I is slightly more efficient
        II = Diagonal(I, numparams)
        mm = [adjmm zzz
              copy(zzz') II]
    end

    jac_prototype = sol.prob.f.jac_prototype
    if !sense.discrete || jac_prototype === nothing
        adjoint_jac_prototype = nothing
    else
        _adjoint_jac_prototype = copy(jac_prototype')
        zzz = similar(_adjoint_jac_prototype, numstates, numparams)
        fill!(zzz, zero(eltype(zzz)))
        II = Diagonal(I, numparams)
        adjoint_jac_prototype = [_adjoint_jac_prototype zzz
                                 copy(zzz') II]
    end

    rodefun = RODEFunction(sense, mass_matrix = mm, jac_prototype = adjoint_jac_prototype)

    # replicated noise
    _sol = deepcopy(sol)
    backwardnoise = reverse(_sol.W)
    # make sure noise grid starts at correct time values, e.g., if sol.W.t is longer than sol.t
    tspan[1] != backwardnoise.t[1] &&
        reinit!(backwardnoise, backwardnoise.t[2] - backwardnoise.t[1], t0 = tspan[1])

    return RODEProblem(rodefun, z0, tspan, p, callback = cb,
        noise = backwardnoise)
end

function reset_p(CBS, interval)
    # check which events are close to tspan[1]
    if !isempty(CBS.discrete_callbacks)
        ts = map(CBS.discrete_callbacks) do cb
            indx = searchsortedfirst(cb.affect!.event_times, interval[1])
            (indx, cb.affect!.event_times[indx])
        end
        perm = minimum(sortperm([t for t in getindex.(ts, 2)]))
    end

    if !isempty(CBS.continuous_callbacks)
        ts2 = map(CBS.continuous_callbacks) do cb
            if !isempty(cb.affect!.event_times) && isempty(cb.affect_neg!.event_times)
                indx = searchsortedfirst(cb.affect!.event_times, interval[1])
                return (indx, cb.affect!.event_times[indx], 0) # zero for affect!
            elseif isempty(cb.affect!.event_times) && !isempty(cb.affect_neg!.event_times)
                indx = searchsortedfirst(cb.affect_neg!.event_times, interval[1])
                return (indx, cb.affect_neg!.event_times[indx], 1) # one for affect_neg!
            elseif !isempty(cb.affect!.event_times) && !isempty(cb.affect_neg!.event_times)
                indx1 = searchsortedfirst(cb.affect!.event_times, interval[1])
                indx2 = searchsortedfirst(cb.affect_neg!.event_times, interval[1])
                if cb.affect!.event_times[indx1] < cb.affect_neg!.event_times[indx2]
                    return (indx1, cb.affect!.event_times[indx1], 0)
                else
                    return (indx2, cb.affect_neg!.event_times[indx2], 1)
                end
            else
                error("Expected event but reset_p couldn't find event time. Please report this error.")
            end
        end
        perm2 = minimum(sortperm([t for t in getindex.(ts2, 2)]))
        # check if continuous or discrete callback was applied first if both occur in interval
        if isempty(CBS.discrete_callbacks)
            if ts2[perm2][3] == 0
                p = deepcopy(CBS.continuous_callbacks[perm2].affect!.pleft[getindex.(ts2, 1)[perm2]])
            else
                p = deepcopy(CBS.continuous_callbacks[perm2].affect_neg!.pleft[getindex.(
                    ts2,
                    1)[perm2]])
            end
        else
            if ts[perm][2] < ts2[perm2][2]
                p = deepcopy(CBS.discrete_callbacks[perm].affect!.pleft[getindex.(ts, 1)[perm]])
            else
                if ts2[perm2][3] == 0
                    p = deepcopy(CBS.continuous_callbacks[perm2].affect!.pleft[getindex.(
                        ts2,
                        1)[perm2]])
                else
                    p = deepcopy(CBS.continuous_callbacks[perm2].affect_neg!.pleft[getindex.(
                        ts2,
                        1)[perm2]])
                end
            end
        end
    else
        p = deepcopy(CBS.discrete_callbacks[perm].affect!.pleft[getindex.(ts, 1)[perm]])
    end

    return p
end
