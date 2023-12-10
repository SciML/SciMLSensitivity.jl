using SciMLSensitivity, Lux, ComponentArrays, LinearAlgebra, DiffEqNoiseProcess, Test
using StochasticDiffEq, Statistics, SciMLSensitivity, Zygote
using DiffEqBase.EnsembleAnalysis
using Optimization, OptimizationOptimisers

using Random
Random.seed!(238248735)

@testset "Neural SDE" begin
    function sys!(du, u, p, t)
        r, e, μ, h, ph, z, i = p
        du[1] = e * 0.5 * (5μ - u[1]) # nutrient input time series
        du[2] = e * 0.05 * (10μ - u[2]) # grazer density time series
        du[3] = 0.2 * exp(u[1]) - 0.05 * u[3] - r * u[3] / (h + u[3]) * u[4] # nutrient concentration
        du[4] = r * u[3] / (h + u[3]) * u[4] - 0.1 * u[4] -
                0.02 * u[4]^z / (ph^z + u[4]^z) * exp(u[2] / 2.0) + i #Algae density
    end

    function noise!(du, u, p, t)
        du[1] = p[end] # n
        du[2] = p[end] # n
        du[3] = 0.0
        du[4] = 0.0
    end

    datasize = 10
    tspan = (0.0f0, 3.0f0)
    tsteps = range(tspan[1], tspan[2], length = datasize)
    u0 = [1.0, 1.0, 1.0, 1.0]
    p_ = [1.1, 1.0, 0.0, 2.0, 1.0, 1.0, 1e-6, 1.0]

    prob = SDEProblem(sys!, noise!, u0, tspan, p_)
    ensembleprob = EnsembleProblem(prob)

    solution = solve(ensembleprob,
        SOSRI(),
        EnsembleThreads();
        trajectories = 1000,
        abstol = 1e-5,
        reltol = 1e-5,
        maxiters = 1e8,
        saveat = tsteps)

    (truemean, truevar) = Array.(timeseries_steps_meanvar(solution))

    ann = Chain(Dense(4, 32, tanh), Dense(32, 32, tanh), Dense(32, 2))
    α, st = Lux.setup(Random.default_rng(), ann)
    α = ComponentArray(α)
    α = Float64.(α)

    function dudt_(du, u, p, t)
        r, e, μ, h, ph, z, i = p_

        MM = first(ann(u, p, st))

        du[1] = e * 0.5 * (5μ - u[1]) # nutrient input time series
        du[2] = e * 0.05 * (10μ - u[2]) # grazer density time series
        du[3] = 0.2 * exp(u[1]) - 0.05 * u[3] - MM[1] # nutrient concentration
        du[4] = MM[2] - 0.1 * u[4] - 0.02 * u[4]^z / (ph^z + u[4]^z) * exp(u[2] / 2.0) + i #Algae density
        return nothing
    end

    function dudt_op(u, p, t)
        r, e, μ, h, ph, z, i = p_

        MM = first(ann(u, p, st))

        [e * 0.5 * (5μ - u[1]), # nutrient input time series
            e * 0.05 * (10μ - u[2]), # grazer density time series
            0.2 * exp(u[1]) - 0.05 * u[3] - MM[1], # nutrient concentration
            MM[2] - 0.1 * u[4] - 0.02 * u[4]^z / (ph^z + u[4]^z) * exp(u[2] / 2.0) + i] #Algae density
    end

    function noise_(du, u, p, t)
        du[1] = p_[end]
        du[2] = p_[end]
        du[3] = 0.0
        du[4] = 0.0
        return nothing
    end

    function noise_op(u, p, t)
        [p_[end],
            p_[end],
            0.0,
            0.0]
    end

    prob_nn = SDEProblem(dudt_, noise_, u0, tspan, p = nothing)
    prob_nn_op = SDEProblem(dudt_op, noise_op, u0, tspan, p = nothing)

    function loss(θ)
        tmp_prob = remake(prob_nn, p = θ)
        ensembleprob = EnsembleProblem(tmp_prob)
        tmp_sol = Array(solve(ensembleprob,
            EM();
            dt = tsteps.step,
            trajectories = 100,
            sensealg = ReverseDiffAdjoint()))
        tmp_mean = mean(tmp_sol, dims = 3)[:, :]
        tmp_var = var(tmp_sol, dims = 3)[:, :]
        sum(abs2, truemean - tmp_mean) + 0.1 * sum(abs2, truevar - tmp_var), tmp_mean
    end

    function loss_op(θ)
        tmp_prob = remake(prob_nn_op, p = θ)
        ensembleprob = EnsembleProblem(tmp_prob)
        tmp_sol = Array(solve(ensembleprob,
            EM();
            dt = tsteps.step,
            trajectories = 100,
            sensealg = ReverseDiffAdjoint()))
        tmp_mean = mean(tmp_sol, dims = 3)[:, :]
        tmp_var = var(tmp_sol, dims = 3)[:, :]
        sum(abs2, truemean - tmp_mean) + 0.1 * sum(abs2, truevar - tmp_var), tmp_mean
    end

    losses = []
    function callback(θ, l, pred)
        begin
            push!(losses, l)
            if length(losses) % 50 == 0
                println("Current loss after $(length(losses)) iterations: $(losses[end])")
            end
            false
        end
    end
    println("Test mutating form")

    optf = Optimization.OptimizationFunction((x, p) -> loss(x), Optimization.AutoZygote())
    optprob = Optimization.OptimizationProblem(optf, α)
    res1 = Optimization.solve(optprob, Adam(0.001), callback = callback, maxiters = 200)

    println("Test non-mutating form")

    optf = Optimization.OptimizationFunction((x, p) -> loss_op(x),
        Optimization.AutoZygote())
    optprob = Optimization.OptimizationProblem(optf, α)
    res2 = Optimization.solve(optprob, Adam(0.001), callback = callback, maxiters = 200)
end

@testset "Adaptive neural SDE" begin
    x_size = 2 # Size of the spatial dimensions in the SDE
    v_size = 2 # Output size of the control

    # Define Neural Network for the control input
    input_size = x_size + 1 # size of the spatial dimensions PLUS one time dimensions
    nn_initial = Chain(Dense(input_size, v_size)) # The actual neural network
    ps, st = Lux.setup(Random.default_rng(), nn_initial)
    ps = ComponentArray(ps)
    nn(x, p) = first(nn_initial(x, p, st)) # The neural network function

    # Define the right hand side of the SDE
    const_mat = zeros(Float64, (x_size, v_size))
    for i in 1:max(x_size, v_size)
        const_mat[i, i] = 1
    end

    function f!(du, u, p, t)
        MM = nn([u; t], p)
        du .= u + const_mat * MM
    end

    function g!(du, u, p, t)
        du .= false * u .+ sqrt(2 * 0.001)
    end

    # Define SDE problem
    u0 = vec(rand(Float64, (x_size, 1)))
    tspan = (0.0, 1.0)
    ts = collect(0:0.1:1)
    prob = SDEProblem{true}(f!, g!, u0, tspan, ps)

    W = WienerProcess(0.0, 0.0, 0.0)
    probscalar = SDEProblem{true}(f!, g!, u0, tspan, ps, noise = W)

    # Defining the loss function
    function loss(pars, prob, alg)
        function prob_func(prob, i, repeat)
            # Prepare new initial state and remake the problem
            u0tmp = vec(rand(Float64, (x_size, 1)))

            remake(prob, p = pars, u0 = u0tmp)
        end

        ensembleprob = EnsembleProblem(prob, prob_func = prob_func)

        _sol = solve(ensembleprob, alg, EnsembleThreads(),
            sensealg = BacksolveAdjoint(autojacvec = ReverseDiffVJP()),
            saveat = ts, trajectories = 10, abstol = 1e-1, reltol = 1e-1)
        A = convert(Array, _sol)
        sum(abs2, A .- 1), mean(A)
    end

    # Actually training/fitting the model
    losses = []
    function callback(θ, l, pred)
        begin
            push!(losses, l)
            if length(losses) % 1 == 0
                println("Current loss after $(length(losses)) iterations: $(losses[end])")
            end
            false
        end
    end

    optf = Optimization.OptimizationFunction((p, _) -> loss(p, probscalar, LambaEM()),
        Optimization.AutoZygote())
    optprob = Optimization.OptimizationProblem(optf, ps)
    res1 = Optimization.solve(optprob, Adam(0.1), callback = callback, maxiters = 5)

    optf = Optimization.OptimizationFunction((p, _) -> loss(p, probscalar, SOSRI()),
        Optimization.AutoZygote())
    optprob = Optimization.OptimizationProblem(optf, ps)
    res2 = Optimization.solve(optprob, Adam(0.1), callback = callback, maxiters = 5)

    optf = Optimization.OptimizationFunction((p, _) -> loss(p, prob, LambaEM()),
        Optimization.AutoZygote())
    optprob = Optimization.OptimizationProblem(optf, ps)
    res1 = Optimization.solve(optprob, Adam(0.1), callback = callback, maxiters = 5)
end
