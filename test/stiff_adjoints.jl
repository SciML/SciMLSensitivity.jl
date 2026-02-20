using Zygote, SciMLSensitivity
println("Starting tests")
using OrdinaryDiffEq, ForwardDiff, Test, Reactant

function lotka_volterra(u, p, t)
    x, y = u
    α, β, δ, γ = p
    return [α * x - β * x * y, -δ * y + γ * x * y]
end

function lotka_volterra(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = dx = α * x - β * x * y
    du[2] = dy = -δ * y + γ * x * y
    return nothing
end

u0 = [1.0, 1.0];
tspan = (0.0, 10.0);
p0 = [1.5, 1.0, 3.0, 1.0];
prob0 = ODEProblem(lotka_volterra, u0, tspan, p0);
# Solve the ODE and collect solutions at fixed intervals
target_data = solve(prob0, RadauIIA5(), saveat = 0:0.5:10.0);

loss_function = function (p)
    prob = remake(prob0; u0 = convert.(eltype(p), prob0.u0), p)
    prediction = solve(
        prob, RadauIIA5(); saveat = 0.0:0.5:10.0, abstol = 1.0e-10,
        reltol = 1.0e-10
    )

    tmpdata = prediction[[1, 2], :]
    tdata = target_data[[1, 2], :]

    # Calculate squared error
    return sum(abs2, tmpdata - tdata)
end
p = [1.5, 1.2, 1.4, 1.6];
fdgrad = ForwardDiff.gradient(loss_function, p)
rdgrad = Zygote.gradient(loss_function, p)[1]

@test fdgrad ≈ rdgrad rtol = 1.0e-5

loss_function = function (p)
    prob = remake(prob0; u0 = convert.(eltype(p), prob0.u0), p)
    prediction = solve(
        prob, TRBDF2(); saveat = 0.0:0.5:10.0, abstol = 1.0e-10,
        reltol = 1.0e-10
    )

    tmpdata = prediction[[1, 2], :]
    tdata = target_data[[1, 2], :]

    # Calculate squared error
    return sum(abs2, tmpdata - tdata)
end

rdgrad = Zygote.gradient(loss_function, p)[1]
@test fdgrad ≈ rdgrad rtol = 1.0e-3

loss_function = function (p)
    prob = remake(prob0; u0 = convert.(eltype(p), prob0.u0), p)
    prediction = solve(
        prob, Rosenbrock23(); saveat = 0.0:0.5:10.0, abstol = 1.0e-8,
        reltol = 1.0e-8
    )

    tmpdata = prediction[[1, 2], :]
    tdata = target_data[[1, 2], :]

    # Calculate squared error
    return sum(abs2, tmpdata - tdata)
end

rdgrad = Zygote.gradient(loss_function, p)[1]
@test fdgrad ≈ rdgrad rtol = 1.0e-3

loss_function = function (p)
    prob = remake(prob0; u0 = convert.(eltype(p), prob0.u0), p)
    prediction = solve(prob, Rodas5(); saveat = 0.0:0.5:10.0, abstol = 1.0e-8, reltol = 1.0e-8)

    tmpdata = prediction[[1, 2], :]
    tdata = target_data[[1, 2], :]

    # Calculate squared error
    return sum(abs2, tmpdata - tdata)
end

rdgrad = Zygote.gradient(loss_function, p)[1]
@test fdgrad ≈ rdgrad rtol = 1.0e-3

### OOP

prob0_oop = ODEProblem{false}(lotka_volterra, u0, tspan, p0);
# Solve the ODE and collect solutions at fixed intervals
target_data = solve(prob0, RadauIIA5(), saveat = 0:0.5:10.0);

loss_function = function (p)
    prob = remake(prob0_oop; u0 = convert.(eltype(p), prob0.u0), p)
    prediction = solve(
        prob, RadauIIA5(); saveat = 0.0:0.5:10.0, abstol = 1.0e-10,
        reltol = 1.0e-10
    )

    tmpdata = prediction[[1, 2], :]
    tdata = target_data[[1, 2], :]

    # Calculate squared error
    return sum(abs2, tmpdata - tdata)
end
p = [1.5, 1.2, 1.4, 1.6];

fdgrad = ForwardDiff.gradient(loss_function, p)
rdgrad = Zygote.gradient(loss_function, p)[1]

@test fdgrad ≈ rdgrad rtol = 1.0e-4

loss_function = function (p)
    prob = remake(prob0_oop; u0 = convert.(eltype(p), prob0.u0), p)
    prediction = solve(
        prob, TRBDF2(); saveat = 0.0:0.5:10.0, abstol = 1.0e-10,
        reltol = 1.0e-10
    )

    tmpdata = prediction[[1, 2], :]
    tdata = target_data[[1, 2], :]

    # Calculate squared error
    return sum(abs2, tmpdata - tdata)
end

rdgrad = Zygote.gradient(loss_function, p)[1]
@test fdgrad ≈ rdgrad rtol = 1.0e-3

loss_function = function (p)
    prob = remake(prob0_oop; u0 = convert.(eltype(p), prob0.u0), p)
    prediction = solve(
        prob, Rosenbrock23(); saveat = 0.0:0.5:10.0, abstol = 1.0e-8,
        reltol = 1.0e-8
    )

    tmpdata = prediction[[1, 2], :]
    tdata = target_data[[1, 2], :]

    # Calculate squared error
    return sum(abs2, tmpdata - tdata)
end

rdgrad = Zygote.gradient(loss_function, p)[1]
@test fdgrad ≈ rdgrad rtol = 1.0e-4

loss_function = function (p)
    prob = remake(prob0_oop; u0 = convert.(eltype(p), prob0.u0), p)
    prediction = solve(
        prob, Rodas5(); saveat = 0.0:0.5:10.0, abstol = 1.0e-12,
        reltol = 1.0e-12
    )

    tmpdata = prediction[[1, 2], :]
    tdata = target_data[[1, 2], :]

    # Calculate squared error
    return sum(abs2, tmpdata - tdata)
end

rdgrad = Zygote.gradient(loss_function, p)[1]
@test fdgrad ≈ rdgrad rtol = 1.0e-3

if VERSION >= v"1.7-"
    # all implicit solvers
    solvers = [
        # SDIRK Methods (ok)
        ImplicitEuler(),
        TRBDF2(),
        KenCarp4(),
        # Fully-Implicit Runge-Kutta Methods (FIRK)
        RadauIIA5(),
        # Fully-Implicit Runge-Kutta Methods (FIRK)
        #PDIRK44(),
        # Rosenbrock Methods
        Rodas3(),
        Rodas4(),
        Rodas5(),
        # Rosenbrock-W Methods
        Rosenbrock23(),
        ROS34PW3(),
        # Stabilized Explicit Methods (ok)
        ROCK2(),
        ROCK4(),
    ]

    p = rand(3)

    function dudt(u, p, t)
        return u .* p
    end

    for solver in solvers
        function loss(p)
            prob = ODEProblem(dudt, [3.0, 2.0, 1.0], (0.0, 1.0), p)
            sol = solve(prob, solver, dt = 0.01, saveat = 0.1, abstol = 1.0e-5, reltol = 1.0e-5)
            return sum(abs2, Array(sol))
        end

        loss(p)
        dp = Zygote.gradient(loss, p)[1]

        function loss(p, sensealg)
            prob = ODEProblem(dudt, [3.0, 2.0, 1.0], (0.0, 1.0), p)
            sol = solve(
                prob, solver; dt = 0.01, saveat = 0.1, sensealg,
                abstol = 1.0e-5, reltol = 1.0e-5
            )
            return sum(abs2, Array(sol))
        end

        dp1 = Zygote.gradient(p -> loss(p, InterpolatingAdjoint()), p)[1]
        @test dp ≈ dp1 rtol = 1.0e-2
        dp1 = Zygote.gradient(p -> loss(p, BacksolveAdjoint()), p)[1]
        @test dp ≈ dp1 rtol = 1.0e-2
        dp1 = Zygote.gradient(p -> loss(p, QuadratureAdjoint()), p)[1]
        @test dp ≈ dp1 rtol = 1.0e-2
        dp1 = Zygote.gradient(p -> loss(p, ForwardDiffSensitivity()), p)[1]
        @test dp ≈ dp1 rtol = 1.0e-2
        if SciMLBase.forwarddiffs_model(solver)
            @test Zygote.gradient(
                p -> loss(p, QuadratureAdjoint(autojacvec = EnzymeVJP())), p
            )[1] isa Vector
            @test Zygote.gradient(
                p -> loss(p, QuadratureAdjoint(autojacvec = ReactantVJP())), p
            )[1] isa Vector
            @test_broken Zygote.gradient(p -> loss(p, ReverseDiffAdjoint()), p)[1] isa
                Vector
        else
            dp1 = Zygote.gradient(
                p -> loss(p, QuadratureAdjoint(autojacvec = EnzymeVJP())), p
            )[1]
            @test dp ≈ dp1 rtol = 1.0e-2
            dp1 = Zygote.gradient(
                p -> loss(p, QuadratureAdjoint(autojacvec = ReactantVJP())), p
            )[1]
            @test dp ≈ dp1 rtol = 1.0e-2
            dp1 = Zygote.gradient(p -> loss(p, ReverseDiffAdjoint()), p)[1]
            @test dp ≈ dp1 rtol = 1.0e-2
        end
    end

    # using SciMLSensitivity, OrdinaryDiffEq, ForwardDiff, Zygote, Test

    function rober(du, u, p, t)
        y₁, y₂, y₃ = u
        k₁, k₂, k₃ = p[1], p[2], p[3]
        du[1] = -k₁ * y₁ + k₃ * y₂ * y₃
        du[2] = k₁ * y₁ - k₂ * y₂^2 - k₃ * y₂ * y₃
        du[3] = k₂ * y₂^2 + sum(p)
        return nothing
    end

    function sum_of_solution_fwd(x)
        _prob = ODEProblem(rober, x[1:3], (0.0, 1.0e4), x[4:end])
        return sum(solve(_prob, Rodas5(), saveat = 1, reltol = 1.0e-12, abstol = 1.0e-12))
    end

    function sum_of_solution_CASA(x; vjp = EnzymeVJP())
        sensealg = QuadratureAdjoint(autodiff = false, autojacvec = vjp)
        _prob = ODEProblem(rober, x[1:3], (0.0, 1.0e4), x[4:end])
        return sum(
            solve(
                _prob, Rodas5P(); reltol = 1.0e-12, abstol = 1.0e-12, saveat = 1,
                sensealg
            )
        )
    end

    u0 = [1.0, 0.0, 0.0]
    p = ones(8)  # change me, the number of parameters

    println("grad1")
    grad1 = ForwardDiff.gradient(sum_of_solution_fwd, [u0; p])
    println("grad2")
    grad2 = Zygote.gradient(sum_of_solution_CASA, [u0; p])[1]
    println("grad3")
    grad3 = Zygote.gradient(x -> sum_of_solution_CASA(x, vjp = ReverseDiffVJP()), [u0; p])[1]
    println("grad4")
    grad4 = Zygote.gradient(
        x -> sum_of_solution_CASA(x, vjp = ReverseDiffVJP(true)),
        [u0; p]
    )[1]
    # Is too numerically dependent
    #println("grad5")
    #@test_broken Zygote.gradient(x -> sum_of_solution_CASA(x, vjp = true), [u0; p])[1] isa Array
    # Takes too long
    #println("grad6")
    #grad6 = Zygote.gradient(x -> sum_of_solution_CASA(x, vjp = false), [u0; p])[1]
    println("grad7")
    grad7 = Zygote.gradient(
        x -> sum_of_solution_CASA(x, vjp = ZygoteVJP()),
        [u0; p]
    )[1]
    println("grad8")
    @test_throws Any Zygote.gradient(
        x -> sum_of_solution_CASA(x, vjp = TrackerVJP()),
        [u0; p]
    )[1]
    println("grad9")
    grad9 = Zygote.gradient(
        x -> sum_of_solution_CASA(x, vjp = ReactantVJP()),
        [u0; p]
    )[1]

    @test grad1 ≈ grad2
    @test grad1 ≈ grad3
    @test grad1 ≈ grad4
    #@test grad1 ≈ grad5
    #@test grad1 ≈ grad6
    @test grad1 ≈ grad7 rtol = 1.0e-2
    @test grad1 ≈ grad9
end
