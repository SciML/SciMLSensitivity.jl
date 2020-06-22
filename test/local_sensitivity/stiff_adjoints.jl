using Zygote, DiffEqSensitivity
println("Starting tests")
using OrdinaryDiffEq, ForwardDiff, Test

function lotka_volterra(u, p, t)
  x, y = u
  α, β, δ, γ = p
  [α * x - β * x * y,-δ * y + γ * x * y]
end

function lotka_volterra(du, u, p, t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = α * x - β * x * y
  du[2] = dy = -δ * y + γ * x * y
end

u0 = [1.0,1.0];
tspan = (0.0,10.0);
p0 = [1.5,1.0,3.0,1.0];
prob0 = ODEProblem(lotka_volterra,u0,tspan,p0);
# Solve the ODE and collect solutions at fixed intervals
target_data = solve(prob0,RadauIIA5(), saveat =  0:0.5:10.0);

loss_function = function(p)
    prob = remake(prob0;u0=convert.(eltype(p),prob0.u0),p=p)
    prediction = solve(prob, RadauIIA5(); saveat = 0.0:0.5:10.0,abstol=1e-10,reltol=1e-10)

    tmpdata=prediction[[1,2],:];
    tdata=target_data[[1,2],:];

    # Calculate squared error
    return sum(abs2,tmpdata-tdata)
end
p=[1.5,1.2,1.4,1.6];
fdgrad = ForwardDiff.gradient(loss_function,p)
rdgrad = Zygote.gradient(loss_function,p)[1]

@test fdgrad ≈ rdgrad rtol=1e-5

loss_function = function(p)
    prob = remake(prob0;u0=convert.(eltype(p),prob0.u0),p=p)
    prediction = solve(prob, TRBDF2(); saveat = 0.0:0.5:10.0,abstol=1e-10,reltol=1e-10)

    tmpdata=prediction[[1,2],:];
    tdata=target_data[[1,2],:];

    # Calculate squared error
    return sum(abs2,tmpdata-tdata)
end

rdgrad = Zygote.gradient(loss_function,p)[1]
@test fdgrad ≈ rdgrad rtol=1e-3

loss_function = function(p)
    prob = remake(prob0;u0=convert.(eltype(p),prob0.u0),p=p)
    prediction = solve(prob, Rosenbrock23(); saveat = 0.0:0.5:10.0,abstol=1e-8,reltol=1e-8)

    tmpdata=prediction[[1,2],:];
    tdata=target_data[[1,2],:];

    # Calculate squared error
    return sum(abs2,tmpdata-tdata)
end

rdgrad = Zygote.gradient(loss_function,p)[1]
@test fdgrad ≈ rdgrad rtol=1e-3

loss_function = function(p)
    prob = remake(prob0;u0=convert.(eltype(p),prob0.u0),p=p)
    prediction = solve(prob, Rodas5(); saveat = 0.0:0.5:10.0,abstol=1e-8,reltol=1e-8)

    tmpdata=prediction[[1,2],:];
    tdata=target_data[[1,2],:];

    # Calculate squared error
    return sum(abs2,tmpdata-tdata)
end

rdgrad = Zygote.gradient(loss_function,p)[1]
@test fdgrad ≈ rdgrad rtol=1e-3

### OOP

prob0_oop = ODEProblem{false}(lotka_volterra,u0,tspan,p0);
# Solve the ODE and collect solutions at fixed intervals
target_data = solve(prob0,RadauIIA5(), saveat =  0:0.5:10.0);

loss_function = function(p)
    prob = remake(prob0_oop;u0=convert.(eltype(p),prob0.u0),p=p)
    prediction = solve(prob, RadauIIA5(); saveat = 0.0:0.5:10.0,abstol=1e-10,reltol=1e-10)

    tmpdata=prediction[[1,2],:];
    tdata=target_data[[1,2],:];

    # Calculate squared error
    return sum(abs2,tmpdata-tdata)
end
p=[1.5,1.2,1.4,1.6];

fdgrad = ForwardDiff.gradient(loss_function,p)
rdgrad = Zygote.gradient(loss_function,p)[1]

@test fdgrad ≈ rdgrad rtol=1e-4

loss_function = function(p)
    prob = remake(prob0_oop;u0=convert.(eltype(p),prob0.u0),p=p)
    prediction = solve(prob, TRBDF2(); saveat = 0.0:0.5:10.0,abstol=1e-10,reltol=1e-10)

    tmpdata=prediction[[1,2],:];
    tdata=target_data[[1,2],:];

    # Calculate squared error
    return sum(abs2,tmpdata-tdata)
end

rdgrad = Zygote.gradient(loss_function,p)[1]
@test fdgrad ≈ rdgrad rtol=1e-3

loss_function = function(p)
    prob = remake(prob0_oop;u0=convert.(eltype(p),prob0.u0),p=p)
    prediction = solve(prob, Rosenbrock23(); saveat = 0.0:0.5:10.0,abstol=1e-8,reltol=1e-8)

    tmpdata=prediction[[1,2],:];
    tdata=target_data[[1,2],:];

    # Calculate squared error
    return sum(abs2,tmpdata-tdata)
end

rdgrad = Zygote.gradient(loss_function,p)[1]
@test fdgrad ≈ rdgrad rtol=1e-4

loss_function = function(p)
    prob = remake(prob0_oop;u0=convert.(eltype(p),prob0.u0),p=p)
    prediction = solve(prob, Rodas5(); saveat = 0.0:0.5:10.0,abstol=1e-8,reltol=1e-8)

    tmpdata=prediction[[1,2],:];
    tdata=target_data[[1,2],:];

    # Calculate squared error
    return sum(abs2,tmpdata-tdata)
end

rdgrad = Zygote.gradient(loss_function,p)[1]
@test fdgrad ≈ rdgrad rtol=1e-4

# all implicit solvers
solvers = [
    # SDIRK Methods (ok)
    ImplicitEuler(),
    ImplicitMidpoint(),
    Trapezoid(),
    TRBDF2(),
    SDIRK2(),
    Kvaerno3(),
    KenCarp3(),
    Cash4(),
    Hairer4(),
    Hairer42(),
    Kvaerno4(),
    KenCarp4(),
    Kvaerno5(),
    KenCarp5(),
    # Fully-Implicit Runge-Kutta Methods (FIRK)
    RadauIIA5(),
    # Fully-Implicit Runge-Kutta Methods (FIRK)
    #PDIRK44(),
    # Rosenbrock Methods
    ROS3P(),
    Rodas3(),
    RosShamp4(),
    Veldd4(),
    Velds4(),
    GRK4T(),
    GRK4A(),
    Ros4LStab(),
    Rodas4(),
    Rodas42(),
    Rodas4P(),
    Rodas5(),
    # Rosenbrock-W Methods
    Rosenbrock23(),
    Rosenbrock32(),
    RosenbrockW6S4OS(),
    ROS34PW1a(),
    ROS34PW1b(),
    ROS34PW2(),
    ROS34PW3(),
    # Stabilized Explicit Methods (ok)
    ROCK2(),
    ROCK4(),
    RKC(),
    # SERK2v2(), not defined?
    ESERK5()];

p = rand(3)

function dudt(u,p,t)
    u .* p
end

for solver in solvers
    function loss(p)
        prob = ODEProblem(dudt, [3.0, 2.0, 1.0], (0.0, 1.0), p)
        sol = solve(prob, solver, dt=0.01)
        sum(abs2, Array(sol))
    end

    println(DiffEqBase.parameterless_type(solver))
    loss(p)
    Zygote.gradient(loss, p)
end
