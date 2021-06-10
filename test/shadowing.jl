using Random; Random.seed!(1234)
using OrdinaryDiffEq
using Statistics
using ForwardDiff, Calculus
using DiffEqSensitivity
using Test


@testset "LSS" begin
  @testset "Lorentz single parameter" begin
    function lorenz!(du,u,p,t)
      du[1] = 10*(u[2]-u[1])
      du[2] = u[1]*(p[1]-u[3]) - u[2]
      du[3] = u[1]*u[2] - (8//3)*u[3]
    end

    p = [28.0]
    tspan_init = (0.0,30.0)
    tspan_attractor = (30.0,50.0)
    u0 = rand(3)
    prob_init = ODEProblem(lorenz!,u0,tspan_init,p)
    sol_init = solve(prob_init,Tsit5())
    prob_attractor = ODEProblem(lorenz!,sol_init[end],tspan_attractor,p)
    sol_attractor = solve(prob_attractor,Vern9(),abstol=1e-14,reltol=1e-14)

    g(u,p,t) = u[end]
    lss_problem1 = ForwardLSSProblem(sol_attractor, ForwardLSS(), g)
    lss_problem2 = ForwardLSSProblem(sol_attractor, ForwardLSS(alpha=DiffEqSensitivity.Cos2Windowing()), g)
    lss_problem3 = ForwardLSSProblem(sol_attractor, ForwardLSS(alpha=10), g)

    res1 = DiffEqSensitivity.solve(lss_problem1)
    res2 = DiffEqSensitivity.solve(lss_problem2)
    res3 = DiffEqSensitivity.solve(lss_problem3)

    @test res1[1] ≈ 1 atol=5e-2
    @test res2[1] ≈ 1 atol=5e-2
    @test res3[1] ≈ 1 atol=5e-2
  end
end
