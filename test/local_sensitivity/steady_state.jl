using Test, LinearAlgebra
using DiffEqSensitivity, SteadyStateDiffEq, DiffEqBase, NLsolve
using OrdinaryDiffEq
using ForwardDiff, Calculus
@testset "Adjoint sensitivities of steady state solver" begin
  function f!(du,u,p,t)
    du[1] = p[1] + p[2]*u[1]
    du[2] = p[3]*u[1] + p[4]*u[2]
  end

  function jac!(J,u,p,t) #df/dx
    J[1,1] = p[2]
    J[2,1] = p[3]
    J[1,2] = 0
    J[2,2] = p[4]
    nothing
  end

  function paramjac!(fp,u,p,t) #df/dp
    fp[1,1] = 1
    fp[2,1] = 0
    fp[1,2] = u[1]
    fp[2,2] = 0
    fp[1,3] = 0
    fp[2,3] = u[1]
    fp[1,4] = 0
    fp[2,4] = u[2]
    nothing
  end

  function dg!(out,u,p,t,i)
    (out.=-2.0.+u)
  end

  function g(u,p,t)
    sum((2.0.-u).^2)/2 + sum(p.^2)/2
  end

  u0 = zeros(2)
  p = [2.0,-2.0,1.0,-4.0]
  prob = SteadyStateProblem(f!,u0,p)
  abstol = 1e-10
  @testset "for p" begin
    println("Calculate adjoint sensitivities from Jacobians")

    sol_analytical = [-p[1]/p[2], p[1]*p[3]/(p[2]*p[4])]

    J = zeros(2,2)
    fp = zeros(2,4)
    gp = zeros(4)
    gx = zeros(1,2)
    delg_delp = copy(p)

    jac!(J,sol_analytical,p,nothing)
    dg!(vec(gx),sol_analytical,p,nothing,nothing)
    paramjac!(fp,sol_analytical,p,nothing)

    lambda = J' \gx'
    res_analytical = delg_delp'-lambda' * fp # = -gx*inv(J)*fp

    @info "Expected result" sol_analytical, res_analytical, delg_delp'-gx*inv(J)*fp


    @info "Calculate adjoint sensitivities from autodiff & numerical diff"
    function G(p)
      tmp_prob = remake(prob,u0=convert.(eltype(p),prob.u0),p=p)
      sol = solve(tmp_prob,
        SSRootfind(nlsolve = (f!,u0,abstol) -> (res=NLsolve.nlsolve(f!,u0,autodiff=:forward,method=:newton,iterations=Int(1e6),ftol=1e-14);res.zero))
        )
      A = convert(Array,sol)
      g(A,p,nothing)
    end
    res1 = ForwardDiff.gradient(G,p)
    res2 = Calculus.gradient(G,p)
    #@info res1, res2, res_analytical

    @test res1 ≈ res_analytical' rtol = 1e-7
    @test res2 ≈ res_analytical' rtol = 1e-7
    @test res1 ≈ res2 rtol = 1e-7


    @info "Adjoint sensitivities"

    # with jac, param_jac
    f1 = ODEFunction(f!;jac=jac!, paramjac=paramjac!)
    prob1 = SteadyStateProblem(f1,u0,p)
    sol1 = solve(prob1,DynamicSS(Rodas5()))
    res1a = adjoint_sensitivities(sol1,DynamicSS(Rodas5()),sensealg=SteadyStateAdjoint(),g,dg!)
    res1b = adjoint_sensitivities(sol1,DynamicSS(Rodas5()),sensealg=SteadyStateAdjoint(),g,nothing)
    res1c = adjoint_sensitivities(sol1,DynamicSS(Rodas5()),sensealg=SteadyStateAdjoint(autodiff=false),g,nothing)
    res1d = adjoint_sensitivities(sol1,DynamicSS(Rodas5()),sensealg=SteadyStateAdjoint(autojacvec=TrackerVJP()),g,nothing)
    res1e = adjoint_sensitivities(sol1,DynamicSS(Rodas5()),sensealg=SteadyStateAdjoint(autojacvec=ReverseDiffVJP()),g,nothing)
    res1f = adjoint_sensitivities(sol1,DynamicSS(Rodas5()),sensealg=SteadyStateAdjoint(autojacvec=ZygoteVJP()),g,nothing)
    res1g = adjoint_sensitivities(sol1,DynamicSS(Rodas5()),sensealg=SteadyStateAdjoint(autodiff=false,autojacvec=false),g,nothing)

    # with jac, without param_jac
    f2 = ODEFunction(f!;jac=jac!)
    prob2 = SteadyStateProblem(f2,u0,p)
    sol2 = solve(prob2,DynamicSS(Rodas5()))
    res2a = adjoint_sensitivities(sol2,DynamicSS(Rodas5()),sensealg=SteadyStateAdjoint(),g,dg!)
    res2b = adjoint_sensitivities(sol2,DynamicSS(Rodas5()),sensealg=SteadyStateAdjoint(),g,nothing)
    res2c = adjoint_sensitivities(sol2,DynamicSS(Rodas5()),sensealg=SteadyStateAdjoint(autodiff=false),g,nothing)
    res2d = adjoint_sensitivities(sol2,DynamicSS(Rodas5()),sensealg=SteadyStateAdjoint(autojacvec=TrackerVJP()),g,nothing)
    res2e = adjoint_sensitivities(sol2,DynamicSS(Rodas5()),sensealg=SteadyStateAdjoint(autojacvec=ReverseDiffVJP()),g,nothing)
    res2f = adjoint_sensitivities(sol2,DynamicSS(Rodas5()),sensealg=SteadyStateAdjoint(autojacvec=ZygoteVJP()),g,nothing)
    res2g = adjoint_sensitivities(sol2,DynamicSS(Rodas5()),sensealg=SteadyStateAdjoint(autodiff=false,autojacvec=false),g,nothing)

    # without jac, without param_jac
    f3 = ODEFunction(f!)
    prob3 = SteadyStateProblem(f3,u0,p)
    sol3 = solve(prob3,DynamicSS(Rodas5()))
    res3a = adjoint_sensitivities(sol3,DynamicSS(Rodas5()),sensealg=SteadyStateAdjoint(),g,dg!)
    res3b = adjoint_sensitivities(sol3,DynamicSS(Rodas5()),sensealg=SteadyStateAdjoint(),g,nothing)
    res3c = adjoint_sensitivities(sol3,DynamicSS(Rodas5()),sensealg=SteadyStateAdjoint(autodiff=false),g,nothing)
    res3d = adjoint_sensitivities(sol3,DynamicSS(Rodas5()),sensealg=SteadyStateAdjoint(autojacvec=TrackerVJP()),g,nothing)
    res3e = adjoint_sensitivities(sol3,DynamicSS(Rodas5()),sensealg=SteadyStateAdjoint(autojacvec=ReverseDiffVJP()),g,nothing)
    res3f = adjoint_sensitivities(sol3,DynamicSS(Rodas5()),sensealg=SteadyStateAdjoint(autojacvec=ZygoteVJP()),g,nothing)
    res3g = adjoint_sensitivities(sol3,DynamicSS(Rodas5()),sensealg=SteadyStateAdjoint(autodiff=false,autojacvec=false),g,nothing)

    @test norm(res_analytical' .- res1a) < 1e-7
    @test norm(res_analytical' .- res1b) < 1e-7
    @test norm(res_analytical' .- res1c) < 1e-7
    @test norm(res_analytical' .- res1d) < 1e-7
    @test norm(res_analytical' .- res1e) < 1e-7
    @test_broken norm(res_analytical' .- res1f) < 1e-7
    @test norm(res_analytical' .- res1g) < 1e-7
    @test norm(res_analytical' .- res2a) < 1e-7
    @test norm(res_analytical' .- res2b) < 1e-7
    @test norm(res_analytical' .- res2c) < 1e-7
    @test norm(res_analytical' .- res2d) < 1e-7
    @test norm(res_analytical' .- res2e) < 1e-7
    @test_broken norm(res_analytical' .- res2f) < 1e-7
    @test norm(res_analytical' .- res2g) < 1e-7
    @test norm(res_analytical' .- res3a) < 1e-7
    @test norm(res_analytical' .- res3b) < 1e-7
    @test norm(res_analytical' .- res3c) < 1e-7
    @test norm(res_analytical' .- res3d) < 1e-7
    @test norm(res_analytical' .- res3e) < 1e-7
    @test_broken norm(res_analytical' .- res3f) < 1e-7
    @test norm(res_analytical' .- res3g) < 1e-7

    @info "oop checks"
    function foop(u,p,t)
      dx = p[1] + p[2]*u[1]
      dy = p[3]*u[1] + p[4]*u[2]
      [dx,dy]
    end
    proboop = SteadyStateProblem(foop,u0,p)
    soloop = solve(proboop,DynamicSS(Rodas5()))


    res4a = adjoint_sensitivities(soloop,DynamicSS(Rodas5()),sensealg=SteadyStateAdjoint(),g,dg!)
    res4b = adjoint_sensitivities(soloop,DynamicSS(Rodas5()),sensealg=SteadyStateAdjoint(),g,nothing)
    res4c = adjoint_sensitivities(soloop,DynamicSS(Rodas5()),sensealg=SteadyStateAdjoint(autodiff=false),g,nothing)
    res4d = adjoint_sensitivities(soloop,DynamicSS(Rodas5()),sensealg=SteadyStateAdjoint(autojacvec=TrackerVJP()),g,nothing)
    res4e = adjoint_sensitivities(soloop,DynamicSS(Rodas5()),sensealg=SteadyStateAdjoint(autojacvec=ReverseDiffVJP()),g,nothing)
    res4f = adjoint_sensitivities(soloop,DynamicSS(Rodas5()),sensealg=SteadyStateAdjoint(autojacvec=ZygoteVJP()),g,nothing)
    res4g = adjoint_sensitivities(soloop,DynamicSS(Rodas5()),sensealg=SteadyStateAdjoint(autodiff=false,autojacvec=false),g,nothing)
    res4h = adjoint_sensitivities(soloop,DynamicSS(Rodas5()),sensealg=SteadyStateAdjoint(autodiff=true,autojacvec=false),g,nothing)

    @test norm(res_analytical' .- res4a) < 1e-7
    @test norm(res_analytical' .- res4b) < 1e-7
    @test norm(res_analytical' .- res4c) < 1e-7
    @test norm(res_analytical' .- res4d) < 1e-7
    @test norm(res_analytical' .- res4e) < 1e-7
    @test norm(res_analytical' .- res4f) < 1e-7
    @test norm(res_analytical' .- res4g) < 1e-7
    @test norm(res_analytical' .- res4h) < 1e-7
  end

  @testset "for u0: (should be zero, steady state does not depend on initial condition)" begin
    res5 = ForwardDiff.gradient(prob.u0) do u0
      tmp_prob = remake(prob,u0=u0)
      sol = solve(tmp_prob,
        SSRootfind(nlsolve = (f!,u0,abstol) -> (res=NLsolve.nlsolve(f!,u0,autodiff=:forward,method=:newton,iterations=Int(1e6),ftol=1e-14);res.zero) )
        )
      A = convert(Array,sol)
      g(A,p,nothing)
    end
    @test abs(dot(res5,res5)) < 1e-7
  end
end

using Zygote
@testset "concrete_solve derivatives steady state solver" begin

  function g1(u,p,t)
    sum(u)
  end

  function g2(u,p,t)
    sum((2.0.-u).^2)/2
  end

  u0 = zeros(2)
  p = [2.0,-2.0,1.0,-4.0]

  @testset "iip" begin
    function f!(du,u,p,t)
      du[1] = p[1] + p[2]*u[1]
      du[2] = p[3]*u[1] + p[4]*u[2]
    end
    prob = SteadyStateProblem(f!,u0,p)


    sol = solve(prob,DynamicSS(Rodas5()))
    res1 = adjoint_sensitivities(sol,DynamicSS(Rodas5()),sensealg=SteadyStateAdjoint(),g1,nothing)
    res2 = adjoint_sensitivities(sol,DynamicSS(Rodas5()),sensealg=SteadyStateAdjoint(),g2,nothing)


    dp1 = Zygote.gradient(p->sum(solve(prob,DynamicSS(Rodas5()),u0=u0,p=p,sensealg=SteadyStateAdjoint())),p)
    dp2 = Zygote.gradient(p->sum((2.0.-solve(prob,DynamicSS(Rodas5()),u0=u0,p=p,sensealg=SteadyStateAdjoint())).^2)/2.0,p)

    @test res1 ≈ dp1[1] rtol=1e-12
    @test res2 ≈ dp2[1] rtol=1e-12

    res1 = Zygote.gradient(p->sum(Array(solve(prob,DynamicSS(Rodas5()),u0=u0,p=p,sensealg=SteadyStateAdjoint()))[1]),p)
    dp1 = Zygote.gradient(p->sum(solve(prob,DynamicSS(Rodas5()),u0=u0,p=p,save_idxs=1:1,sensealg=SteadyStateAdjoint())),p)
    dp2 = Zygote.gradient(p->solve(prob,DynamicSS(Rodas5()),u0=u0,p=p,save_idxs=1,sensealg=SteadyStateAdjoint())[1],p)
    @test res1[1] ≈ dp1[1] rtol=1e-10
    @test res1[1] ≈ dp2[1] rtol=1e-10
  end

  @testset "oop" begin
    function f(u,p,t)
      dx = p[1] + p[2]*u[1]
      dy = p[3]*u[1] + p[4]*u[2]
      [dx,dy]
    end
    proboop = SteadyStateProblem(f,u0,p)


    soloop = solve(proboop,DynamicSS(Rodas5()))
    res1oop = adjoint_sensitivities(soloop,DynamicSS(Rodas5()),sensealg=SteadyStateAdjoint(),g1,nothing)
    res2oop = adjoint_sensitivities(soloop,DynamicSS(Rodas5()),sensealg=SteadyStateAdjoint(),g2,nothing)


    dp1oop = Zygote.gradient(p->sum(solve(proboop,DynamicSS(Rodas5()),u0=u0,p=p,sensealg=SteadyStateAdjoint())),p)
    dp2oop = Zygote.gradient(p->sum((2.0.-solve(proboop,DynamicSS(Rodas5()),u0=u0,p=p,sensealg=SteadyStateAdjoint())).^2)/2.0,p)

    @test res1oop ≈ dp1oop[1] rtol=1e-12
    @test res2oop ≈ dp2oop[1] rtol=1e-12

    res1oop = Zygote.gradient(p->sum(Array(solve(proboop,DynamicSS(Rodas5()),u0=u0,p=p,sensealg=SteadyStateAdjoint()))[1]),p)
    dp1oop = Zygote.gradient(p->sum(solve(proboop,DynamicSS(Rodas5()),u0=u0,p=p,save_idxs=1:1,sensealg=SteadyStateAdjoint())),p)
    dp2oop = Zygote.gradient(p->solve(proboop,DynamicSS(Rodas5()),u0=u0,p=p,save_idxs=1,sensealg=SteadyStateAdjoint())[1],p)
    @test res1oop[1] ≈ dp1oop[1] rtol=1e-10
    @test res1oop[1] ≈ dp2oop[1] rtol=1e-10
  end
end
