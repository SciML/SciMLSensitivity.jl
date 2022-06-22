using OrdinaryDiffEq, Zygote
using DiffEqSensitivity, Test, ForwardDiff

abstol = 1e-12
reltol = 1e-12
savingtimes = 0.5

function test_continuous_callback(cb, g, dg!; only_backsolve=false)
  function fiip(du, u, p, t)
    du[1] = u[2]
    du[2] = -p[1]
  end
  function foop(u, p, t)
    dx = u[2]
    dy = -p[1]
    [dx, dy]
  end

  u0 = [5.0, 0.0]
  tspan = (0.0, 2.5)
  p = [9.8, 0.8]

  prob = ODEProblem(fiip, u0, tspan, p)
  proboop = ODEProblem(fiip, u0, tspan, p)

  sol1 = solve(prob, Tsit5(), u0=u0, p=p, callback=cb, abstol=abstol, reltol=reltol, saveat=savingtimes)
  sol2 = solve(prob, Tsit5(), u0=u0, p=p, abstol=abstol, reltol=reltol, saveat=savingtimes)

  if cb.save_positions == [1, 1]
    @test length(sol1.t) != length(sol2.t)
  else
    @test length(sol1.t) == length(sol2.t)
  end

  du01, dp1 = @time Zygote.gradient(
    (u0, p) -> g(solve(prob, Tsit5(), u0=u0, p=p, callback=cb, abstol=abstol, reltol=reltol, saveat=savingtimes, sensealg=BacksolveAdjoint())),
    u0, p)

  du01b, dp1b = Zygote.gradient(
    (u0, p) -> g(solve(proboop, Tsit5(), u0=u0, p=p, callback=cb, abstol=abstol, reltol=reltol, saveat=savingtimes, sensealg=BacksolveAdjoint())),
    u0, p)

  du01c, dp1c = Zygote.gradient(
    (u0, p) -> g(solve(proboop, Tsit5(), u0=u0, p=p, callback=cb, abstol=abstol, reltol=reltol, saveat=savingtimes, sensealg=BacksolveAdjoint(checkpointing=false))),
    u0, p)

  if !only_backsolve
    @test_broken du02, dp2 = @time Zygote.gradient(
      (u0, p) -> g(solve(prob, Tsit5(), u0=u0, p=p, callback=cb, abstol=abstol, reltol=reltol, saveat=savingtimes, sensealg=ReverseDiffAdjoint())), u0, p)

    du03, dp3 = @time Zygote.gradient(
      (u0, p) -> g(solve(prob, Tsit5(), u0=u0, p=p, callback=cb, abstol=abstol, reltol=reltol, saveat=savingtimes, sensealg=InterpolatingAdjoint(checkpointing=true))),
      u0, p)

    du03c, dp3c = Zygote.gradient(
      (u0, p) -> g(solve(prob, Tsit5(), u0=u0, p=p, callback=cb, abstol=abstol, reltol=reltol, saveat=savingtimes, sensealg=InterpolatingAdjoint(checkpointing=false))),
      u0, p)

    du04, dp4 = @time Zygote.gradient(
      (u0, p) -> g(solve(prob, Tsit5(), u0=u0, p=p, callback=cb, abstol=abstol, reltol=reltol, saveat=savingtimes, sensealg=QuadratureAdjoint())),
      u0, p)
  end
  dstuff = @time ForwardDiff.gradient(
    (θ) -> g(solve(prob, Tsit5(), u0=θ[1:2], p=θ[3:4], callback=cb, abstol=abstol, reltol=reltol, saveat=savingtimes)),
    [u0; p])

  @info dstuff

  @test du01 ≈ dstuff[1:2]
  @test dp1 ≈ dstuff[3:4]
  @test du01b ≈ dstuff[1:2]
  @test dp1b ≈ dstuff[3:4]
  @test du01c ≈ dstuff[1:2]
  @test dp1c ≈ dstuff[3:4]
  if !only_backsolve
    @test_broken du01 ≈ du02
    @test du01 ≈ du03 rtol = 1e-7
    @test du01 ≈ du03c rtol = 1e-7
    @test du03 ≈ du03c
    @test du01 ≈ du04
    @test_broken dp1 ≈ dp2
    @test dp1 ≈ dp3
    @test dp1 ≈ dp3c
    @test dp3 ≈ dp3c
    @test dp1 ≈ dp4 rtol = 1e-7

    @test_broken du02 ≈ dstuff[1:2]
    @test_broken dp2 ≈ dstuff[3:4]
  end

  cb2 = DiffEqSensitivity.track_callbacks(CallbackSet(cb), prob.tspan[1], prob.u0, prob.p, BacksolveAdjoint(autojacvec=ReverseDiffVJP()))
  sol_track = solve(prob, Tsit5(), u0=u0, p=p, callback=cb2, abstol=abstol, reltol=reltol, saveat=savingtimes)

  adj_prob = ODEAdjointProblem(sol_track, BacksolveAdjoint(autojacvec=ReverseDiffVJP()), dg!, sol_track.t, nothing,
    callback=cb2,
    abstol=abstol, reltol=reltol)
  adj_sol = solve(adj_prob, Tsit5(), abstol=abstol, reltol=reltol)
  @test du01 ≈ adj_sol[1:2, end]
  @test dp1 ≈ adj_sol[3:4, end]


  # adj_prob = ODEAdjointProblem(sol_track,InterpolatingAdjoint(),dg!,sol_track.t,nothing,
  # 						 callback = cb2,
  # 						 abstol=abstol,reltol=reltol)
  # adj_sol = solve(adj_prob, Tsit5(), abstol=abstol,reltol=reltol)
  #
  # @test du01 ≈ -adj_sol[1:2,end]
  # @test dp1 ≈ adj_sol[3:6,end]
end

println("Continuous Callbacks")
@testset "Continuous callbacks" begin
  @testset "simple loss function bouncing ball" begin
    g(sol) = sum(sol)
    function dg!(out, u, p, t, i)
      (out .= 1)
    end

    @testset "callbacks with no effect" begin
      condition(u, t, integrator) = u[1] # Event when event_f(u,t) == 0
      affect!(integrator) = (integrator.u[2] += 0)
      cb = ContinuousCallback(condition, affect!, save_positions=(false, false))
      test_continuous_callback(cb, g, dg!)
    end
    @testset "callbacks with no effect except saving the state" begin
      condition(u, t, integrator) = u[1]
      affect!(integrator) = (integrator.u[2] += 0)
      cb = ContinuousCallback(condition, affect!, save_positions=(true, true))
      test_continuous_callback(cb, g, dg!)
    end
    @testset "+= callback" begin
      condition(u, t, integrator) = u[1]
      affect!(integrator) = (integrator.u[2] += 50.0)
      cb = ContinuousCallback(condition, affect!, save_positions=(true, true))
      test_continuous_callback(cb, g, dg!)
    end
    @testset "= callback with parameter dependence and save" begin
      condition(u, t, integrator) = u[1]
      affect!(integrator) = (integrator.u[2] = -integrator.p[2] * integrator.u[2])
      cb = ContinuousCallback(condition, affect!, save_positions=(true, true))
      test_continuous_callback(cb, g, dg!)
    end
    @testset "= callback with parameter dependence but without save" begin
      condition(u, t, integrator) = u[1]
      affect!(integrator) = (integrator.u[2] = -integrator.p[2] * integrator.u[2])
      cb = ContinuousCallback(condition, affect!, save_positions=(false, false))
      test_continuous_callback(cb, g, dg!; only_backsolve=true)
    end
    @testset "= callback with non-linear affect" begin
      condition(u, t, integrator) = u[1]
      affect!(integrator) = (integrator.u[2] = integrator.u[2]^2)
      cb = ContinuousCallback(condition, affect!, save_positions=(true, true))
      test_continuous_callback(cb, g, dg!)
    end
    @testset "= callback with terminate" begin
      condition(u, t, integrator) = u[1]
      affect!(integrator) = (integrator.u[2] = -integrator.p[2] * integrator.u[2]; terminate!(integrator))
      cb = ContinuousCallback(condition, affect!, save_positions=(true, true))
      test_continuous_callback(cb, g, dg!; only_backsolve=true)
    end
  end
  @testset "MSE loss function bouncing-ball like" begin
    g(u) = sum((1.0 .- u) .^ 2) ./ 2
    dg!(out, u, p, t, i) = (out .= -1.0 .+ u)
    condition(u, t, integrator) = u[1]
    @testset "callback with non-linear affect" begin
      function affect!(integrator)
        integrator.u[1] += 3.0
        integrator.u[2] = integrator.u[2]^2
      end
      cb = ContinuousCallback(condition, affect!, save_positions=(true, true))
      test_continuous_callback(cb, g, dg!)
    end
    @testset "callback with non-linear affect and terminate" begin
      function affect!(integrator)
        integrator.u[1] += 3.0
        integrator.u[2] = integrator.u[2]^2
        terminate!(integrator)
      end
      cb = ContinuousCallback(condition, affect!, save_positions=(true, true))
      test_continuous_callback(cb, g, dg!; only_backsolve=true)
    end
  end
  @testset "MSE loss function free particle" begin
    g(u) = sum((1.0 .- u) .^ 2) ./ 2
    function fiip(du, u, p, t)
      du[1] = u[2]
      du[2] = 0
    end
    function foop(u, p, t)
      dx = u[2]
      dy = 0
      [dx, dy]
    end

    u0 = [5.0, -1.0]
    p = [0.0, 0.0]
    tspan = (0.0, 2.0)

    prob = ODEProblem(fiip, u0, tspan, p)
    proboop = ODEProblem(fiip, u0, tspan, p)

    condition(u, t, integrator) = u[1] # Event when event_f(u,t) == 0
    affect!(integrator) = (integrator.u[2] = -integrator.u[2])
    cb = ContinuousCallback(condition, affect!)

    du01, dp1 = Zygote.gradient(
      (u0, p) -> g(solve(prob, Tsit5(), u0=u0, p=p, callback=cb, abstol=abstol, reltol=reltol, saveat=savingtimes, sensealg=BacksolveAdjoint())),
      u0, p)

    dstuff = @time ForwardDiff.gradient(
      (θ) -> g(solve(prob, Tsit5(), u0=θ[1:2], p=θ[3:4], callback=cb, abstol=abstol, reltol=reltol, saveat=savingtimes)),
      [u0; p])

    @info dstuff

    @test du01 ≈ dstuff[1:2]
    @test dp1 ≈ dstuff[3:4]
  end
end
