using OrdinaryDiffEq, Zygote
using DiffEqSensitivity, Test, ForwardDiff

abstol = 1e-12
reltol = 1e-12
savingtimes = 0.5

function test_discrete_callback(cb, tstops, g, dg!, cboop=nothing, tprev=false)
  function fiip(du, u, p, t)
    du[1] = dx = p[1] * u[1] - p[2] * u[1] * u[2]
    du[2] = dy = -p[3] * u[2] + p[4] * u[1] * u[2]
  end
  function foop(u, p, t)
    dx = p[1] * u[1] - p[2] * u[1] * u[2]
    dy = -p[3] * u[2] + p[4] * u[1] * u[2]
    [dx, dy]
  end

  p = [1.5, 1.0, 3.0, 1.0]
  u0 = [1.0; 1.0]

  prob = ODEProblem(fiip, u0, (0.0, 10.0), p)
  proboop = ODEProblem(foop, u0, (0.0, 10.0), p)

  sol1 = solve(prob, Tsit5(), u0=u0, p=p, callback=cb, tstops=tstops, abstol=abstol, reltol=reltol, saveat=savingtimes)
  sol2 = solve(prob, Tsit5(), u0=u0, p=p, tstops=tstops, abstol=abstol, reltol=reltol, saveat=savingtimes)

  if cb.save_positions == [1, 1]
    @test length(sol1.t) != length(sol2.t)
  else
    @test length(sol1.t) == length(sol2.t)
  end

  du01, dp1 = Zygote.gradient(
    (u0, p) -> g(solve(prob, Tsit5(), u0=u0, p=p, callback=cb, tstops=tstops, abstol=abstol, reltol=reltol, saveat=savingtimes, sensealg=BacksolveAdjoint())),
    u0, p)

  du01b, dp1b = Zygote.gradient(
    (u0, p) -> g(solve(proboop, Tsit5(), u0=u0, p=p, callback=cb, tstops=tstops, abstol=abstol, reltol=reltol, saveat=savingtimes, sensealg=BacksolveAdjoint())),
    u0, p)

  du01c, dp1c = Zygote.gradient(
    (u0, p) -> g(solve(proboop, Tsit5(), u0=u0, p=p, callback=cb, tstops=tstops, abstol=abstol, reltol=reltol, saveat=savingtimes, sensealg=BacksolveAdjoint(checkpointing=false))),
    u0, p)

  if cboop === nothing
    du02, dp2 = Zygote.gradient(
      (u0, p) -> g(solve(prob, Tsit5(), u0=u0, p=p, callback=cb, tstops=tstops, abstol=abstol, reltol=reltol, saveat=savingtimes, sensealg=ReverseDiffAdjoint())), u0, p)
  else
    du02, dp2 = Zygote.gradient(
      (u0, p) -> g(solve(prob, Tsit5(), u0=u0, p=p, callback=cboop, tstops=tstops, abstol=abstol, reltol=reltol, saveat=savingtimes, sensealg=ReverseDiffAdjoint())), u0, p)
  end

  du03, dp3 = Zygote.gradient(
    (u0, p) -> g(solve(prob, Tsit5(), u0=u0, p=p, callback=cb, tstops=tstops, abstol=abstol, reltol=reltol, saveat=savingtimes, sensealg=InterpolatingAdjoint(checkpointing=true))),
    u0, p)

  du03c, dp3c = Zygote.gradient(
    (u0, p) -> g(solve(prob, Tsit5(), u0=u0, p=p, callback=cb, tstops=tstops, abstol=abstol, reltol=reltol, saveat=savingtimes, sensealg=InterpolatingAdjoint(checkpointing=false))),
    u0, p)

  du04, dp4 = Zygote.gradient(
    (u0, p) -> g(solve(prob, Tsit5(), u0=u0, p=p, callback=cb, tstops=tstops, abstol=abstol, reltol=reltol, saveat=savingtimes, sensealg=QuadratureAdjoint())),
    u0, p)

  dstuff = ForwardDiff.gradient(
    (θ) -> g(solve(prob, Tsit5(), u0=θ[1:2], p=θ[3:6], callback=cb, tstops=tstops, abstol=abstol, reltol=reltol, saveat=savingtimes)),
    [u0; p])

  @info dstuff

  # tests wrt discrete sensitivities
  if tprev
    # tprev depends on stepping behaviour of integrator. Thus sensitivities are necessarily (slightly) different.
    @test du02 ≈ dstuff[1:2] rtol = 1e-3
    @test dp2 ≈ dstuff[3:6] rtol = 1e-3
    @test du01 ≈ dstuff[1:2] rtol = 1e-3
    @test dp1 ≈ dstuff[3:6] rtol = 1e-3
    @test du01 ≈ du02 rtol = 1e-3
    @test dp1 ≈ dp2 rtol = 1e-3
  else
    @test du02 ≈ dstuff[1:2]
    @test dp2 ≈ dstuff[3:6]
    @test du01 ≈ dstuff[1:2]
    @test dp1 ≈ dstuff[3:6]
    @test du01 ≈ du02
    @test dp1 ≈ dp2
  end

  # tests wrt continuous sensitivities
  @test du01b ≈ du01
  @test dp1b ≈ dp1
  @test du01c ≈ du01
  @test dp1c ≈ dp1
  @test du01 ≈ du03 rtol = 1e-7
  @test du01 ≈ du03c rtol = 1e-7
  @test du03 ≈ du03c
  @test du01 ≈ du04
  @test dp1 ≈ dp3
  @test dp1 ≈ dp3c
  @test dp1 ≈ dp4 rtol = 1e-7

  cb2 = DiffEqSensitivity.track_callbacks(CallbackSet(cb), prob.tspan[1], prob.u0, prob.p, BacksolveAdjoint(autojacvec=ReverseDiffVJP()))
  sol_track = solve(prob, Tsit5(), u0=u0, p=p, callback=cb2, tstops=tstops, abstol=abstol, reltol=reltol, saveat=savingtimes)
  #cb_adj = DiffEqSensitivity.setup_reverse_callbacks(cb2,BacksolveAdjoint())

  adj_prob = ODEAdjointProblem(sol_track, BacksolveAdjoint(autojacvec=ReverseDiffVJP()), dg!, sol_track.t, nothing,
    callback=cb2,
    abstol=abstol, reltol=reltol)
  adj_sol = solve(adj_prob, Tsit5(), abstol=abstol, reltol=reltol)
  @test du01 ≈ adj_sol[1:2, end]
  @test dp1 ≈ adj_sol[3:6, end]


  # adj_prob = ODEAdjointProblem(sol_track,InterpolatingAdjoint(),dg!,sol_track.t,nothing,
  # 						 callback = cb2,
  # 						 abstol=abstol,reltol=reltol)
  # adj_sol = solve(adj_prob, Tsit5(), abstol=abstol,reltol=reltol)
  #
  # @test du01 ≈ -adj_sol[1:2,end]
  # @test dp1 ≈ adj_sol[3:6,end]
end

@testset "Discrete callbacks" begin
  @testset "ODEs" begin
    println("ODEs")
    @testset "simple loss function" begin
      g(sol) = sum(sol)
      function dg!(out, u, p, t, i)
        (out .= 1)
      end
      @testset "callbacks with no effect" begin
        condition(u, t, integrator) = t == 5
        affect!(integrator) = integrator.u[1] += 0.0
        cb = DiscreteCallback(condition, affect!, save_positions=(false, false))
        tstops = [5.0]
        test_discrete_callback(cb, tstops, g, dg!)
      end
      @testset "callbacks with no effect except saving the state" begin
        condition(u, t, integrator) = t == 5
        affect!(integrator) = integrator.u[1] += 0.0
        cb = DiscreteCallback(condition, affect!)
        tstops = [5.0]
        test_discrete_callback(cb, tstops, g, dg!)
      end
      @testset "callback at single time point" begin
        condition(u, t, integrator) = t == 5
        affect!(integrator) = integrator.u[1] += 2.0
        cb = DiscreteCallback(condition, affect!)
        tstops = [5.0]
        test_discrete_callback(cb, tstops, g, dg!)
      end
      @testset "callback at multiple time points" begin
        affecttimes = [2.03, 4.0, 8.0]
        condition(u, t, integrator) = t ∈ affecttimes
        affect!(integrator) = integrator.u[1] += 2.0
        cb = DiscreteCallback(condition, affect!)
        test_discrete_callback(cb, affecttimes, g, dg!)
      end
      @testset "state-dependent += callback at single time point" begin
        condition(u, t, integrator) = t == 5
        affect!(integrator) = (integrator.u .+= integrator.p[2] / 8 * sin.(integrator.u))
        cb = DiscreteCallback(condition, affect!)
        tstops = [5.0]
        test_discrete_callback(cb, tstops, g, dg!)
      end
      @testset "other callback at single time point" begin
        condition(u, t, integrator) = t == 5
        affect!(integrator) = (integrator.u[1] = 2.0; @show "triggered!")
        cb = DiscreteCallback(condition, affect!)
        tstops = [5.0]
        test_discrete_callback(cb, tstops, g, dg!)
      end
      @testset "parameter changing callback at single time point" begin
        condition(u, t, integrator) = t == 5.1
        affect!(integrator) = (integrator.p .= 2 * integrator.p .- 0.5)
        affect(integrator) = (integrator.p = 2 * integrator.p .- 0.5)
        cb = DiscreteCallback(condition, affect!)
        cboop = DiscreteCallback(condition, affect)
        cb = DiscreteCallback(condition, affect!)
        tstops = [5.1]
        test_discrete_callback(cb, tstops, g, dg!, cboop)
      end
      @testset "tprev dependent callback" begin
        condition(u, t, integrator) = t == 5
        affect!(integrator) = (@show integrator.tprev; integrator.u[1] += integrator.t - integrator.tprev)
        cb = DiscreteCallback(condition, affect!)
        tstops = [4.999, 5.0]
        test_discrete_callback(cb, tstops, g, dg!, nothing, true)
      end
    end
    @testset "MSE loss function" begin
      g(u) = sum((1.0 .- u) .^ 2) ./ 2
      dg!(out, u, p, t, i) = (out .= -1.0 .+ u)
      @testset "callbacks with no effect" begin
        condition(u, t, integrator) = t == 5
        affect!(integrator) = integrator.u[1] += 0.0
        cb = DiscreteCallback(condition, affect!, save_positions=(false, false))
        tstops = [5.0]
        test_discrete_callback(cb, tstops, g, dg!)
      end
      @testset "callbacks with no effect except saving the state" begin
        condition(u, t, integrator) = t == 5
        affect!(integrator) = integrator.u[1] += 0.0
        cb = DiscreteCallback(condition, affect!)
        tstops = [5.0]
        test_discrete_callback(cb, tstops, g, dg!)
      end
      @testset "callback at single time point" begin
        condition(u, t, integrator) = t == 5
        affect!(integrator) = integrator.u[1] += 2.0
        cb = DiscreteCallback(condition, affect!)
        tstops = [5.0]
        test_discrete_callback(cb, tstops, g, dg!)
      end
      @testset "callback at multiple time points" begin
        affecttimes = [2.03, 4.0, 8.0]
        condition(u, t, integrator) = t ∈ affecttimes
        affect!(integrator) = integrator.u[1] += 2.0
        cb = DiscreteCallback(condition, affect!)
        test_discrete_callback(cb, affecttimes, g, dg!)
      end
      @testset "state-dependent += callback at single time point" begin
        condition(u, t, integrator) = t == 5
        affect!(integrator) = (integrator.u .+= integrator.p[2] / 8 * sin.(integrator.u))
        cb = DiscreteCallback(condition, affect!)
        tstops = [5.0]
        test_discrete_callback(cb, tstops, g, dg!)
      end
      @testset "other callback at single time point" begin
        condition(u, t, integrator) = t == 5
        affect!(integrator) = (integrator.u[1] = 2.0; @show "triggered!")
        cb = DiscreteCallback(condition, affect!)
        tstops = [5.0]
        test_discrete_callback(cb, tstops, g, dg!)
      end
      @testset "parameter changing callback at single time point" begin
        condition(u, t, integrator) = t == 5.1
        affect!(integrator) = (integrator.p .= 2 * integrator.p .- 0.5)
        affect(integrator) = (integrator.p = 2 * integrator.p .- 0.5)
        cb = DiscreteCallback(condition, affect!)
        cboop = DiscreteCallback(condition, affect)
        tstops = [5.1]
        test_discrete_callback(cb, tstops, g, dg!, cboop)
      end
      @testset "tprev dependent callback" begin
        condition(u, t, integrator) = t == 5
        affect!(integrator) = (@show integrator.tprev; integrator.u[1] += integrator.t - integrator.tprev)
        cb = DiscreteCallback(condition, affect!)
        tstops = [4.999, 5.0]
        test_discrete_callback(cb, tstops, g, dg!, nothing, true)
      end
    end
  end
end
