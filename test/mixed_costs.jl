using DiffEqSensitivity
using OrdinaryDiffEq
using ForwardDiff, FiniteDiff, Zygote
using QuadGK
using Test

abstol = 1e-14
reltol = 1e-14
savingtimes = collect(1.0:9.0)

function fiip(du, u, p, t)
  du[1] = dx = p[1] * u[1] - p[2] * u[1] * u[2]
  du[2] = dy = -p[3] * u[2] + p[4] * u[1] * u[2]
end

## Continuous cost functionals

function continuous_cost_forward(input)
  u0 = input[1:2]
  p = input[3:end]

  prob = ODEProblem(fiip, u0, (0.0, 10.0), p)
  sol = solve(prob, Tsit5(), abstol=abstol, reltol=reltol)
  cost, err = quadgk((t) -> sol(t)[1]^2 + p[1], prob.tspan..., atol=abstol, rtol=reltol)
  cost
end
p = [1.5, 1.0, 3.0, 1.0]
u0 = [1.0; 1.0]
input = vcat(u0, p)

dForwardDiff = ForwardDiff.gradient(continuous_cost_forward, input)
dFiniteDiff = FiniteDiff.finite_difference_gradient(continuous_cost_forward, input)
@test dForwardDiff ≈ dFiniteDiff

prob = ODEProblem(fiip, u0, (0.0, 10.0), p)
sol = solve(prob, Tsit5(), reltol=reltol, abstol=abstol)
g(u, p, t) = u[1]^2 + p[1]
function dgdu(out, u, p, t)
  out[1] = 2u[1]
  out[2] = 0.0
end
function dgdp(out, u, p, t)
  out[1] = 1.0
  out[2] = 0.0
  out[3] = 0.0
  out[4] = 0.0
end

# BacksolveAdjoint, all vjps
du0, dp = adjoint_sensitivities(sol, Tsit5(), g, nothing, (dgdu, dgdp), sensealg=BacksolveAdjoint(autojacvec=EnzymeVJP()), abstol=abstol, reltol=reltol)
@test du0 ≈ dForwardDiff[1:2]
@test dp' ≈ dForwardDiff[3:6]
du0, dp = adjoint_sensitivities(sol, Tsit5(), g, nothing, (dgdu, dgdp), sensealg=BacksolveAdjoint(autojacvec=ReverseDiffVJP()), abstol=abstol, reltol=reltol)
@test du0 ≈ dForwardDiff[1:2]
@test dp' ≈ dForwardDiff[3:6]
du0, dp = adjoint_sensitivities(sol, Tsit5(), g, nothing, (dgdu, dgdp), sensealg=BacksolveAdjoint(autojacvec=TrackerVJP()), abstol=abstol, reltol=reltol)
@test du0 ≈ dForwardDiff[1:2]
@test dp' ≈ dForwardDiff[3:6]
du0, dp = adjoint_sensitivities(sol, Tsit5(), g, nothing, (dgdu, dgdp), sensealg=BacksolveAdjoint(autojacvec=false), abstol=abstol, reltol=reltol)
@test du0 ≈ dForwardDiff[1:2]
@test dp' ≈ dForwardDiff[3:6]
du0, dp = adjoint_sensitivities(sol, Tsit5(), g, nothing, (dgdu, dgdp), sensealg=BacksolveAdjoint(autojacvec=ZygoteVJP()), abstol=abstol, reltol=reltol)
@test du0 ≈ dForwardDiff[1:2]
@test dp' ≈ dForwardDiff[3:6]

# InterpolatingAdjoint, all vjps
du0, dp = adjoint_sensitivities(sol, Tsit5(), g, nothing, (dgdu, dgdp), sensealg=InterpolatingAdjoint(autojacvec=EnzymeVJP()), abstol=abstol, reltol=reltol)
@test du0 ≈ dForwardDiff[1:2]
@test dp' ≈ dForwardDiff[3:6]
du0, dp = adjoint_sensitivities(sol, Tsit5(), g, nothing, (dgdu, dgdp), sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP()), abstol=abstol, reltol=reltol)
@test du0 ≈ dForwardDiff[1:2]
@test dp' ≈ dForwardDiff[3:6]
du0, dp = adjoint_sensitivities(sol, Tsit5(), g, nothing, (dgdu, dgdp), sensealg=InterpolatingAdjoint(autojacvec=TrackerVJP()), abstol=abstol, reltol=reltol)
@test du0 ≈ dForwardDiff[1:2]
@test dp' ≈ dForwardDiff[3:6]
du0, dp = adjoint_sensitivities(sol, Tsit5(), g, nothing, (dgdu, dgdp), sensealg=InterpolatingAdjoint(autojacvec=false), abstol=abstol, reltol=reltol)
@test du0 ≈ dForwardDiff[1:2]
@test dp' ≈ dForwardDiff[3:6]
du0, dp = adjoint_sensitivities(sol, Tsit5(), g, nothing, (dgdu, dgdp), sensealg=BacksolveAdjoint(autojacvec=ZygoteVJP()), abstol=abstol, reltol=reltol)
@test du0 ≈ dForwardDiff[1:2]
@test dp' ≈ dForwardDiff[3:6]

# QuadratureAdjoint, all vjps
du0, dp = adjoint_sensitivities(sol, Tsit5(), g, nothing, (dgdu, dgdp), sensealg=QuadratureAdjoint(autojacvec=EnzymeVJP(), abstol=abstol, reltol=reltol), abstol=abstol, reltol=reltol)
@test du0 ≈ dForwardDiff[1:2]
@test dp' ≈ dForwardDiff[3:6]
du0, dp = adjoint_sensitivities(sol, Tsit5(), g, nothing, (dgdu, dgdp), sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(), abstol=abstol, reltol=reltol), abstol=abstol, reltol=reltol)
@test du0 ≈ dForwardDiff[1:2]
@test dp' ≈ dForwardDiff[3:6]
du0, dp = adjoint_sensitivities(sol, Tsit5(), g, nothing, (dgdu, dgdp), sensealg=QuadratureAdjoint(autojacvec=false, abstol=abstol, reltol=reltol), abstol=abstol, reltol=reltol)
@test du0 ≈ dForwardDiff[1:2]
@test dp' ≈ dForwardDiff[3:6]
##

## Discrete costs

function discrete_cost_forward(input, sensealg=nothing)
  u0 = input[1:2]
  p = input[3:end]

  prob = ODEProblem(fiip, u0, (0.0, 10.0), p)
  sol = Array(solve(prob, Tsit5(), abstol=abstol, reltol=reltol, saveat=savingtimes, sensealg=sensealg, save_start=false, save_end=false))
  cost = zero(eltype(p))
  for u in eachcol(sol)
    cost += u[1]^2 #+ p[1]
  end
  cost
end
dForwardDiff = ForwardDiff.gradient(discrete_cost_forward, input)
dFiniteDiff = FiniteDiff.finite_difference_gradient(discrete_cost_forward, input)
@test dForwardDiff ≈ dFiniteDiff

function dgdu(out, u, p, t, i)
  out[1] = 2u[1]
  out[2] = 0.0
end
function dgdp(out, u, p, t, i)
  out[1] = 0.0
  out[2] = 0.0
  out[3] = 0.0
  out[4] = 0.0
end

# BacksolveAdjoint, all vjps
du0, dp = adjoint_sensitivities(sol, Tsit5(), dgdu, savingtimes, sensealg=BacksolveAdjoint(autojacvec=EnzymeVJP()), abstol=abstol, reltol=reltol)
@test du0 ≈ dForwardDiff[1:2]
@test dp' ≈ dForwardDiff[3:6]
du0, dp = adjoint_sensitivities(sol, Tsit5(), dgdu, savingtimes, sensealg=BacksolveAdjoint(autojacvec=ReverseDiffVJP()), abstol=abstol, reltol=reltol)
@test du0 ≈ dForwardDiff[1:2]
@test dp' ≈ dForwardDiff[3:6]
du0, dp = adjoint_sensitivities(sol, Tsit5(), dgdu, savingtimes, sensealg=BacksolveAdjoint(autojacvec=TrackerVJP()), abstol=abstol, reltol=reltol)
@test du0 ≈ dForwardDiff[1:2]
@test dp' ≈ dForwardDiff[3:6]
du0, dp = adjoint_sensitivities(sol, Tsit5(), dgdu, savingtimes, sensealg=BacksolveAdjoint(autojacvec=false), abstol=abstol, reltol=reltol)
@test du0 ≈ dForwardDiff[1:2]
@test dp' ≈ dForwardDiff[3:6]
du0, dp = adjoint_sensitivities(sol, Tsit5(), dgdu, savingtimes, sensealg=BacksolveAdjoint(autojacvec=ZygoteVJP()), abstol=abstol, reltol=reltol)
@test du0 ≈ dForwardDiff[1:2]
@test dp' ≈ dForwardDiff[3:6]

# InterpolatingAdjoint, all vjps
du0, dp = adjoint_sensitivities(sol, Tsit5(), dgdu, savingtimes, sensealg=InterpolatingAdjoint(autojacvec=EnzymeVJP()), abstol=abstol, reltol=reltol)
@test du0 ≈ dForwardDiff[1:2]
@test dp' ≈ dForwardDiff[3:6]
du0, dp = adjoint_sensitivities(sol, Tsit5(), dgdu, savingtimes, sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP()), abstol=abstol, reltol=reltol)
@test du0 ≈ dForwardDiff[1:2]
@test dp' ≈ dForwardDiff[3:6]
du0, dp = adjoint_sensitivities(sol, Tsit5(), dgdu, savingtimes, sensealg=InterpolatingAdjoint(autojacvec=TrackerVJP()), abstol=abstol, reltol=reltol)
@test du0 ≈ dForwardDiff[1:2]
@test dp' ≈ dForwardDiff[3:6]
du0, dp = adjoint_sensitivities(sol, Tsit5(), dgdu, savingtimes, sensealg=InterpolatingAdjoint(autojacvec=false), abstol=abstol, reltol=reltol)
@test du0 ≈ dForwardDiff[1:2]
@test dp' ≈ dForwardDiff[3:6]
du0, dp = adjoint_sensitivities(sol, Tsit5(), dgdu, savingtimes, sensealg=BacksolveAdjoint(autojacvec=ZygoteVJP()), abstol=abstol, reltol=reltol)
@test du0 ≈ dForwardDiff[1:2]
@test dp' ≈ dForwardDiff[3:6]

# QuadratureAdjoint, all vjps
du0, dp = adjoint_sensitivities(sol, Tsit5(), dgdu, savingtimes, sensealg=QuadratureAdjoint(autojacvec=EnzymeVJP(), abstol=abstol, reltol=reltol), abstol=abstol, reltol=reltol)
@test du0 ≈ dForwardDiff[1:2]
@test dp' ≈ dForwardDiff[3:6]
du0, dp = adjoint_sensitivities(sol, Tsit5(), dgdu, savingtimes, sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(), abstol=abstol, reltol=reltol), abstol=abstol, reltol=reltol)
@test du0 ≈ dForwardDiff[1:2]
@test dp' ≈ dForwardDiff[3:6]
du0, dp = adjoint_sensitivities(sol, Tsit5(), dgdu, savingtimes, sensealg=QuadratureAdjoint(autojacvec=false, abstol=abstol, reltol=reltol), abstol=abstol, reltol=reltol)
@test du0 ≈ dForwardDiff[1:2]
@test dp' ≈ dForwardDiff[3:6]

# concrete solve interface
dZygote = Zygote.gradient(input -> discrete_cost_forward(input, BacksolveAdjoint()), input)[1]
@test dZygote ≈ dForwardDiff
dZygote = Zygote.gradient(input -> discrete_cost_forward(input, InterpolatingAdjoint()), input)[1]
@test dZygote ≈ dForwardDiff
dZygote = Zygote.gradient(input -> discrete_cost_forward(input, QuadratureAdjoint()), input)[1]
@test dZygote ≈ dForwardDiff
##


## Mixed costs
