using Test, LinearAlgebra
using DiffEqSensitivity, StochasticDiffEq
using Random

@info "SDE scalar Adjoints"

seed = 5
Random.seed!(seed)

tstart = 0.0
tend = 0.1
dt = 0.005
trange = (tstart, tend)
t = tstart:dt:tend
tarray = collect(t)

function g(u,p,t)
  sum(u.^2.0/2.0)
end

function dg!(out,u,p,t,i)
  (out.=-u)
end

p2 = [1.01,0.37]

#using DiffEqNoiseProcess

dtscalar = tend/1e4

f(u,p,t) = p[1]*u
σ(u,p,t) = p[2]*u
linear_analytic(u0,p,t,W) = @.(u0*exp((p[1]-p[2]^2/2)*t+p[2]*W))

Random.seed!(seed)
#W = WienerProcess(0.0,0.0,0.0)
u0 = rand(1)


Random.seed!(seed)
prob = SDEProblem(SDEFunction(f,σ,
  #analytic=linear_analytic
  ),σ,u0,trange,p2,
  #noise=W
 )
sol = solve(prob,SOSRI(), dt=dtscalar, adaptive=false, save_noise=true)
#@test isapprox(sol.u_analytic,sol.u, atol=1e-4)

res_sde_u0a, res_sde_pa = adjoint_sensitivities(sol,EM(),dg!,Array(t)
  ,dt=dtscalar,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=DiffEqSensitivity.ReverseDiffVJP()))

@info res_sde_u0a, res_sde_pa

res_sde_u0b, res_sde_pb = adjoint_sensitivities(sol,EM(),dg!,Array(t)
  ,dt=dtscalar,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=false))

@info res_sde_u0b, res_sde_pb

@test isapprox(res_sde_u0a, res_sde_u0b, rtol=1e-9)
@test isapprox(res_sde_pa, res_sde_pb, rtol=1e-9)

using ForwardDiff
function GSDE(p)
  Random.seed!(seed)
  tmp_prob = remake(prob,u0=eltype(p).(prob.u0),p=p,
                  tspan=eltype(p).(prob.tspan))
  _sol = solve(tmp_prob,SOSRI(),dt=dtscalar,adaptive=false,saveat=Array(t), sensealg=DiffEqBase.SensitivityADPassThrough())
  A = convert(Array,_sol)
  res = g(A,p,nothing)
end

res_sde_forward = ForwardDiff.gradient(GSDE,p2)

@info res_sde_forward

Wfix = [sol.W(t)[1][1] for t in tarray]
resp1 = sum(@. tarray*u0^2*exp(2*(p2[1]-p2[2]^2/2)*tarray+2*p2[2]*Wfix))
resp2 = sum(@. (Wfix-p2[2]*tarray)*u0^2*exp(2*(p2[1]-p2[2]^2/2)*tarray+2*p2[2]*Wfix))
resp = [resp1, resp2]

@test isapprox(resp, res_sde_pa', rtol=1e-3)
@test isapprox(resp, res_sde_forward, rtol=1e-6)



#
#
# # scalar noise inplace not working yet!!
# @testset "SDE inplace scalar noise tests (Ito)" begin
#   using DiffEqNoiseProcess
#
#   dtscalar = tend/1e2
#
#   f!(du,u,p,t) = (du .= p[1]*u)
#   σ!(du,u,p,t) = (du .= p[2]*u)
#
#   @info "scalar SDE"
#
#   Random.seed!(seed)
#   W = WienerProcess(0.0,0.0,0.0)
#   u0 = rand(2)
#
#   linear_analytic(u0,p,t,W) = @.(u0*exp((p[1]-p[2]^2/2)*t+p[2]*W))
#
#   prob = SDEProblem(SDEFunction(f!,σ!,analytic=linear_analytic),σ!,u0,trange,p2,
#     noise=W
#     )
#   sol = solve(prob,SOSRI(), dt=dtscalar, adaptive=false, save_noise=true)
#
#   @test isapprox(sol.u_analytic,sol.u, atol=1e-4)
#
#   res_sde_u0, res_sde_p = adjoint_sensitivities(sol,EM(),dg!,Array(t)
#     ,dt=dtscalar,adaptive=false,sensealg=BacksolveAdjoint())
#
#   @show res_sde_u0, res_sde_p
#
#   res_sde_u02, res_sde_p2 = adjoint_sensitivities(sol,EM(),dg!,Array(t)
#     ,dt=dtscalar,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=false))
#
#   @test isapprox(res_sde_u0, res_sde_u02,  rtol=1e-4)
#   @test isapprox(res_sde_p, res_sde_p2,  atol=1e-4)
#
#   res_sde_u02, res_sde_p2 = adjoint_sensitivities(sol,EM(),dg!,Array(t)
#     ,dt=dtscalar,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=DiffEqSensitivity.ReverseDiffVJP()))
#
#   @test isapprox(res_sde_u0, res_sde_u02,  rtol=1e-4)
#   @test isapprox(res_sde_p, res_sde_p2,  atol=1e-4)
#
#   @show res_sde_u02, res_sde_p2
#
#
#   res_sde_u02, res_sde_p2 = adjoint_sensitivities(sol,EM(),dg!,Array(t)
#     ,dt=tend/1e2,adaptive=false,sensealg=InterpolatingAdjoint())
#
#
#   @test isapprox(res_sde_u0, res_sde_u02,  rtol=1e-4)
#   @test isapprox(res_sde_p, res_sde_p2,  atol=1e-4)
#
#   @show res_sde_u02, res_sde_p2
#
#   res_sde_u02, res_sde_p2 = adjoint_sensitivities(sol,EM(),dg!,Array(t)
#     ,dt=dtscalar,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=false))
#
#   @test isapprox(res_sde_u0, res_sde_u02,  rtol=1e-4)
#   @test isapprox(res_sde_p, res_sde_p2,  atol=1e-4)
#
#   @show res_sde_u02, res_sde_p2
#
#   res_sde_u02, res_sde_p2 = adjoint_sensitivities(sol,EM(),dg!,Array(t)
#     ,dt=dtscalar,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=DiffEqSensitivity.ReverseDiffVJP()))
#
#   @test isapprox(res_sde_u0, res_sde_u02,  rtol=1e-4)
#   @test isapprox(res_sde_p, res_sde_p2,  atol=1e-4)
#
#   @show res_sde_u02, res_sde_p2
#
#   function compute_grads(sol, scale=1.0)
#     _sol = deepcopy(sol)
#     _sol.W.save_everystep = false
#     xdis = _sol(tarray)
#     helpu1 = [u[1] for u in xdis.u]
#     tmp1 = sum((@. xdis.t*helpu1*helpu1))
#
#     Wtmp = [_sol.W(t)[1][1] for t in tarray]
#     tmp2 = sum((@. (Wtmp-sol.prob.p[2]*t)*helpu1*helpu1))
#
#     tmp3 = sum((@. helpu1*helpu1))/helpu1[1]
#
#     return [tmp3, scale*tmp3], [tmp1*(1.0+scale^2), tmp2*(1.0+scale^2)]
#   end
#
#   true_grads = compute_grads(sol, u0[2]/u0[1])
#
#   @show  true_grads
#
#   @test isapprox(res_sde_u0, res_sde_u02,  rtol=1e-4)
#   @test isapprox(res_sde_p, res_sde_p2,  atol=1e-4)
#   @test isapprox(true_grads[2], res_sde_p', atol=1e-4)
#   @test isapprox(true_grads[1], res_sde_u0, rtol=1e-4)
#   @test isapprox(true_grads[2], res_sde_p2', atol=1e-4)
#   @test isapprox(true_grads[1], res_sde_u02, rtol=1e-4)
# end
#
# @testset "SDE oop scalar noise tests" begin
#   using DiffEqNoiseProcess
#
#   dtscalar = tend/1e2
#
#   f(u,p,t) = p[1]*u
#   σ(u,p,t) = p[2]*u
#
#   Random.seed!(seed)
#   W = WienerProcess(0.0,0.0,0.0)
#   u0 = rand(2)
#
#   linear_analytic(u0,p,t,W) = @.(u0*exp((p[1]-p[2]^2/2)*t+p[2]*W))
#
#   prob = SDEProblem(SDEFunction(f,σ,analytic=linear_analytic),σ,u0,trange,p2,
#     noise=W
#    )
#   sol = solve(prob,EulerHeun(), dt=dtscalar, save_noise=true)
#
#   @test isapprox(sol.u_analytic,sol.u, atol=1e-4)
#
#   res_sde_u0, res_sde_p = adjoint_sensitivities(sol,EM(),dg!,Array(t)
#     ,dt=dtscalar,adaptive=false,sensealg=BacksolveAdjoint())
#
#   @show res_sde_u0, res_sde_p
#
#   res_sde_u02, res_sde_p2 = adjoint_sensitivities(sol,EM(),dg!,Array(t)
#     ,dt=dtscalar,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=false))
#
#   @test isapprox(res_sde_u0, res_sde_u02,  rtol=1e-4)
#   @test isapprox(res_sde_p, res_sde_p2,  atol=1e-4)
#
#   res_sde_u02, res_sde_p2 = adjoint_sensitivities(sol,EM(),dg!,Array(t)
#     ,dt=dtscalar,adaptive=false,sensealg=BacksolveAdjoint(autojacvec=DiffEqSensitivity.ReverseDiffVJP()))
#
#   @test isapprox(res_sde_u0, res_sde_u02,  rtol=1e-4)
#   @test isapprox(res_sde_p, res_sde_p2,  atol=1e-4)
#
#   @show res_sde_u02, res_sde_p2
#
#   res_sde_u02, res_sde_p2 = adjoint_sensitivities(sol,EM(),dg!,Array(t)
#     ,dt=tend/1e2,adaptive=false,sensealg=InterpolatingAdjoint())
#
#
#   @test isapprox(res_sde_u0, res_sde_u02,  rtol=1e-4)
#   @test isapprox(res_sde_p, res_sde_p2,  atol=1e-4)
#
#   @show res_sde_u02, res_sde_p2
#
#   res_sde_u02, res_sde_p2 = adjoint_sensitivities(sol,EM(),dg!,Array(t)
#     ,dt=dtscalar,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=false))
#
#   @test isapprox(res_sde_u0, res_sde_u02,  rtol=1e-4)
#   @test isapprox(res_sde_p, res_sde_p2,  atol=1e-4)
#
#   @show res_sde_u02, res_sde_p2
#
#   res_sde_u02, res_sde_p2 = adjoint_sensitivities(sol,EM(),dg!,Array(t)
#     ,dt=dtscalar,adaptive=false,sensealg=InterpolatingAdjoint(autojacvec=DiffEqSensitivity.ReverseDiffVJP()))
#
#   @test isapprox(res_sde_u0, res_sde_u02,  rtol=1e-4)
#   @test isapprox(res_sde_p, res_sde_p2,  atol=1e-4)
#
#   @show res_sde_u02, res_sde_p2
#
#   function compute_grads(sol, scale=1.0)
#     _sol = deepcopy(sol)
#     _sol.W.save_everystep = false
#     xdis = _sol(tarray)
#     helpu1 = [u[1] for u in xdis.u]
#     tmp1 = sum((@. xdis.t*helpu1*helpu1))
#
#     Wtmp = [_sol.W(t)[1][1] for t in tarray]
#     tmp2 = sum((@. (Wtmp-sol.prob.p[2]*t)*helpu1*helpu1))
#
#     tmp3 = sum((@. helpu1*helpu1))/helpu1[1]
#
#     return [tmp3, scale*tmp3], [tmp1*(1.0+scale^2), tmp2*(1.0+scale^2)]
#   end
#
#   true_grads = compute_grads(sol, u0[2]/u0[1])
#
#   @show  true_grads
#
#
#   @test isapprox(true_grads[2], res_sde_p', atol=1e-4)
#   @test isapprox(true_grads[1], res_sde_u0, rtol=1e-4)
#   @test isapprox(true_grads[2], res_sde_p2', atol=1e-4)
#   @test isapprox(true_grads[1], res_sde_u02, rtol=1e-4)
#
# end
