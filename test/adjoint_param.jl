using Test, OrdinaryDiffEq, SciMLSensitivity, ForwardDiff, QuadGK, Zygote

abstol = 1e-12
reltol = 1e-12

function pendulum_eom(dx, x, p, t)
    dx[1] = p[1] * x[2]
    dx[2] = -sin(x[1]) + (-p[1] * sin(x[1]) + p[2] * x[2])  # Second term is a simple controller that stabilizes π
end

x0 = [0.1, 0.0]
tspan = (0.0, 10.0)
p = [1.0, -24.05, -19.137]
prob = ODEProblem(pendulum_eom, x0, tspan, p)
sol = solve(prob, Vern9(), abstol = abstol, reltol = reltol)

g(x, p, t) = 1.0 * (x[1] - π)^2 + 1.0 * x[2]^2 + 5.0 * (-p[1] * sin(x[1]) + p[2] * x[2])^2
dgdu(out, y, p, t) = ForwardDiff.gradient!(out, y -> g(y, p, t), y)
dgdp(out, y, p, t) = ForwardDiff.gradient!(out, p -> g(y, p, t), p)

res_interp = adjoint_sensitivities(sol, Vern9(), dgdu_continuous = dgdu,
    dgdp_continuous = dgdp, abstol = abstol,
    reltol = reltol, sensealg = InterpolatingAdjoint())
res_quad = adjoint_sensitivities(sol, Vern9(), dgdu_continuous = dgdu,
    dgdp_continuous = dgdp, abstol = abstol,
    reltol = reltol, sensealg = QuadratureAdjoint())
res_back = adjoint_sensitivities(sol, Vern9(), dgdu_continuous = dgdu,
    dgdp_continuous = dgdp, abstol = abstol,
    reltol = reltol, sensealg = BacksolveAdjoint(checkpointing = true))

function G(p)
    tmp_prob = remake(prob, p = p, u0 = convert.(eltype(p), prob.u0))
    sol = solve(tmp_prob, Vern9(), abstol = abstol, reltol = reltol)
    res, err = quadgk((t) -> g(sol(t), p, t), 0.0, 10.0, atol = 1e-8, rtol = 1e-8)
    res
end
res2 = ForwardDiff.gradient(G, p)

@test res_interp[2]'≈res2 atol=1e-5
@test res_quad[2]'≈res2 atol=1e-5
@test res_back[2]'≈res2 atol=1e-5

p = [2.0, 3.0]
u0 = [2.0]
function f(du, u, p, t)
    du[1] = -u[1] * p[1] - p[2]
end

prob = ODEProblem(f, u0, (0.0, 1.0), p)
sol = solve(prob, Tsit5(), abstol = abstol, reltol = reltol);

g(u, p, t) = -u[1] * p[1] - p[2]

dgdu(out, y, p, t) = ForwardDiff.gradient!(out, y -> g(y, p, t), y)
dgdp(out, y, p, t) = ForwardDiff.gradient!(out, p -> g(y, p, t), p)

du0, dp = adjoint_sensitivities(sol, Vern9(), dgdu_continuous = dgdu,
    dgdp_continuous = dgdp; abstol = abstol, reltol = reltol)

function G(p)
    tmp_prob = remake(prob, p = p, u0 = convert.(eltype(p), prob.u0))
    sol = solve(tmp_prob, Vern9(), abstol = abstol, reltol = reltol)
    res, err = quadgk((t) -> g(sol(t), p, t), 0.0, 1.0, atol = 1e-8, rtol = 1e-8)
    res
end
res2 = ForwardDiff.gradient(G, p)
@test dp'≈res2 atol=1e-5

function model(p, sensealg)
    N_oscillators = 30
    u0 = repeat([0.0; 1.0], 1, N_oscillators) # size(u0) = (2, 30)

    function du!(du, u, p, t)
        W, b = p # Parameters
        dy = @view du[1, :] # 30 elements
        dy′ = @view du[2, :]
        y = @view u[1, :]
        y′ = @view u[2, :]
        @. dy′ = -y * W
        @. dy = y′ * b
        return nothing
    end

    output = solve(ODEProblem(du!,
            u0,
            (0.0, 10.0),
            p),
        Tsit5(),
        saveat = collect(0:0.1:7),
        sensealg = sensealg,
        abstol = abstol, reltol = reltol)
    return Array(output[1, :, :]) # only return y, not y′
end

p = [1.5, 0.1]
y = model(p, QuadratureAdjoint())
function loss(p, sensealg = QuadratureAdjoint(autojacvec = ReverseDiffVJP(true)))
    sum(model(p, sensealg))
end
dp1 = Zygote.gradient(loss, p)[1]
dp2 = ForwardDiff.gradient(loss, p)
@test dp1 ≈ dp2
# @test_broken dp2 = Zygote.gradient(p -> loss(p, QuadratureAdjoint()), p)[1]
# @test dp1 ≈ dp2
