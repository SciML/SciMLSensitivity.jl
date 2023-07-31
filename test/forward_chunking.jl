using SciMLSensitivity, OrdinaryDiffEq, Zygote, Test, ForwardDiff

function fiip(du, u, p, t)
    du[1] = dx = p[1] * u[1] - p[2] * p[51] * p[75] * u[1] * u[2]
    du[2] = dy = -p[3] * p[81] * p[25] * u[2] + (sum(@view(p[4:end])) / 100) * u[1] * u[2]
end
function foop(u, p, t)
    dx = p[1] * u[1] - p[2] * p[51] * p[75] * u[1] * u[2]
    dy = -p[3] * p[81] * p[25] * u[2] + (sum(@view(p[4:end])) / 100) * p[4] * u[1] * u[2]
    [dx, dy]
end

p = [1.5, 1.0, 3.0, 1.0];
u0 = [1.0; 1.0];
p = reshape(vcat(p, ones(100)), 4, 26)
prob = ODEProblem(fiip, u0, (0.0, 10.0), p)
proboop = ODEProblem(foop, u0, (0.0, 10.0), p)

loss = (u0, p) -> sum(solve(prob, Tsit5(), u0 = u0, p = p, abstol = 1e-14, reltol = 1e-14,
    saveat = 0.1, sensealg = ForwardDiffSensitivity()))
@time du01, dp1 = Zygote.gradient(loss, u0, p)

loss = (u0, p) -> sum(solve(prob, Tsit5(), u0 = u0, p = p, abstol = 1e-14, reltol = 1e-14,
    saveat = 0.1, sensealg = InterpolatingAdjoint()))
@time du02, dp2 = Zygote.gradient(loss, u0, p)

loss = (u0, p) -> sum(solve(prob, Tsit5(), u0 = u0, p = p, abstol = 1e-14, reltol = 1e-14,
    saveat = 0.1,
    sensealg = ForwardDiffSensitivity(chunk_size = 104)))
@time du03, dp3 = Zygote.gradient(loss, u0, p)

dp = ForwardDiff.gradient(p -> loss(u0, p), p)
du0 = ForwardDiff.gradient(u0 -> loss(u0, p), u0)

@test du01≈du0 rtol=1e-12
@test du01≈du02 rtol=1e-12
@test du01≈du03 rtol=1e-12
@test dp1≈dp rtol=1e-12
@test dp1≈dp2 rtol=1e-12
@test dp1≈dp3 rtol=1e-12

loss = (u0, p) -> sum(solve(proboop, Tsit5(), u0 = u0, p = p, abstol = 1e-14,
    reltol = 1e-14, saveat = 0.1,
    sensealg = ForwardDiffSensitivity()))
@time du01, dp1 = Zygote.gradient(loss, u0, p)

loss = (u0, p) -> sum(solve(proboop, Tsit5(), u0 = u0, p = p, abstol = 1e-14,
    reltol = 1e-14, saveat = 0.1,
    sensealg = InterpolatingAdjoint()))
@time du02, dp2 = Zygote.gradient(loss, u0, p)

loss = (u0, p) -> sum(solve(proboop, Tsit5(), u0 = u0, p = p, abstol = 1e-14,
    reltol = 1e-14, saveat = 0.1,
    sensealg = ForwardDiffSensitivity(chunk_size = 104)))
@time du03, dp3 = Zygote.gradient(loss, u0, p)

dp = ForwardDiff.gradient(p -> loss(u0, p), p)
du0 = ForwardDiff.gradient(u0 -> loss(u0, p), u0)

@test du01≈du0 rtol=1e-12
@test du01≈du02 rtol=1e-12
@test du01≈du03 rtol=1e-12
@test dp1≈dp rtol=1e-12
@test dp1≈dp2 rtol=1e-12
@test dp1≈dp3 rtol=1e-12

function fiip(du, u, p, t)
    du[1] = dx = p[1] * u[1] - p[2] * u[1] * u[2]
    du[2] = dy = -p[3] * u[2] + p[4] * u[1] * u[2]
    du[3:end] .= p[4]
end
function foop(u, p, t)
    dx = p[1] * u[1] - p[2] * u[1] * u[2]
    dy = -p[3] * u[2] + p[4] * u[1] * u[2]
    reshape(vcat(dx, dy, repeat([p[4]], 100)), 2, 51)
end

p = [1.5, 1.0, 3.0, 1.0];
u0 = [1.0; 1.0];
u0 = reshape(vcat(u0, ones(100)), 2, 51)
prob = ODEProblem(fiip, u0, (0.0, 10.0), p)
proboop = ODEProblem(foop, u0, (0.0, 10.0), p)

loss = (u0, p) -> sum(solve(prob, Tsit5(), u0 = u0, p = p, abstol = 1e-14, reltol = 1e-14,
    saveat = 0.1, sensealg = ForwardDiffSensitivity()))
@time du01, dp1 = Zygote.gradient(loss, u0, p)

loss = (u0, p) -> sum(solve(prob, Tsit5(), u0 = u0, p = p, abstol = 1e-14, reltol = 1e-14,
    saveat = 0.1, sensealg = InterpolatingAdjoint()))
@time du02, dp2 = Zygote.gradient(loss, u0, p)

loss = (u0, p) -> sum(solve(prob, Tsit5(), u0 = u0, p = p, abstol = 1e-14, reltol = 1e-14,
    saveat = 0.1,
    sensealg = ForwardDiffSensitivity(chunk_size = 102)))
@time du03, dp3 = Zygote.gradient(loss, u0, p)

dp = ForwardDiff.gradient(p -> loss(u0, p), p)
du0 = ForwardDiff.gradient(u0 -> loss(u0, p), u0)

@test du01≈du0 rtol=1e-12
@test du01≈du02 rtol=1e-12
@test du01≈du03 rtol=1e-12
@test dp1≈dp rtol=1e-12
@test dp1≈dp2 rtol=1e-12
@test dp1≈dp3 rtol=1e-12

loss = (u0, p) -> sum(solve(proboop, Tsit5(), u0 = u0, p = p, abstol = 1e-14,
    reltol = 1e-14, saveat = 0.1,
    sensealg = ForwardDiffSensitivity()))
@time du01, dp1 = Zygote.gradient(loss, u0, p)

loss = (u0, p) -> sum(solve(proboop, Tsit5(), u0 = u0, p = p, abstol = 1e-14,
    reltol = 1e-14, saveat = 0.1,
    sensealg = InterpolatingAdjoint()))
@time du02, dp2 = Zygote.gradient(loss, u0, p)

loss = (u0, p) -> sum(solve(proboop, Tsit5(), u0 = u0, p = p, abstol = 1e-14,
    reltol = 1e-14, saveat = 0.1,
    sensealg = ForwardDiffSensitivity(chunk_size = 102)))
@time du03, dp3 = Zygote.gradient(loss, u0, p)

dp = ForwardDiff.gradient(p -> loss(u0, p), p)
du0 = ForwardDiff.gradient(u0 -> loss(u0, p), u0)

@test du01≈du0 rtol=1e-12
@test du01≈du02 rtol=1e-12
@test du01≈du03 rtol=1e-12
@test dp1≈dp rtol=1e-12
@test dp1≈dp2 rtol=1e-12
@test dp1≈dp3 rtol=1e-12
