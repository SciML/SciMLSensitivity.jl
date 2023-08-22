using SciMLSensitivity, OrdinaryDiffEq, ForwardDiff, Calculus
using Test
function fb(du, u, p, t)
    du[1] = dx = p[1] * u[1] - p[2] * u[1] * u[2]
    du[2] = dy = -t * p[3] * u[2] + t * u[1] * u[2]
end
function jac(J, u, p, t)
    (x, y, a, b, c) = (u[1], u[2], p[1], p[2], p[3])
    J[1, 1] = a + y * b * -1
    J[2, 1] = t * y
    J[1, 2] = b * x * -1
    J[2, 2] = t * c * -1 + t * x
end
function paramjac(pJ, u, p, t)
    (x, y, a, b, c) = (u[1], u[2], p[1], p[2], p[3])
    pJ[1, 1] = x
    pJ[2, 1] = 0.0
    pJ[1, 2] = -x * y
    pJ[2, 2] = 0.0
    pJ[1, 3] = 0.0
    pJ[2, 3] = -t * y
end

f = ODEFunction(fb, jac = jac, paramjac = paramjac)
p = [1.5, 1.0, 3.0]
prob = ODEForwardSensitivityProblem(f, [1.0; 1.0], (0.0, 10.0), p)
probInpl = ODEForwardSensitivityProblem(fb, [1.0; 1.0], (0.0, 10.0), p)
probnoad = ODEForwardSensitivityProblem(fb, [1.0; 1.0], (0.0, 10.0), p,
    ForwardSensitivity(autodiff = false))
probnoadjacvec = ODEForwardSensitivityProblem(fb, [1.0; 1.0], (0.0, 10.0), p,
    ForwardSensitivity(autodiff = false,
        autojacvec = true))
probnoad2 = ODEForwardSensitivityProblem(f, [1.0; 1.0], (0.0, 10.0), p,
    ForwardSensitivity(autodiff = false))
probvecmat = ODEForwardSensitivityProblem(fb, [1.0; 1.0], (0.0, 10.0), p,
    ForwardSensitivity(autojacvec = false,
        autojacmat = true))
sol = solve(prob, Tsit5(), abstol = 1e-14, reltol = 1e-14)
@test_broken solve(probInpl, KenCarp4(), abstol = 1e-14, reltol = 1e-14).retcode == :Success
solInpl = solve(probInpl, KenCarp4(autodiff = false), abstol = 1e-14, reltol = 1e-14)
solInpl2 = solve(probInpl, Rodas4(autodiff = false), abstol = 1e-10, reltol = 1e-10)
solnoad = solve(probnoad, KenCarp4(autodiff = false), abstol = 1e-14, reltol = 1e-14)
solnoadjacvec = solve(probnoadjacvec, KenCarp4(autodiff = false), abstol = 1e-14,
    reltol = 1e-14)
solnoad2 = solve(probnoad, KenCarp4(autodiff = false), abstol = 1e-14, reltol = 1e-14)
solvecmat = solve(probvecmat, Tsit5(), abstol = 1e-14, reltol = 1e-14)

x = sol[1:(sol.prob.f.numindvar), :]

@test sol(5.0) ≈ solnoad(5.0)
@test sol(5.0) ≈ solnoad2(5.0)
@test sol(5.0)≈solnoadjacvec(5.0) atol=1e-6 rtol=1e-6
@test sol(5.0) ≈ solInpl(5.0)
@test isapprox(solInpl(5.0), solInpl2(5.0), rtol = 1e-5)
@test sol(5.0) ≈ solvecmat(5.0)

# Get the sensitivities

da = sol[(sol.prob.f.numindvar + 1):(sol.prob.f.numindvar * 2), :]
db = sol[(sol.prob.f.numindvar * 2 + 1):(sol.prob.f.numindvar * 3), :]
dc = sol[(sol.prob.f.numindvar * 3 + 1):(sol.prob.f.numindvar * 4), :]

sense_res1 = [da[:, end] db[:, end] dc[:, end]]

prob = ODEForwardSensitivityProblem(f.f, [1.0; 1.0], (0.0, 10.0), p,
    ForwardSensitivity(autojacvec = true))
sol = solve(prob, Tsit5(), abstol = 1e-14, reltol = 1e-14, saveat = 0.01)
x = sol[1:(sol.prob.f.numindvar), :]

# Get the sensitivities

res = sol[1:(sol.prob.f.numindvar), :]
da = sol[(sol.prob.f.numindvar + 1):(sol.prob.f.numindvar * 2), :]
db = sol[(sol.prob.f.numindvar * 2 + 1):(sol.prob.f.numindvar * 3), :]
dc = sol[(sol.prob.f.numindvar * 3 + 1):(sol.prob.f.numindvar * 4), :]

sense_res2 = [da[:, end] db[:, end] dc[:, end]]

function test_f(p)
    prob = ODEProblem(f, eltype(p).([1.0, 1.0]), (0.0, 10.0), p)
    solve(prob, Tsit5(), abstol = 1e-14, reltol = 1e-14, save_everystep = false)[end]
end

p = [1.5, 1.0, 3.0]
fd_res = ForwardDiff.jacobian(test_f, p)
calc_res = Calculus.finite_difference_jacobian(test_f, p)

@test sense_res1 ≈ sense_res2 ≈ fd_res
@test sense_res1 ≈ sense_res2 ≈ calc_res

## Check extraction

xall, dpall = extract_local_sensitivities(sol)
@test xall == res
@test dpall[1] == da

_, dpall_matrix = extract_local_sensitivities(sol, Val(true))
@test mapreduce(x -> x[:, 2], hcat, dpall) == dpall_matrix[2]

x, dp = extract_local_sensitivities(sol, length(sol.t))
sense_res2 = reduce(hcat, dp)
@test sense_res1 == sense_res2

@test extract_local_sensitivities(sol, sol.t[3]) == extract_local_sensitivities(sol, 3)

tmp = similar(sol[1])
@test extract_local_sensitivities(tmp, sol, sol.t[3]) == extract_local_sensitivities(sol, 3)

# asmatrix=true
@test extract_local_sensitivities(sol, length(sol), true) == (x, sense_res2)
@test extract_local_sensitivities(sol, sol.t[end], true) == (x, sense_res2)
@test extract_local_sensitivities(tmp, sol, sol.t[end], true) == (x, sense_res2)

# Return type inferred
@inferred extract_local_sensitivities(sol, 1)
@inferred extract_local_sensitivities(sol, 1, Val(true))
@inferred extract_local_sensitivities(sol, sol.t[3])
@inferred extract_local_sensitivities(sol, sol.t[3], Val(true))
@inferred extract_local_sensitivities(tmp, sol, sol.t[3])
@inferred extract_local_sensitivities(tmp, sol, sol.t[3], Val(true))

### ForwardDiff version

prob = ODEForwardSensitivityProblem(f.f, [1.0; 1.0], (0.0, 10.0), p,
    ForwardDiffSensitivity())
sol = solve(prob, Tsit5(), abstol = 1e-14, reltol = 1e-14, saveat = 0.01)

xall, dpall = extract_local_sensitivities(sol)
@test xall ≈ res
@test dpall[1]≈da atol=1e-9

_, dpall_matrix = extract_local_sensitivities(sol, Val(true))
@test mapreduce(x -> x[:, 2], hcat, dpall) == dpall_matrix[2]

x, dp = extract_local_sensitivities(sol, length(sol.t))
sense_res2 = reduce(hcat, dp)
@test fd_res == sense_res2

@test extract_local_sensitivities(sol, sol.t[3]) == extract_local_sensitivities(sol, 3)

tmp = similar(sol[1])
@test extract_local_sensitivities(tmp, sol, sol.t[3]) == extract_local_sensitivities(sol, 3)

# asmatrix=true
@test extract_local_sensitivities(sol, length(sol), true) == (x, sense_res2)
@test extract_local_sensitivities(sol, sol.t[end], true) == (x, sense_res2)
@test extract_local_sensitivities(tmp, sol, sol.t[end], true) == (x, sense_res2)

# Return type inferred
@inferred extract_local_sensitivities(sol, 1)
@inferred extract_local_sensitivities(sol, 1, Val(true))
@inferred extract_local_sensitivities(sol, sol.t[3])
@inferred extract_local_sensitivities(sol, sol.t[3], Val(true))
@inferred extract_local_sensitivities(tmp, sol, sol.t[3])
@inferred extract_local_sensitivities(tmp, sol, sol.t[3], Val(true))

# Test mass matrix
function rober_MM(du, u, p, t)
    y₁, y₂, y₃ = u
    k₁, k₂, k₃ = p
    du[1] = -k₁ * y₁ + k₃ * y₂ * y₃
    du[2] = k₁ * y₁ - k₂ * y₂^2 - k₃ * y₂ * y₃
    du[3] = y₁ + y₂ + y₃ - 1
    nothing
end
function rober_no_MM(du, u, p, t)
    y₁, y₂, y₃ = u
    k₁, k₂, k₃ = p
    du[1] = -k₁ * y₁ + k₃ * y₂ * y₃
    du[2] = k₁ * y₁ - k₂ * y₂^2 - k₃ * y₂ * y₃
    du[3] = k₂ * y₂^2
    nothing
end

M = [1.0 0 0; 0 1.0 0; 0 0 0]
p = [0.04, 3e7, 1e4]
u0 = [1.0, 0.0, 0.0]
tspan = (0.0, 12.0)

f_MM = ODEFunction(rober_MM, mass_matrix = M)
f_no_MM = ODEFunction(rober_no_MM)

prob_MM_ForwardSensitivity = ODEForwardSensitivityProblem(f_MM, u0, tspan, p,
    ForwardSensitivity())
sol_MM_ForwardSensitivity = solve(prob_MM_ForwardSensitivity, Rodas4(autodiff = false),
    reltol = 1e-14, abstol = 1e-14)

prob_MM_ForwardDiffSensitivity = ODEForwardSensitivityProblem(f_MM, u0, tspan, p,
    ForwardDiffSensitivity())
sol_MM_ForwardDiffSensitivity = solve(prob_MM_ForwardDiffSensitivity,
    Rodas4(autodiff = false), reltol = 1e-14,
    abstol = 1e-14)

prob_no_MM = ODEForwardSensitivityProblem(f_no_MM, u0, tspan, p, ForwardSensitivity())
sol_no_MM = solve(prob_no_MM, Rodas4(autodiff = false), reltol = 1e-14, abstol = 1e-14)

sen_MM_ForwardSensitivity = extract_local_sensitivities(sol_MM_ForwardSensitivity, 10.0,
    true)
sen_MM_ForwardDiffSensitivity = extract_local_sensitivities(sol_MM_ForwardDiffSensitivity,
    10.0, true)
sen_no_MM = extract_local_sensitivities(sol_no_MM, 10.0, true)

@test sen_MM_ForwardSensitivity[2]≈sen_MM_ForwardDiffSensitivity[2] atol=1e-10 rtol=1e-10
@test sen_MM_ForwardSensitivity[2]≈sen_no_MM[2] atol=1e-10 rtol=1e-10
# Test Float32

function f32(du, u, p, t)
    du[1] = dx = p[1] * u[1] - p[2] * u[1] * u[2]
    du[2] = dy = -p[3] * u[2] + u[1] * u[2]
end
p = [1.5f0, 1.0f0, 3.0f0]
prob = ODEForwardSensitivityProblem(f32, [1.0f0; 1.0f0], (0.0f0, 10.0f0), p)
sol = solve(prob, Tsit5())

# Out Of Place Error
function lotka_volterra_oop(u, p, t)
    du = zeros(2)
    du[1] = p[1] * u[1] - p[2] * u[1] * u[2]
    du[2] = -p[3] * u[2] + p[4] * u[1] * u[2]
    return du
end

u0 = [1.0, 1.0]
p = [1.5, 1.0, 3.0, 1.0]
@test_throws SciMLSensitivity.ForwardSensitivityOutOfPlaceError ODEForwardSensitivityProblem(lotka_volterra_oop,
    u0,
    (0.0,
        10.0),
    p)

# Make sure original jac is actually called if it is passed and autodiff is fully off
jac_call_count = 0
function jac_with_count(J, u, p, t)
    global jac_call_count
    jac_call_count += 1
    (x, y, a, b, c) = (u[1], u[2], p[1], p[2], p[3])
    J[1, 1] = a + y * b * -1
    J[2, 1] = t * y
    J[1, 2] = b * x * -1
    J[2, 2] = t * c * -1 + t * x
end

f = ODEFunction(fb, jac = jac_with_count, paramjac = paramjac)
p = [1.5, 1.0, 3.0]
absolutely_no_ad_sensealg = ForwardSensitivity(autodiff = false,
    autojacvec = false,
    autojacmat = false)
prob = ODEForwardSensitivityProblem(f,
    [1.0; 1.0],
    (0.0, 10.0),
    p,
    absolutely_no_ad_sensealg)
@test SciMLSensitivity.has_original_jac(prob.f)
@assert jac_call_count == 0
solve(prob, Tsit5(), abstol = 1e-14, reltol = 1e-14)
@test jac_call_count > 0
