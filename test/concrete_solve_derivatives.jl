using SciMLSensitivity, OrdinaryDiffEq, StochasticDiffEq
using Test, ForwardDiff, Random
import Tracker, ReverseDiff, ChainRulesCore, Mooncake, Enzyme

Enzyme.API.typeWarning!(false)

#=
AD Backend Abstraction Layer

We loop over multiple AD backends for the outer-level differentiation.
Zygote is only tested on Julia <= 1.11 due to compatibility issues with Julia 1.12+.
=#

function gradient_reversediff(f, x)
    return ReverseDiff.gradient(f, x)
end

function gradient_tracker(f, x)
    return Tracker.gradient(f, x)[1]
end

function gradient_enzyme(f, x)
    return only(Enzyme.gradient(Enzyme.Reverse, f, x))
end

function gradient_mooncake(f, x)
    return Mooncake.value_and_gradient!!(Mooncake.build_rrule(f, x), f, x)[2][2]
end

function gradient_forwarddiff(f, x)
    return ForwardDiff.gradient(f, x)
end

# Build list of reverse-mode backends to test
# Each entry is (name, gradient_function)
# On Julia 1.12+, Enzyme and Mooncake have compatibility issues, so we only
# test them on Julia <= 1.11. Zygote is also only tested on Julia <= 1.11.
const REVERSE_BACKENDS = Tuple{String, Function}[
    ("ReverseDiff", gradient_reversediff),
    ("Tracker", gradient_tracker),
]

if VERSION <= v"1.11"
    # Include Enzyme, Mooncake, and Zygote on Julia <= 1.11
    using Zygote
    function gradient_zygote(f, x)
        return Zygote.gradient(f, x)[1]
    end
    push!(REVERSE_BACKENDS, ("Enzyme", gradient_enzyme))
    push!(REVERSE_BACKENDS, ("Mooncake", gradient_mooncake))
    push!(REVERSE_BACKENDS, ("Zygote", gradient_zygote))
else
    # On Julia 1.12+, import Zygote but don't use it for outer differentiation
    import Zygote
end

#=
Compatibility Matrix

Define which AD backend × sensealg combinations are expected to:
- :works - works correctly
- :broken - broken (use @test_broken)
- :throws_enzyme - throws EnzymeTrackedRealError
- :throws_mooncake - throws MooncakeTrackedRealError
- :skip - skip this combination entirely
=#

const BACKEND_SENSEALG_STATUS = Dict{Tuple{String, String}, Symbol}(
    # Enzyme can't differentiate through Tracker or ReverseDiff internals
    ("Enzyme", "ReverseDiffAdjoint") => :throws_enzyme,
    ("Enzyme", "TrackerAdjoint") => :throws_enzyme,
    # Mooncake can't differentiate through Tracker or ReverseDiff internals
    ("Mooncake", "ReverseDiffAdjoint") => :throws_mooncake,
    ("Mooncake", "TrackerAdjoint") => :throws_mooncake,
    # Zygote has issues with ZygoteAdjoint and MooncakeAdjoint
    ("Zygote", "ZygoteAdjoint") => :broken,
    ("Zygote", "MooncakeAdjoint") => :broken,
    # ReverseDiff/Tracker with EnzymeAdjoint causes segfaults
    ("ReverseDiff", "EnzymeAdjoint") => :skip,
    ("Tracker", "EnzymeAdjoint") => :skip,
    # ForwardSensitivity is broken when perturbing u0 (only p works)
    ("ReverseDiff", "ForwardSensitivity") => :broken,
    ("Tracker", "ForwardSensitivity") => :broken,
    ("Enzyme", "ForwardSensitivity") => :broken,
    ("Mooncake", "ForwardSensitivity") => :broken,
    ("Zygote", "ForwardSensitivity") => :skip,  # Returns nothing for u0
)

function get_status(backend_name::String, sensealg_name::String)
    return get(BACKEND_SENSEALG_STATUS, (backend_name, sensealg_name), :works)
end

#=
Test Helper Functions
=#

function run_gradient_test(
        grad_fn, loss_fn, x, ref_grad, backend_name, sensealg_name; rtol = 1.0e-10
    )
    status = get_status(backend_name, sensealg_name)

    return if status == :skip
        @test_skip false
    elseif status == :throws_enzyme
        @test_throws SciMLSensitivity.EnzymeTrackedRealError grad_fn(loss_fn, x)
    elseif status == :throws_mooncake
        @test_throws SciMLSensitivity.MooncakeTrackedRealError grad_fn(loss_fn, x)
    elseif status == :broken
        @test_broken grad_fn(loss_fn, x) ≈ ref_grad rtol = rtol
    else
        result = grad_fn(loss_fn, x)
        @test result ≈ ref_grad rtol = rtol
    end
end

#=
Problem Definitions
=#

function fiip(du, u, p, t)
    du[1] = dx = p[1] * u[1] - p[2] * u[1] * u[2]
    return du[2] = dy = -p[3] * u[2] + p[4] * u[1] * u[2]
end

function foop(u, p, t)
    dx = p[1] * u[1] - p[2] * u[1] * u[2]
    dy = -p[3] * u[2] + p[4] * u[1] * u[2]
    return [dx, dy]
end

#=
Basic Solve Tests
=#

p = [1.5, 1.0, 3.0, 1.0]
u0 = [1.0; 1.0]
prob = ODEProblem(fiip, u0, (0.0, 10.0), p)
proboop = ODEProblem(foop, u0, (0.0, 10.0), p)

sol = solve(prob, Tsit5(), abstol = 1.0e-14, reltol = 1.0e-14)
@test sol isa ODESolution
sumsol = sum(sol)
@test sum(solve(prob, Tsit5(), u0 = u0, p = p, abstol = 1.0e-14, reltol = 1.0e-14)) == sumsol
@test sum(
    solve(
        prob, Tsit5(), u0 = u0, p = p, abstol = 1.0e-14, reltol = 1.0e-14,
        sensealg = ForwardDiffSensitivity()
    )
) == sumsol
@test sum(
    solve(
        prob, Tsit5(), u0 = u0, p = p, abstol = 1.0e-14, reltol = 1.0e-14,
        sensealg = BacksolveAdjoint()
    )
) == sumsol

#=
Compute Reference Gradients using ForwardDiff
ForwardDiff is the most reliable reference for gradient computation.
=#

# Reference gradient for IIP problem with saveat
ref_loss_iip = u0p -> sum(
    solve(
        prob, Tsit5(), u0 = u0p[1:2], p = u0p[3:end],
        abstol = 1.0e-14, reltol = 1.0e-14, saveat = 0.1
    )
)
u0p_iip = vcat(u0, p)
ref_grad_iip = ForwardDiff.gradient(ref_loss_iip, u0p_iip)

# Reference gradient for OOP problem with saveat
ref_loss_oop = u0p -> sum(
    solve(
        proboop, Tsit5(), u0 = u0p[1:2], p = u0p[3:end],
        abstol = 1.0e-14, reltol = 1.0e-14, saveat = 0.1
    )
)
ref_grad_oop = ForwardDiff.gradient(ref_loss_oop, u0p_iip)

#=
Main AD Backend × Sensealg Matrix Tests
=#

@testset "IIP Adjoint Sensitivities - $backend_name" for (backend_name, grad_fn) in REVERSE_BACKENDS
    u0p = vcat(u0, p)

    @testset "sensealg = QuadratureAdjoint" begin
        loss = u0p -> sum(
            solve(
                prob, Tsit5(), u0 = u0p[1:2], p = u0p[3:end],
                abstol = 1.0e-14, reltol = 1.0e-14, saveat = 0.1,
                sensealg = QuadratureAdjoint()
            )
        )
        run_gradient_test(grad_fn, loss, u0p, ref_grad_iip, backend_name, "QuadratureAdjoint")
    end

    @testset "sensealg = InterpolatingAdjoint" begin
        loss = u0p -> sum(
            solve(
                prob, Tsit5(), u0 = u0p[1:2], p = u0p[3:end],
                abstol = 1.0e-14, reltol = 1.0e-14, saveat = 0.1,
                sensealg = InterpolatingAdjoint()
            )
        )
        run_gradient_test(grad_fn, loss, u0p, ref_grad_iip, backend_name, "InterpolatingAdjoint")
    end

    @testset "sensealg = BacksolveAdjoint" begin
        loss = u0p -> sum(
            solve(
                prob, Tsit5(), u0 = u0p[1:2], p = u0p[3:end],
                abstol = 1.0e-14, reltol = 1.0e-14, saveat = 0.1,
                sensealg = BacksolveAdjoint()
            )
        )
        run_gradient_test(grad_fn, loss, u0p, ref_grad_iip, backend_name, "BacksolveAdjoint")
    end

    @testset "sensealg = TrackerAdjoint" begin
        loss = u0p -> sum(
            solve(
                prob, Tsit5(), u0 = u0p[1:2], p = u0p[3:end],
                abstol = 1.0e-14, reltol = 1.0e-14, saveat = 0.1,
                sensealg = TrackerAdjoint()
            )
        )
        run_gradient_test(grad_fn, loss, u0p, ref_grad_iip, backend_name, "TrackerAdjoint")
    end

    @testset "sensealg = ReverseDiffAdjoint" begin
        loss = u0p -> sum(
            solve(
                prob, Tsit5(), u0 = u0p[1:2], p = u0p[3:end],
                abstol = 1.0e-14, reltol = 1.0e-14, saveat = 0.1,
                sensealg = ReverseDiffAdjoint()
            )
        )
        run_gradient_test(grad_fn, loss, u0p, ref_grad_iip, backend_name, "ReverseDiffAdjoint")
    end

    @testset "sensealg = EnzymeAdjoint" begin
        loss = u0p -> sum(
            solve(
                prob, Tsit5(), u0 = u0p[1:2], p = u0p[3:end],
                abstol = 1.0e-14, reltol = 1.0e-14, saveat = 0.1,
                sensealg = EnzymeAdjoint()
            )
        )
        run_gradient_test(grad_fn, loss, u0p, ref_grad_iip, backend_name, "EnzymeAdjoint")
    end

    @testset "sensealg = GaussAdjoint" begin
        loss = u0p -> sum(
            solve(
                prob, Tsit5(), u0 = u0p[1:2], p = u0p[3:end],
                abstol = 1.0e-14, reltol = 1.0e-14, saveat = 0.1,
                sensealg = GaussAdjoint()
            )
        )
        run_gradient_test(grad_fn, loss, u0p, ref_grad_iip, backend_name, "GaussAdjoint")
    end

    @testset "sensealg = GaussKronrodAdjoint" begin
        loss = u0p -> sum(
            solve(
                prob, Tsit5(), u0 = u0p[1:2], p = u0p[3:end],
                abstol = 1.0e-14, reltol = 1.0e-14, saveat = 0.1,
                sensealg = GaussKronrodAdjoint()
            )
        )
        run_gradient_test(grad_fn, loss, u0p, ref_grad_iip, backend_name, "GaussKronrodAdjoint")
    end

    @testset "sensealg = ForwardDiffSensitivity" begin
        loss = u0p -> sum(
            solve(
                prob, Tsit5(), u0 = u0p[1:2], p = u0p[3:end],
                abstol = 1.0e-14, reltol = 1.0e-14, saveat = 0.1,
                sensealg = ForwardDiffSensitivity()
            )
        )
        run_gradient_test(
            grad_fn, loss, u0p, ref_grad_iip, backend_name, "ForwardDiffSensitivity"
        )
    end
end

@testset "OOP Adjoint Sensitivities - $backend_name" for (backend_name, grad_fn) in REVERSE_BACKENDS
    u0p = vcat(u0, p)

    @testset "sensealg = QuadratureAdjoint" begin
        loss = u0p -> sum(
            solve(
                proboop, Tsit5(), u0 = u0p[1:2], p = u0p[3:end],
                abstol = 1.0e-14, reltol = 1.0e-14, saveat = 0.1,
                sensealg = QuadratureAdjoint()
            )
        )
        run_gradient_test(grad_fn, loss, u0p, ref_grad_oop, backend_name, "QuadratureAdjoint")
    end

    @testset "sensealg = InterpolatingAdjoint" begin
        loss = u0p -> sum(
            solve(
                proboop, Tsit5(), u0 = u0p[1:2], p = u0p[3:end],
                abstol = 1.0e-14, reltol = 1.0e-14, saveat = 0.1,
                sensealg = InterpolatingAdjoint()
            )
        )
        run_gradient_test(grad_fn, loss, u0p, ref_grad_oop, backend_name, "InterpolatingAdjoint")
    end

    @testset "sensealg = BacksolveAdjoint" begin
        loss = u0p -> sum(
            solve(
                proboop, Tsit5(), u0 = u0p[1:2], p = u0p[3:end],
                abstol = 1.0e-14, reltol = 1.0e-14, saveat = 0.1,
                sensealg = BacksolveAdjoint()
            )
        )
        run_gradient_test(grad_fn, loss, u0p, ref_grad_oop, backend_name, "BacksolveAdjoint")
    end

    @testset "sensealg = TrackerAdjoint" begin
        loss = u0p -> sum(
            solve(
                proboop, Tsit5(), u0 = u0p[1:2], p = u0p[3:end],
                abstol = 1.0e-14, reltol = 1.0e-14, saveat = 0.1,
                sensealg = TrackerAdjoint()
            )
        )
        run_gradient_test(grad_fn, loss, u0p, ref_grad_oop, backend_name, "TrackerAdjoint")
    end

    @testset "sensealg = ReverseDiffAdjoint" begin
        loss = u0p -> sum(
            solve(
                proboop, Tsit5(), u0 = u0p[1:2], p = u0p[3:end],
                abstol = 1.0e-14, reltol = 1.0e-14, saveat = 0.1,
                sensealg = ReverseDiffAdjoint()
            )
        )
        run_gradient_test(grad_fn, loss, u0p, ref_grad_oop, backend_name, "ReverseDiffAdjoint")
    end

    @testset "sensealg = EnzymeAdjoint" begin
        loss = u0p -> sum(
            solve(
                proboop, Tsit5(), u0 = u0p[1:2], p = u0p[3:end],
                abstol = 1.0e-14, reltol = 1.0e-14, saveat = 0.1,
                sensealg = EnzymeAdjoint()
            )
        )
        run_gradient_test(grad_fn, loss, u0p, ref_grad_oop, backend_name, "EnzymeAdjoint")
    end

    @testset "sensealg = GaussAdjoint" begin
        loss = u0p -> sum(
            solve(
                proboop, Tsit5(), u0 = u0p[1:2], p = u0p[3:end],
                abstol = 1.0e-14, reltol = 1.0e-14, saveat = 0.1,
                sensealg = GaussAdjoint()
            )
        )
        run_gradient_test(grad_fn, loss, u0p, ref_grad_oop, backend_name, "GaussAdjoint")
    end

    @testset "sensealg = GaussKronrodAdjoint" begin
        loss = u0p -> sum(
            solve(
                proboop, Tsit5(), u0 = u0p[1:2], p = u0p[3:end],
                abstol = 1.0e-14, reltol = 1.0e-14, saveat = 0.1,
                sensealg = GaussKronrodAdjoint()
            )
        )
        run_gradient_test(grad_fn, loss, u0p, ref_grad_oop, backend_name, "GaussKronrodAdjoint")
    end

    @testset "sensealg = ForwardDiffSensitivity" begin
        loss = u0p -> sum(
            solve(
                proboop, Tsit5(), u0 = u0p[1:2], p = u0p[3:end],
                abstol = 1.0e-14, reltol = 1.0e-14, saveat = 0.1,
                sensealg = ForwardDiffSensitivity()
            )
        )
        run_gradient_test(
            grad_fn, loss, u0p, ref_grad_oop, backend_name, "ForwardDiffSensitivity"
        )
    end
end

#=
Struct-Based Loss Tests (merged from alternative_ad_frontend.jl)
Tests callable structs with different AD backends
=#

@testset "Struct-Based Loss Functions" begin
    odef(du, u, p, t) = du .= u .* p
    prob_struct = ODEProblem(odef, [2.0], (0.0, 1.0), [3.0])

    struct senseloss0{T}
        sense::T
    end
    function (f::senseloss0)(u0p)
        prob = ODEProblem{true}(odef, u0p[1:1], (0.0, 1.0), u0p[2:2])
        return sum(solve(prob, Tsit5(), abstol = 1.0e-12, reltol = 1.0e-12, saveat = 0.1))
    end

    u0p = [2.0, 3.0]
    du0p = zeros(2)

    # Reference gradient
    ref_grad_struct = ForwardDiff.gradient(senseloss0(InterpolatingAdjoint()), u0p)

    @testset "senseloss0 with $backend_name" for (backend_name, grad_fn) in REVERSE_BACKENDS
        if backend_name == "Enzyme"
            # Enzyme uses autodiff interface
            Enzyme.autodiff(
                Enzyme.Reverse, senseloss0(InterpolatingAdjoint()),
                Enzyme.Active, Enzyme.Duplicated(u0p, du0p)
            )
            @test du0p ≈ ref_grad_struct
        else
            result = grad_fn(senseloss0(InterpolatingAdjoint()), u0p)
            @test result ≈ ref_grad_struct
        end
    end

    struct senseloss{T}
        sense::T
    end
    function (f::senseloss)(u0p)
        return sum(
            solve(
                prob_struct, Tsit5(), u0 = u0p[1:1], p = u0p[2:2], abstol = 1.0e-12,
                reltol = 1.0e-12, saveat = 0.1, sensealg = f.sense
            )
        )
    end

    u0p = [2.0, 3.0]
    ref_grad_senseloss = ForwardDiff.gradient(senseloss(InterpolatingAdjoint()), u0p)

    @testset "senseloss with various sensealgs - $backend_name" for (backend_name, grad_fn) in REVERSE_BACKENDS
        @testset "InterpolatingAdjoint" begin
            result = grad_fn(senseloss(InterpolatingAdjoint()), u0p)
            @test result ≈ ref_grad_senseloss
        end

        @testset "ReverseDiffAdjoint" begin
            status = get_status(backend_name, "ReverseDiffAdjoint")
            if status == :throws_enzyme
                @test_throws SciMLSensitivity.EnzymeTrackedRealError grad_fn(senseloss(ReverseDiffAdjoint()), u0p)
            elseif status == :throws_mooncake
                @test_throws SciMLSensitivity.MooncakeTrackedRealError grad_fn(senseloss(ReverseDiffAdjoint()), u0p)
            else
                result = grad_fn(senseloss(ReverseDiffAdjoint()), u0p)
                @test result ≈ ref_grad_senseloss
            end
        end

        @testset "TrackerAdjoint" begin
            status = get_status(backend_name, "TrackerAdjoint")
            if status == :throws_enzyme
                @test_throws SciMLSensitivity.EnzymeTrackedRealError grad_fn(senseloss(TrackerAdjoint()), u0p)
            elseif status == :throws_mooncake
                @test_throws SciMLSensitivity.MooncakeTrackedRealError grad_fn(senseloss(TrackerAdjoint()), u0p)
            else
                result = grad_fn(senseloss(TrackerAdjoint()), u0p)
                @test result ≈ ref_grad_senseloss
            end
        end

        @testset "ForwardDiffSensitivity" begin
            result = grad_fn(senseloss(ForwardDiffSensitivity()), u0p)
            @test result ≈ ref_grad_senseloss
        end
    end

    # Test with p-only differentiation (senseloss3 and senseloss4 from alternative_ad_frontend.jl)
    struct senseloss_p{T}
        sense::T
    end
    function (f::senseloss_p)(p)
        return sum(
            solve(
                prob_struct, Tsit5(), p = p, abstol = 1.0e-12,
                reltol = 1.0e-12, saveat = 0.1, sensealg = f.sense
            )
        )
    end

    p_only = [3.0]
    ref_grad_p = ForwardDiff.gradient(senseloss_p(InterpolatingAdjoint()), p_only)

    @testset "senseloss p-only - $backend_name" for (backend_name, grad_fn) in REVERSE_BACKENDS
        @testset "InterpolatingAdjoint" begin
            result = grad_fn(senseloss_p(InterpolatingAdjoint()), p_only)
            @test result ≈ ref_grad_p
        end

        @testset "ForwardSensitivity" begin
            # ForwardSensitivity works when only differentiating p (not u0)
            result = grad_fn(senseloss_p(ForwardSensitivity()), p_only)
            @test result ≈ ref_grad_p
        end
    end
end

#=
Additional Tests: save_idxs, save_everystep, etc.
=#

@testset "save_idxs Tests" begin
    ref_loss_idx = u0p -> sum(
        Array(
            solve(
                prob, Tsit5(), u0 = u0p[1:2], p = u0p[3:end],
                abstol = 1.0e-14, reltol = 1.0e-14, saveat = 0.1,
                sensealg = InterpolatingAdjoint()
            )
        )[1, :]
    )
    u0p = vcat(u0, p)
    ref_grad_idx = ForwardDiff.gradient(ref_loss_idx, u0p)

    @testset "save_idxs - $backend_name" for (backend_name, grad_fn) in REVERSE_BACKENDS
        # Skip Tracker on Julia 1.12+ due to compatibility issues
        if backend_name == "Tracker" && VERSION >= v"1.12"
            @test_broken false
            continue
        end
        loss = u0p -> sum(
            solve(
                prob, Tsit5(), u0 = u0p[1:2], p = u0p[3:end],
                abstol = 1.0e-14, reltol = 1.0e-14, saveat = 0.1, save_idxs = 1:1,
                sensealg = InterpolatingAdjoint()
            )
        )
        result = grad_fn(loss, u0p)
        @test result ≈ ref_grad_idx rtol = 1.0e-12
    end
end

@testset "save_everystep=false Tests" begin
    ref_loss_end = u0p -> sum(
        solve(
            prob, Tsit5(), u0 = u0p[1:2], p = u0p[3:end],
            abstol = 1.0e-14, reltol = 1.0e-14,
            save_everystep = false, save_start = false,
            sensealg = InterpolatingAdjoint()
        )
    )
    u0p = vcat(u0, p)
    ref_grad_end = ForwardDiff.gradient(ref_loss_end, u0p)

    @testset "save_end only - $backend_name" for (backend_name, grad_fn) in REVERSE_BACKENDS
        # Skip Tracker on Julia 1.12+ due to compatibility issues
        if backend_name == "Tracker" && VERSION >= v"1.12"
            @test_broken false
            continue
        end
        loss = u0p -> sum(
            solve(
                prob, Tsit5(), u0 = u0p[1:2], p = u0p[3:end],
                abstol = 1.0e-14, reltol = 1.0e-14,
                save_everystep = false, save_start = false,
                sensealg = InterpolatingAdjoint()
            )
        )
        result = grad_fn(loss, u0p)
        @test result ≈ ref_grad_end rtol = 1.0e-11
    end
end

@testset "Non-integer saveat Tests" begin
    # tspan[2]-tspan[1] not a multiple of saveat
    ref_loss_saveat = u0p -> sum(
        solve(
            proboop, Tsit5(), u0 = u0p[1:2], p = u0p[3:end],
            abstol = 1.0e-14, reltol = 1.0e-14, saveat = 2.3,
            sensealg = ReverseDiffAdjoint()
        )
    )
    u0p = vcat(u0, p)
    ref_grad_saveat = ForwardDiff.gradient(ref_loss_saveat, u0p)

    @testset "saveat=2.3 - $backend_name" for (backend_name, grad_fn) in REVERSE_BACKENDS
        # Skip Tracker on Julia 1.12+ due to compatibility issues
        if backend_name == "Tracker" && VERSION >= v"1.12"
            @test_broken false
            continue
        end
        loss = u0p -> sum(
            solve(
                proboop, Tsit5(), u0 = u0p[1:2], p = u0p[3:end],
                abstol = 1.0e-14, reltol = 1.0e-14, saveat = 2.3,
                sensealg = InterpolatingAdjoint()
            )
        )
        result = grad_fn(loss, u0p)
        @test result ≈ ref_grad_saveat rtol = 1.0e-12
    end
end

@testset "VecOfArray Derivatives" begin
    ref_loss_vec = p -> sum(
        last(
            solve(
                prob, Tsit5(), p = p, saveat = 10.0,
                abstol = 1.0e-14, reltol = 1.0e-14
            )
        )
    )
    ref_grad_vec = ForwardDiff.gradient(ref_loss_vec, p)

    @testset "VecOfArray - $backend_name" for (backend_name, grad_fn) in REVERSE_BACKENDS
        # Skip Tracker on Julia 1.12+ due to compatibility issues
        if backend_name == "Tracker" && VERSION >= v"1.12"
            @test_broken false
            continue
        end
        # Skip ReverseDiff on Julia 1.12+ due to VecOfArray compatibility issues
        if backend_name == "ReverseDiff" && VERSION >= v"1.12"
            @test_broken false
            continue
        end
        result = grad_fn(ref_loss_vec, p)
        @test result ≈ ref_grad_vec
    end
end

#=
Matrix Multiplication ODE (from alternative_ad_frontend.jl)
=#

@testset "Matrix Multiplication ODE" begin
    solvealg_test = Tsit5()
    sensealg_test = InterpolatingAdjoint()
    tspan = (0.0, 1.0)
    u0_mat = rand(4, 8)
    p0 = rand(16)
    f_aug(u, p, t) = reshape(p, 4, 4) * u

    function loss_mat(p)
        prob = ODEProblem(f_aug, u0_mat, tspan, p; alg = solvealg_test, sensealg = sensealg_test)
        sol = solve(prob)
        return sum(sol[:, :, end])
    end

    function loss_mat2(p)
        prob = ODEProblem(f_aug, u0_mat, tspan, p)
        sol = solve(prob, solvealg_test; sensealg = sensealg_test)
        return sum(sol[:, :, end])
    end

    res1 = loss_mat(p0)
    res3 = loss_mat2(p0)
    @test res1 ≈ res3 atol = 1.0e-14

    @testset "Matrix ODE - $backend_name" for (backend_name, grad_fn) in REVERSE_BACKENDS
        if backend_name in ["Enzyme"]
            # Enzyme has issues with matrix ODEs
            @test_broken grad_fn(loss_mat, p0) ≈ ForwardDiff.gradient(loss_mat, p0)
        elseif backend_name == "Tracker" && VERSION >= v"1.12"
            # Tracker has issues on Julia 1.12+
            @test_broken false
            @test_broken false
        elseif backend_name == "ReverseDiff" && VERSION >= v"1.12"
            # ReverseDiff has issues with matrix ODEs on Julia 1.12+
            @test_broken false
            @test_broken false
        else
            res2 = grad_fn(loss_mat, p0)
            res4 = grad_fn(loss_mat2, p0)
            @test res2 ≈ res4 atol = 1.0e-14
            @test res2 ≈ ForwardDiff.gradient(loss_mat, p0) atol = 1.0e-10
        end
    end
end

#=
Recursion Test (from alternative_ad_frontend.jl)
https://discourse.julialang.org/t/diffeqsensitivity-jl-issues-with-reversediffadjoint-sensealg/88774
=#

@testset "Recursion Test" begin
    function ode_rec!(derivative, state, parameters, t)
        return derivative .= parameters
    end

    function solve_euler_rec(state, times, parameters)
        problem = ODEProblem{true}(
            ode_rec!, state, times[[1, end]], parameters; saveat = times,
            sensealg = ReverseDiffAdjoint()
        )
        return solve(problem, Euler(); dt = 1.0e-1)
    end

    initial_state = ones(2)
    solution_times = [1.0, 2.0]

    # This should not stack overflow
    result = ReverseDiff.gradient(
        p -> sum(sum(solve_euler_rec(initial_state, solution_times, p))), zeros(2)
    )
    @test length(result) == 2
end

#=
BouncingBall ODE Test (from alternative_ad_frontend.jl)
https://github.com/SciML/SciMLSensitivity.jl/issues/943
=#

@testset "BouncingBall ODE" begin
    using FiniteDiff

    GRAVITY = 9.81
    MASS = 1.0

    t_start = 0.0
    t_step = 0.05
    t_stop = 2.0
    tData = t_start:t_step:t_stop
    u0_ball = [1.0, 0.0]
    p_ball = [GRAVITY, MASS]

    function fx_ball(u, p, t)
        g, m = p
        return [u[2], -g]
    end

    ff_ball = ODEFunction{false}(fx_ball)
    prob_ball = ODEProblem{false}(ff_ball, u0_ball, (t_start, t_stop), p_ball)

    solver_ball = Rosenbrock23(autodiff = false)

    function loss_ball(p)
        solution = solve(
            prob_ball; p = p, alg = solver_ball, saveat = tData,
            sensealg = ReverseDiffAdjoint(), abstol = 1.0e-10, reltol = 1.0e-10
        )
        # Check if solution is an ODESolution with .u field vs a tracked/plain array
        if !isa(solution, ReverseDiff.TrackedArray) &&
                !isa(solution, Tracker.TrackedArray) &&
                !isa(solution, Array)
            return sum(abs.(collect(u[1] for u in solution.u)))
        else
            return sum(abs.(solution[1, :]))
        end
    end

    grad_fi = FiniteDiff.finite_difference_gradient(loss_ball, p_ball)
    grad_fd = ForwardDiff.gradient(loss_ball, p_ball)

    @test grad_fd ≈ grad_fi atol = 1.0e-2

    @testset "BouncingBall - $backend_name" for (backend_name, grad_fn) in REVERSE_BACKENDS
        if backend_name in ["Enzyme", "Mooncake"]
            # Complex solver interactions
            @test_broken grad_fn(loss_ball, p_ball) ≈ grad_fd atol = 1.0e-4
        elseif backend_name == "Tracker" && VERSION >= v"1.12"
            # Tracker has issues on Julia 1.12+
            @test_broken false
        else
            result = grad_fn(loss_ball, p_ball)
            @test result ≈ grad_fd atol = 1.0e-4
        end
    end
end

#=
SDE Tests
=#

@testset "SDE Gradients" begin
    function σiip(du, u, p, t)
        du[1] = p[5] * u[1]
        return du[2] = p[6] * u[2]
    end

    function σoop(u, p, t)
        dx = p[5] * u[1]
        dy = p[6] * u[2]
        return [dx, dy]
    end

    function σoop(u::Tracker.TrackedArray, p, t)
        dx = p[5] * u[1]
        dy = p[6] * u[2]
        return Tracker.collect([dx, dy])
    end

    p_sde = [1.5, 1.0, 3.0, 1.0, 0.1, 0.1]
    u0_sde = [1.0; 1.0]
    tarray = collect(0.0:0.01:1)
    seed = 100

    prob_sde = SDEProblem(fiip, σiip, u0_sde, (0.0, 1.0), p_sde)
    proboop_sde = SDEProblem(foop, σoop, u0_sde, (0.0, 1.0), p_sde)

    # Reference using adjoint_sensitivities
    _sol = solve(
        proboop_sde, EulerHeun(), dt = 1.0e-2, adaptive = false, save_noise = true,
        seed = seed
    )
    ū0_ref, adj_ref = adjoint_sensitivities(
        _sol, EulerHeun(), t = tarray,
        dgdu_discrete = ((out, u, p, t, i) -> out .= 1),
        sensealg = BacksolveAdjoint()
    )

    @testset "SDE OOP - $backend_name" for (backend_name, grad_fn) in REVERSE_BACKENDS
        u0p_sde = vcat(u0_sde, p_sde)
        loss = u0p -> sum(
            solve(
                proboop_sde, EulerHeun(),
                u0 = u0p[1:2], p = u0p[3:end], dt = 1.0e-2, saveat = 0.01,
                sensealg = BacksolveAdjoint(),
                seed = seed
            )
        )

        if backend_name in ["Enzyme"]
            # SDE + Enzyme has issues
            @test_broken grad_fn(loss, u0p_sde) !== nothing
        else
            result = grad_fn(loss, u0p_sde)
            @test isapprox(result[1:2], ū0_ref, rtol = 1.0e-4)
            @test isapprox(result[3:end], adj_ref', rtol = 1.0e-4)
        end
    end
end
