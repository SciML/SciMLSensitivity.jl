using NonlinearSolve, SciMLSensitivity
using SciMLBase

using Test, ForwardDiff, Zygote, Enzyme, Mooncake, LinearAlgebra

# Reverse-mode sensitivities of a NonlinearLeastSquaresProblem solve. The adjoint
# differentiates the (projected) stationarity equation, so components pinned at an
# active bound have zero parameter sensitivity. This is the reverse-mode
# counterpart of the ForwardDiff coverage in NonlinearSolve's
# bounds_sensitivity_tests.jl and compares against the same analytic
# expectations and the ForwardDiff result.

function mooncake_grad(f, p)
    cache = Mooncake.prepare_gradient_cache(f, p)
    return Mooncake.value_and_gradient!!(cache, f, p)[2][2]
end

function enzyme_grad(f, p)
    return Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), f, p)[1]
end

# Top-level residuals over `const` globals: problem construction stores into a
# captured closure field while specializing `f`, which trips Enzyme's
# read-only activity check.
const NLLS_COUPLING = [1.0 1.0; 0.0 1.0]
nlls_coupled_residual(u, p) = NLLS_COUPLING * u - p

const NLLS_OVERDETERMINED = [1.0 0.0; 0.0 1.0; 1.0 1.0]
nlls_overdetermined_residual(u, p) = NLLS_OVERDETERMINED * u - p

@testset "Bounded least-squares reverse-mode sensitivities" begin
    for inplace in (false, true), derivative in (:autodiff, :jac, :vjp)
        f(u, p) = u .- p
        f!(r, u, p) = (r .= u .- p; nothing)
        jac(u, p) = Matrix{eltype(u)}(I, length(u), length(u))
        jac!(J, u, p) = (J .= jac(u, p); nothing)
        vjp(v, u, p) = v
        vjp!(out, v, u, p) = (out .= v; nothing)
        options = derivative === :jac ? (; jac = inplace ? jac! : jac) :
            derivative === :vjp ? (; vjp = inplace ? vjp! : vjp) : (;)
        nf = NonlinearFunction{inplace}(inplace ? f! : f; options...)
        function solution(p)
            prob = NonlinearLeastSquaresProblem(
                nf, [0.5, 0.5, 0.5, 0.5], p;
                lb = [0.0, 0.0, 0.0, 0.5], ub = [1.0, 1.0, 1.0, 0.5]
            )
            return solve(prob, BoundedTrustRegion(); abstol = 1.0e-10, reltol = 1.0e-10).u
        end
        loss(p) = sum(abs2, solution(p))
        p = [-1.0, 2.0, 0.3, 2.0]
        # sum(abs2, u*) with u* = [0, 1, 0.3, 0.5]: only the interior component
        # u[3] = p[3] carries a nonzero sensitivity.
        expected = 2 .* [0.0, 0.0, 0.3, 0.0]
        @test solution(p) ≈ [0.0, 1.0, 0.3, 0.5] atol = 1.0e-8
        @test ForwardDiff.gradient(loss, p) ≈ expected atol = 1.0e-10
        @test only(Zygote.gradient(loss, p)) ≈ expected atol = 1.0e-10
        @test enzyme_grad(loss, p) ≈ expected atol = 1.0e-10
        @test mooncake_grad(loss, p) ≈ expected atol = 1.0e-10
    end

    for (lb, ub, p, expected) in (
            (nothing, 1.0, 2.0, 0.0), (0.0, nothing, -1.0, 0.0),
            (0.0, 1.0, 0.3, 1.0), (0.5, 0.5, 2.0, 0.0),
        )
        scalar_parameter(p) = only(
            solve(
                NonlinearLeastSquaresProblem(
                    (u, p) -> u .- p,
                    [0.5], p; lb, ub
                ), BoundedTrustRegion(); abstol = 1.0e-10, reltol = 1.0e-10
            ).u
        )
        @test ForwardDiff.derivative(scalar_parameter, p) ≈ expected atol = 1.0e-10
        @test only(Zygote.gradient(scalar_parameter, p)) ≈ expected atol = 1.0e-10
        @test mooncake_grad(scalar_parameter, p) ≈ expected atol = 1.0e-10
    end

    for derivative in (:autodiff, :jac, :vjp), (p, expected) in ((2.0, 0.0), (0.3, 1.0))
        options = derivative === :jac ? (; jac = (u, p) -> one(u)) :
            derivative === :vjp ? (; vjp = (v, u, p) -> v) : (;)
        nf = NonlinearFunction((u, p) -> u - p; options...)
        scalar_state(p) = solve(
            NonlinearLeastSquaresProblem(
                nf, 0.5, p; lb = 0.0,
                ub = 1.0
            ), BoundedTrustRegion(); abstol = 1.0e-10, reltol = 1.0e-10
        ).u
        @test ForwardDiff.derivative(scalar_state, p) ≈ expected atol = 1.0e-10
        @test only(Zygote.gradient(scalar_state, p)) ≈ expected atol = 1.0e-10
        @test mooncake_grad(scalar_state, p) ≈ expected atol = 1.0e-10
    end

    # Coupled free variable: u[1] pinned at its bound still feeds the stationarity
    # system of the free u[2].
    coupled(p) = solve(
        NonlinearLeastSquaresProblem(
            nlls_coupled_residual,
            [0.5, 0.5], p; lb = [0.0, -Inf], ub = [1.0, Inf]
        ), BoundedTrustRegion();
        abstol = 1.0e-10, reltol = 1.0e-10
    ).u
    coupled_loss(p) = sum(abs2, coupled(p))
    p = [3.0, 0.0]
    @test coupled(p) ≈ [1.0, 1.0] atol = 1.0e-8
    @test ForwardDiff.gradient(coupled_loss, p) ≈
        only(Zygote.gradient(coupled_loss, p)) atol = 1.0e-10
    @test only(Zygote.gradient(coupled_loss, p)) ≈
        enzyme_grad(coupled_loss, p) atol = 1.0e-10
    @test enzyme_grad(coupled_loss, p) ≈ mooncake_grad(coupled_loss, p) atol = 1.0e-10

    # Unbounded least-squares solves use the same stationarity adjoint: a linear
    # overdetermined problem has du*/dp = pinv(A).
    overdetermined(p) = solve(
        NonlinearLeastSquaresProblem(
            nlls_overdetermined_residual, [0.1, 0.2], p
        ), LevenbergMarquardt(); abstol = 1.0e-12, reltol = 1.0e-12
    ).u
    over_loss(p) = sum(abs2, overdetermined(p))
    p = [0.3, -0.4, 0.7]
    # du*/dp = pinv(A_over), so d/dp sum(u*.^2) = 2 * pinv(A_over)' * u*.
    expected_over = 2 .* (pinv(NLLS_OVERDETERMINED)' * overdetermined(p))
    @test overdetermined(p) ≈ pinv(NLLS_OVERDETERMINED) * p atol = 1.0e-8
    @test ForwardDiff.gradient(over_loss, p) ≈ expected_over atol = 1.0e-10
    @test only(Zygote.gradient(over_loss, p)) ≈ expected_over atol = 1.0e-10
    @test only(Zygote.gradient(over_loss, p)) ≈ enzyme_grad(over_loss, p) atol = 1.0e-10
    @test enzyme_grad(over_loss, p) ≈ mooncake_grad(over_loss, p) atol = 1.0e-10
end
