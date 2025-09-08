using Test, SciMLSensitivity, Enzyme, OrdinaryDiffEq

# Test that VJP choice types are treated as inactive by Enzyme  
# The AbstractSensitivityAlgorithm inactive rule is handled in SciMLBase
# This addresses issue #1225 where sensealg in ODEProblem constructor would fail

@testset "Enzyme VJP Choice Inactive Types" begin

    # Test 1: Basic test that VJP objects can be stored in data structures during Enzyme differentiation
    @testset "VJP types in data structures" begin
        vjp = EnzymeVJP()

        function test_func(x)
            # Store the VJP in a data structure (this would fail without inactive rules)
            data = (value = x[1] + x[2], vjp = vjp)
            return data.value * 2.0
        end

        x = [1.0, 2.0]
        dx = Enzyme.make_zero(x)

        # This should not throw an error
        @test_nowarn Enzyme.autodiff(Enzyme.Reverse, test_func, Enzyme.Active, Enzyme.Duplicated(x, dx))
        @test dx ≈ [2.0, 2.0]
    end

    # Test 2: Test different VJP choice types are inactive
    @testset "Different VJP types inactive" begin
        vjp_types = [EnzymeVJP(), ZygoteVJP(), ReverseDiffVJP(), TrackerVJP()]

        for vjp in vjp_types
            function test_func(x)
                data = (value = x[1] * x[2], vjp = vjp)
                return data.value + 1.0
            end

            x = [2.0, 3.0]
            dx = Enzyme.make_zero(x)

            @test_nowarn Enzyme.autodiff(Enzyme.Reverse, test_func, Enzyme.Active, Enzyme.Duplicated(x, dx))
        end
    end

    # Test 3: Test sensitivity algorithms with VJP choices (integration test)
    # Note: This test also depends on SciMLBase having AbstractSensitivityAlgorithm as inactive
    @testset "Sensitivity algorithms with VJP choices" begin
        function f(du, u, p, t)
            du[1] = -p[1] * u[1]
            du[2] = p[2] * u[2]
        end

        function loss_func(p)
            u0 = [1.0, 2.0]
            # Both VJP choice and sensitivity algorithm should be inactive
            prob = ODEProblem(
                f, u0, (0.0, 0.1), p, sensealg = BacksolveAdjoint(autojacvec = EnzymeVJP()))
            sol = solve(prob, Tsit5())
            return sol.u[end][1] + sol.u[end][2]
        end

        p = [0.5, 1.5]
        dp = Enzyme.make_zero(p)

        # This should not throw the "Error handling recursive stores for String" error
        # This is the original failing case from issue #1225
        @test_nowarn Enzyme.autodiff(Enzyme.Reverse, loss_func, Enzyme.Active, Enzyme.Duplicated(p, dp))

        # Verify the gradient is computed (non-zero and finite)
        @test all(isfinite, dp)
        @test any(x -> abs(x) > 1e-10, dp)  # At least one component should be non-trivial
    end
end
