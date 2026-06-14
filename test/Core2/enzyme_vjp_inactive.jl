using Test, SciMLSensitivity, Enzyme, Reactant

# Test that VJP choice types are treated as inactive by Enzyme
# This addresses issue #1225 where sensealg in ODEProblem constructor would fail
# because Enzyme tried to differentiate through VJP choice objects

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
        @test_nowarn Enzyme.autodiff(
            Enzyme.Reverse, test_func, Enzyme.Active, Enzyme.Duplicated(x, dx)
        )
        @test dx ≈ [2.0, 2.0]
    end

    # Test 2: Test different VJP choice types are inactive
    @testset "Different VJP types inactive" begin
        vjp_types = [EnzymeVJP(), ZygoteVJP(), ReverseDiffVJP(), TrackerVJP(), ReactantVJP()]

        for vjp in vjp_types
            function test_func(x)
                data = (value = x[1] * x[2], vjp = vjp)
                return data.value + 1.0
            end

            x = [2.0, 3.0]
            dx = Enzyme.make_zero(x)

            @test_nowarn Enzyme.autodiff(
                Enzyme.Reverse, test_func, Enzyme.Active, Enzyme.Duplicated(x, dx)
            )
        end
    end

    # Test 3: Sensitivity algorithms with VJP choices can be stored in tuples
    @testset "Sensitivity algorithm with VJP in data structure" begin
        sensealg = BacksolveAdjoint(autojacvec = EnzymeVJP())

        function test_func(x)
            data = (value = x[1] * x[2], alg = sensealg)
            return data.value
        end

        x = [3.0, 4.0]
        dx = Enzyme.make_zero(x)

        @test_nowarn Enzyme.autodiff(
            Enzyme.Reverse, test_func, Enzyme.Active, Enzyme.Duplicated(x, dx)
        )
        @test dx ≈ [4.0, 3.0]
    end
end
