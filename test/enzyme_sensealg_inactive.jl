using Test, SciMLSensitivity, Enzyme, OrdinaryDiffEq

# Test that VJP choice types are treated as inactive by Enzyme  
# The AbstractSensitivityAlgorithm inactive rule is now in SciMLBase
# This addresses issue #1225 where sensealg in ODEProblem constructor would fail

@testset "Enzyme VJP Choice Inactive Types" begin
    
    # Test 1: Basic test that sensealg objects can be stored in data structures during Enzyme differentiation
    @testset "Sensealg in data structures" begin
        sensealg = BacksolveAdjoint(autojacvec=EnzymeVJP())
        
        function test_func(x)
            # Store the sensealg in a data structure (this would fail without inactive rules)
            data = (value=x[1] + x[2], alg=sensealg)
            return data.value * 2.0
        end
        
        x = [1.0, 2.0]
        dx = Enzyme.make_zero(x)
        
        # This should not throw an error
        @test_nowarn Enzyme.autodiff(Enzyme.Reverse, test_func, Enzyme.Active, Enzyme.Duplicated(x, dx))
        @test dx ≈ [2.0, 2.0]
    end
    
    # Test 2: Test different sensitivity algorithm types
    @testset "Different sensealg types" begin
        sensealgs = [
            BacksolveAdjoint(autojacvec=EnzymeVJP()),
            InterpolatingAdjoint(autojacvec=ZygoteVJP()),
            QuadratureAdjoint(autojacvec=ReverseDiffVJP()),
            ForwardSensitivity()
        ]
        
        for sensealg in sensealgs
            function test_func(x)
                # Store each sensealg type
                data = (value=x[1] * x[2], alg=sensealg)
                return data.value + 1.0
            end
            
            x = [2.0, 3.0]
            dx = Enzyme.make_zero(x)
            
            @test_nowarn Enzyme.autodiff(Enzyme.Reverse, test_func, Enzyme.Active, Enzyme.Duplicated(x, dx))
        end
    end
    
    # Test 3: Test VJP choice types are also inactive
    @testset "VJP types inactive" begin
        vjp_types = [EnzymeVJP(), ZygoteVJP(), ReverseDiffVJP(), TrackerVJP()]
        
        for vjp in vjp_types
            function test_func(x)
                data = (value=x[1]^2, vjp=vjp)
                return data.value
            end
            
            x = [3.0]
            dx = Enzyme.make_zero(x)
            
            @test_nowarn Enzyme.autodiff(Enzyme.Reverse, test_func, Enzyme.Active, Enzyme.Duplicated(x, dx))
            @test dx ≈ [6.0]  # derivative of x^2 at x=3 is 2*3=6
        end
    end
    
    # Test 4: Integration test with actual ODEProblem (this was the original failing case)
    @testset "ODEProblem with sensealg in constructor" begin
        function f(du, u, p, t)
            du[1] = -p[1] * u[1]
            du[2] = p[2] * u[2]
        end
        
        function loss_func(p)
            u0 = [1.0, 2.0]
            # This used to fail - sensealg stored in problem kwargs
            prob = ODEProblem(f, u0, (0.0, 0.1), p, sensealg=BacksolveAdjoint(autojacvec=EnzymeVJP()))
            sol = solve(prob, Tsit5())
            return sol.u[end][1] + sol.u[end][2]
        end
        
        p = [0.5, 1.5]
        dp = Enzyme.make_zero(p)
        
        # This should not throw the "Error handling recursive stores for String" error
        @test_nowarn Enzyme.autodiff(Enzyme.Reverse, loss_func, Enzyme.Active, Enzyme.Duplicated(p, dp))
        
        # Verify the gradient is computed (non-zero and finite)
        @test all(isfinite, dp)
        @test any(x -> abs(x) > 1e-10, dp)  # At least one component should be non-trivial
    end
end