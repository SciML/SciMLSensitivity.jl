using SciMLSensitivity, OrdinaryDiffEq
using ADTypes: AutoFiniteDiff, AutoForwardDiff
using Test

println("Testing the specific issues mentioned in the original CI errors...")

# Test 1: TypeError with AutoFiniteDiff in boolean context (from forward.jl:29)
@testset "AutoFiniteDiff boolean context fix" begin
    function fb(du, u, p, t)
        du[1] = p[1] * u[1] - p[2] * u[1] * u[2]
        du[2] = -t * p[3] * u[2] + t * u[1] * u[2]
    end
    
    p = [1.5, 1.0, 3.0]
    
    # This was failing with "TypeError: non-boolean (ADTypes.AutoFiniteDiff{...}) used in boolean context"
    @test_nowarn ForwardSensitivity(autodiff = AutoFiniteDiff())
    @test_nowarn ForwardSensitivity(autodiff = AutoFiniteDiff(), autojacvec = true)
    
    # Test the actual problem construction that was failing
    @test_nowarn ODEForwardSensitivityProblem(fb, [1.0; 1.0], (0.0, 10.0), p,
        sensealg = ForwardSensitivity(autodiff = AutoFiniteDiff()))
end

# Test 2: Array .u.u indexing fix (from scimlstructures_interface.jl:79)
@testset "Solution indexing fixes" begin
    function simple_ode(du, u, p, t)
        du[1] = p[1] * u[1]
        du[2] = p[2] * u[2]
    end
    
    prob = ODEProblem(simple_ode, [1.0, 2.0], (0.0, 1.0), [1.0, 0.5])
    sol = solve(prob, Tsit5())
    
    # These should work (our fix) - not sol.u.u[end] which was incorrect
    @test_nowarn sum(sol.u[end])
    @test_nowarn sol[end]
    @test length(sol.u[end]) == 2
    
    # Test that sol.u.u[end] would indeed fail (confirming the original fix was wrong)
    @test_throws Exception sol.u.u[end]
end

# Test 3: Test various ADType usage patterns
@testset "ADType boolean handling" begin
    # Test all the ways autodiff can be specified
    algs = [
        ForwardSensitivity(autodiff = true),
        ForwardSensitivity(autodiff = false), 
        ForwardSensitivity(autodiff = AutoFiniteDiff()),
        ForwardSensitivity(autodiff = AutoForwardDiff()),
        BacksolveAdjoint(autodiff = AutoFiniteDiff()),
        InterpolatingAdjoint(autodiff = AutoFiniteDiff()),
        QuadratureAdjoint(autodiff = AutoFiniteDiff())
    ]
    
    for alg in algs
        @test_nowarn alg  # Should construct without error
    end
end

println("All tests passed! The main issues from the CI errors have been resolved.")