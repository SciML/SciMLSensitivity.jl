using DiffEqSensitivity, SafeTestsets
using Test, Pkg

const GROUP = get(ENV, "GROUP", "All")

function activate_gpu_env()
    Pkg.activate("gpu")
    Pkg.develop(PackageSpec(path=dirname(@__DIR__)))
    Pkg.instantiate()
end

@time begin

if GROUP == "All" || GROUP == "Core1" || GROUP == "Downstream"
    @time @safetestset "Forward Sensitivity" begin include("forward.jl") end
    @time @safetestset "Sparse Adjoint Sensitivity" begin include("sparse_adjoint.jl") end
    @time @safetestset "Second Order Sensitivity" begin include("second_order.jl") end
    @time @safetestset "Concrete Solve Derivatives" begin include("concrete_solve_derivatives.jl") end
    @time @safetestset "Branching Derivatives" begin include("branching_derivatives.jl") end
    @time @safetestset "Derivative Shapes" begin include("derivative_shapes.jl") end
    @time @safetestset "save_idxs" begin include("save_idxs.jl") end
    @time @safetestset "ArrayPartitions" begin include("array_partitions.jl") end
    @time @safetestset "Complex Adjoints" begin include("complex_adjoints.jl") end
    @time @safetestset "Forward Remake" begin include("forward_remake.jl") end
    @time @safetestset "Prob Kwargs" begin include("prob_kwargs.jl") end
    @time @safetestset "DiscreteProblem Adjoints" begin include("discrete.jl") end
    @time @safetestset "Time Type Mixing Adjoints" begin include("time_type_mixing.jl") end
end

if GROUP == "All" || GROUP == "Core2"
    @time @safetestset "hasbranching" begin include("hasbranching.jl") end
    @time @safetestset "Literal Adjoint" begin include("literal_adjoint.jl") end
    @time @safetestset "ForwardDiff Chunking Adjoints" begin include("forward_chunking.jl") end
    @time @safetestset "Stiff Adjoints" begin include("stiff_adjoints.jl") end
    @time @safetestset "Autodiff Events" begin include("autodiff_events.jl") end
    @time @safetestset "Null Parameters" begin include("null_parameters.jl") end
    @time @safetestset "Forward Mode Prob Kwargs" begin include("forward_prob_kwargs.jl") end
    @time @safetestset "Steady State Adjoint" begin include("steady_state.jl") end
    @time @safetestset "Concrete Solve Derivatives of Second Order ODEs" begin include("second_order_odes.jl") end
    @time @safetestset "Parameter Compatibility Errors" begin include("parameter_compatibility_errors.jl") end
end

if GROUP == "All" || GROUP == "Core3" || GROUP == "Downstream"
    @time @safetestset "Adjoint Sensitivity" begin include("adjoint.jl") end
    @time @safetestset "Continuous and discrete costs" begin include("mixed_costs.jl") end
end

if GROUP == "All" || GROUP == "Core4"
    @time @safetestset "Ensemble Tests" begin include("ensembles.jl") end
    @time @safetestset "GDP Regression Tests" begin include("gdp_regression_test.jl") end
    @time @safetestset "Layers Tests" begin include("layers.jl") end
    @time @safetestset "Layers SDE" begin include("layers_sde.jl") end
    @time @safetestset "Layers DDE" begin include("layers_dde.jl") end
    @time @safetestset "SDE - Neural" begin include("sde_neural.jl") end

    # No `@safetestset` since it requires running in Main
    @time @testset "Distributed" begin include("distributed.jl") end
end

if GROUP == "All" || GROUP == "Core5"
    @time @safetestset "Partial Neural Tests" begin include("partial_neural.jl") end
    @time @safetestset "Size Handling in Adjoint Tests" begin include("size_handling_adjoint.jl") end
    @time @safetestset "Callback - ReverseDiff" begin include("callback_reversediff.jl") end
    @time @safetestset "Alternative AD Frontend" begin include("alternative_ad_frontend.jl") end
    @time @safetestset "Hybrid DE" begin include("hybrid_de.jl") end
    @time @safetestset "HybridNODE" begin include("HybridNODE.jl") end
    @time @safetestset "ForwardDiff Sparsity Components" begin include("forwarddiffsensitivity_sparsity_components.jl") end
    @time @safetestset "Complex No u" begin include("complex_no_u.jl") end
end

if GROUP == "All" || GROUP == "SDE1"
    @time @safetestset "SDE Adjoint" begin include("sde_stratonovich.jl") end
    @time @safetestset "SDE Scalar Noise" begin include("sde_scalar_stratonovich.jl") end
    @time @safetestset "SDE Checkpointing" begin include("sde_checkpointing.jl") end
end

if GROUP == "All" || GROUP == "SDE2"
    @time @safetestset "SDE Non-Diagonal Noise" begin include("sde_nondiag_stratonovich.jl") end
end

if GROUP == "All" || GROUP == "SDE3"
    @time @safetestset "RODE Tests" begin include("rode.jl") end
    @time @safetestset "SDE Ito Conversion Tests" begin include("sde_transformation_test.jl") end
    @time @safetestset "SDE Ito Scalar Noise" begin include("sde_scalar_ito.jl") end
end

if GROUP == "Callbacks1"
    @time @safetestset "Discrete Callbacks with ForwardDiffSensitivity" begin include("callbacks/forward_sensitivity_callback.jl") end
    @time @safetestset "Discrete Callbacks with Adjoints" begin include("callbacks/discrete_callbacks.jl") end
    @time @safetestset "SDE Callbacks" begin include("callbacks/SDE_callbacks.jl") end
end

if GROUP == "Callbacks2"
    @time @safetestset "Continuous vs. discrete Callbacks" begin include("callbacks/continuous_vs_discrete.jl") end
    @time @safetestset "Continuous Callbacks with Adjoints" begin include("callbacks/continuous_callbacks.jl") end
    @time @safetestset "VectorContinuousCallbacks with Adjoints" begin include("callbacks/vector_continuous_callbacks.jl") end
end

if GROUP == "Shadowing"
    @time @safetestset "Shadowing Tests" begin include("shadowing.jl") end
end

if GROUP == "GPU"
    activate_gpu_env()
    @time @safetestset "Standard DiffEqFlux GPU" begin include("gpu/diffeqflux_standard_gpu.jl") end
    @time @safetestset "Mixed GPU/CPU" begin include("gpu/mixed_gpu_cpu_adjoint.jl") end
end
end
