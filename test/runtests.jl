using SciMLSensitivity, SafeTestsets
using Test, Pkg

const GROUP = get(ENV, "GROUP", "All")

function activate_gpu_env()
    Pkg.activate("gpu")
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    Pkg.instantiate()
end

@time begin
    if GROUP == "All" || GROUP == "Core1" || GROUP == "Downstream"
        @time @safetestset "Forward Sensitivity" include("forward.jl")
        @time @safetestset "Sparse Adjoint Sensitivity" include("sparse_adjoint.jl")
        @time @safetestset "Adjoint Shapes" include("adjoint_shapes.jl")
        @time @safetestset "Second Order Sensitivity" include("second_order.jl")
        @time @safetestset "Concrete Solve Derivatives" include("concrete_solve_derivatives.jl")
        @time @safetestset "Branching Derivatives" include("branching_derivatives.jl")
        @time @safetestset "Derivative Shapes" include("derivative_shapes.jl")
        @time @safetestset "save_idxs" include("save_idxs.jl")
        @time @safetestset "ArrayPartitions" include("array_partitions.jl")
        @time @safetestset "Complex Adjoints" include("complex_adjoints.jl")
        @time @safetestset "ReverseDiffAdjoint Output Type" include("reversediff_output_types.jl")
        @time @safetestset "Forward Remake" include("forward_remake.jl")
        @time @safetestset "Prob Kwargs" include("prob_kwargs.jl")
        @time @safetestset "DiscreteProblem Adjoints" include("discrete.jl")
        @time @safetestset "Time Type Mixing Adjoints" include("time_type_mixing.jl")
    end

    if GROUP == "All" || GROUP == "Core2"
        @time @safetestset "Literal Adjoint" include("literal_adjoint.jl")
        @time @safetestset "ForwardDiff Chunking Adjoints" include("forward_chunking.jl")
        @time @safetestset "Stiff Adjoints" include("stiff_adjoints.jl")
        @time @safetestset "Scalar u0" include("scalar_u.jl")
        @time @safetestset "Error Messages" include("error_messages.jl")
        @time @safetestset "Autodiff Events" include("autodiff_events.jl")
    end

    if GROUP == "All" || GROUP == "Core3" || GROUP == "Downstream"
        @time @safetestset "Adjoint Sensitivity" include("adjoint.jl")

        @time @safetestset "Physical ODE Adjoint Regression Test" include("physical_ode_regression.jl")

        @time @safetestset "Continuous adjoint params" include("adjoint_param.jl")
        @time @safetestset "Continuous and discrete costs" include("mixed_costs.jl")
        @time @safetestset "Fully Out of Place adjoint sensitivity" include("adjoint_oop.jl")
        @time @safetestset "Differentiate LazyBuffer with ReverseDiff" include("lazybuffer.jl")
    end

    if GROUP == "All" || GROUP == "Core4"
        @time @safetestset "Ensemble Tests" include("ensembles.jl")
        @time @safetestset "GDP Regression Tests" include("gdp_regression_test.jl")
        @time @safetestset "Layers Tests" include("layers.jl")
        @time @safetestset "Layers SDE" include("layers_sde.jl")
        @time @safetestset "Layers DDE" include("layers_dde.jl")
        @time @safetestset "SDE - Neural" include("sde_neural.jl")
        # No `@safetestset` since it requires running in Main
        @time @testset "Distributed" include("distributed.jl")
    end

    if GROUP == "All" || GROUP == "Core5"
        @time @safetestset "Nested AD Regression Tests" include("nested_ad_regression.jl")
        @time @safetestset "Size Handling in Adjoint Tests" include("size_handling_adjoint.jl")
        @time @safetestset "Callback - ReverseDiff" include("callback_reversediff.jl")
        @time @safetestset "Alternative AD Frontend" include("alternative_ad_frontend.jl")
        @time @safetestset "Hybrid DE" include("hybrid_de.jl")
        @time @safetestset "HybridNODE" include("HybridNODE.jl")
        @time @safetestset "ForwardDiff Sparsity Components" include("forwarddiffsensitivity_sparsity_components.jl")
        @time @safetestset "Complex No u" include("complex_no_u.jl")
        @time @safetestset "Parameter Handling" include("parameter_handling.jl")
        @time @safetestset "Quality Assurance" include("aqua.jl")
    end

    if GROUP == "All" || GROUP == "Core6"
        @time @safetestset "Enzyme Closures" include("enzyme_closure.jl")
        @time @safetestset "Complex Matrix FiniteDiff Adjoint" include("complex_matrix_finitediff.jl")
        @time @safetestset "Null Parameters" include("null_parameters.jl")
        @time @safetestset "Forward Mode Prob Kwargs" include("forward_prob_kwargs.jl")
        @time @safetestset "Steady State Adjoint" include("steady_state.jl")
        @time @safetestset "Concrete Solve Derivatives of Second Order ODEs" include("second_order_odes.jl")
        @time @safetestset "Parameter Compatibility Errors" include("parameter_compatibility_errors.jl")
    end

    if GROUP == "All" || GROUP == "SDE1"
        @time @safetestset "SDE Adjoint" include("sde_stratonovich.jl")
        @time @safetestset "SDE Scalar Noise" include("sde_scalar_stratonovich.jl")
        @time @safetestset "SDE Checkpointing" include("sde_checkpointing.jl")
    end

    if GROUP == "All" || GROUP == "SDE2"
        @time @safetestset "SDE Non-Diagonal Noise" include("sde_nondiag_stratonovich.jl")
    end

    if GROUP == "All" || GROUP == "SDE3"
        @time @safetestset "RODE Tests" include("rode.jl")
        @time @safetestset "SDE Ito Conversion Tests" include("sde_transformation_test.jl")
        @time @safetestset "SDE Ito Scalar Noise" include("sde_scalar_ito.jl")
    end

    if GROUP == "Callbacks1"
        @time @safetestset "Discrete Callbacks with ForwardDiffSensitivity" include("callbacks/forward_sensitivity_callback.jl")
        @time @safetestset "Discrete Callbacks with Adjoints" include("callbacks/discrete_callbacks.jl")
        @time @safetestset "SDE Callbacks" include("callbacks/SDE_callbacks.jl")
        @time @safetestset "Non-tracked callbacks" include("callbacks/non_tracked_callbacks.jl")
    end

    if GROUP == "Callbacks2"
        @time @safetestset "Continuous vs. discrete Callbacks" include("callbacks/continuous_vs_discrete.jl")
        @time @safetestset "Continuous Callbacks with Adjoints" include("callbacks/continuous_callbacks.jl")
        @time @safetestset "VectorContinuousCallbacks with Adjoints" include("callbacks/vector_continuous_callbacks.jl")
    end

    if GROUP == "Shadowing"
        @time @safetestset "Shadowing Tests" include("shadowing.jl")
    end

    if GROUP == "GPU"
        activate_gpu_env()
        @time @safetestset "Standard DiffEqFlux GPU" include("gpu/diffeqflux_standard_gpu.jl")
        @time @safetestset "Mixed GPU/CPU" include("gpu/mixed_gpu_cpu_adjoint.jl")
    end
end
