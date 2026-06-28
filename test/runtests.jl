using SciMLSensitivity, SafeTestsets
using Test
using SciMLTesting
import Mooncake

run_tests(;
    core = () -> nothing,
    groups = Dict(
        "Core1" => function ()
            return @testset "Core1" begin
                @time @safetestset "Forward Sensitivity" include("Core1/forward.jl")
                @time @safetestset "Sparse Adjoint Sensitivity" include("Core1/sparse_adjoint.jl")
                @time @safetestset "Adjoint Shapes" include("Core1/adjoint_shapes.jl")
                @time @safetestset "Second Order Sensitivity" include("Core1/second_order.jl")
                @time @safetestset "Concrete Solve Derivatives" include("Core1/concrete_solve_derivatives.jl")
                @time @safetestset "Branching Derivatives" include("Core1/branching_derivatives.jl")
                @time @safetestset "Derivative Shapes" include("Core1/derivative_shapes.jl")
                @time @safetestset "save_idxs" include("Core1/save_idxs.jl")
                @time @safetestset "ArrayPartitions" include("Core1/array_partitions.jl")
                @time @safetestset "Complex Adjoints" include("Core1/complex_adjoints.jl")
                @time @safetestset "ReverseDiffAdjoint Output Type" include("Core1/reversediff_output_types.jl")
                @time @safetestset "Forward Remake" include("Core1/forward_remake.jl")
                @time @safetestset "Prob Kwargs" include("Core1/prob_kwargs.jl")
                @time @safetestset "Mooncake VJP Prob Kwargs" include("Core1/mooncake_vjp_prob_kwargs.jl")
                @time @safetestset "DiscreteProblem Adjoints" include("Core1/discrete.jl")
                @time @safetestset "Time Type Mixing Adjoints" include("Core1/time_type_mixing.jl")
                @time @safetestset "SciMLStructures Interface" include("Core1/scimlstructures_interface.jl")
                @time @safetestset "Functor Parameters" include("Core1/functor_params.jl")
                @time @safetestset "Sensitivity Verbosity" include("Core1/sensitivity_verbosity.jl")
            end
        end,
        "Core2" => function ()
            return @testset "Core 2" begin
                @time @safetestset "Literal Adjoint" include("Core2/literal_adjoint.jl")
                @time @safetestset "ForwardDiff Chunking Adjoints" include("Core2/forward_chunking.jl")
                @time @safetestset "Stiff Adjoints" include("Core2/stiff_adjoints.jl")
                @time @safetestset "Scalar u0" include("Core2/scalar_u.jl")
                @time @safetestset "Error Messages" include("Core2/error_messages.jl")
                @time @safetestset "Autodiff Events" include("Core2/autodiff_events.jl")
                @time @safetestset "Enzyme VJP Inactive" include("Core2/enzyme_vjp_inactive.jl")
                @time @safetestset "Enzyme VJP View ComponentArray" include("Core2/enzyme_view_componentarray.jl")
            end
        end,
        "Core3" => function ()
            return @testset "Core 3" begin
                @time @safetestset "Default DiffEq Alg" include("Core3/default_alg_diff.jl")
                @time @safetestset "Adjoint Sensitivity" include("Core3/adjoint.jl")
                @time @safetestset "automatic sensealg choice" include("Core3/automatic_sensealg_choice.jl")
                @time @safetestset "GaussAdjoint ZygoteVJP In-Place" include("Core3/gauss_zygote_inplace.jl")
                @time @safetestset "User-provided VJP" include("Core3/user_vjp.jl")
            end
        end,
        "Core4" => function ()
            return @testset "Core 4" begin
                @time @safetestset "Ensemble Tests" include("Core4/ensembles.jl")
                @time @safetestset "GDP Regression Tests" include("Core4/gdp_regression_test.jl")
                @time @safetestset "Layers Tests" include("Core4/layers.jl")
                @time @safetestset "Layers SDE" include("Core4/layers_sde.jl")
                @time @safetestset "Layers DDE" include("Core4/layers_dde.jl")
                @time @safetestset "SDE - Neural" include("Core4/sde_neural.jl")
                # No `@safetestset` since it requires running in Main
                @time @testset "Distributed" include("Core4/distributed.jl")
            end
        end,
        "Core5" => function ()
            return @testset "Core 5" begin
                @time @safetestset "Nested AD Regression Tests" include("Core5/nested_ad_regression.jl")
                @time @safetestset "Size Handling in Adjoint Tests" include("Core5/size_handling_adjoint.jl")
                @time @safetestset "Callback - ReverseDiff" include("Core5/callback_reversediff.jl")
                @time @safetestset "Hybrid DE" include("Core5/hybrid_de.jl")
                @time @safetestset "HybridNODE" include("Core5/HybridNODE.jl")
                @time @safetestset "ForwardDiff Sparsity Components" include("Core5/forwarddiffsensitivity_sparsity_components.jl")
                @time @safetestset "Forward Sensitivity Sparse Jacobian" include("Core5/forward_sensitivity_sparse_jac.jl")
                @time @safetestset "Complex No u" include("Core5/complex_no_u.jl")
                @time @safetestset "Parameter Handling" include("Core5/parameter_handling.jl")
            end
        end,
        "Core6" => function ()
            return @testset "Core 6" begin
                @time @safetestset "Enzyme Closures" include("Core6/enzyme_closure.jl")
                @time @safetestset "Complex Matrix FiniteDiff Adjoint" include("Core6/complex_matrix_finitediff.jl")
                @time @safetestset "Null Parameters" include("Core6/null_parameters.jl")
                @time @safetestset "Forward Mode Prob Kwargs" include("Core6/forward_prob_kwargs.jl")
                @time @safetestset "Steady State Adjoint" include("Core6/steady_state.jl")
                @time @safetestset "Optimization Adjoint" include("Core6/optimization_adjoint.jl")
                @time @safetestset "Concrete Solve Derivatives of Second Order ODEs" include("Core6/second_order_odes.jl")
                @time @safetestset "Parameter Compatibility Errors" include("Core6/parameter_compatibility_errors.jl")
            end
        end,
        "Core7" => function ()
            return @testset "Core 7" begin
                @time @safetestset "Physical ODE Adjoint Regression Test" include("Core7/physical_ode_regression.jl")
                @time @safetestset "Continuous adjoint params" include("Core7/adjoint_param.jl")
                @time @safetestset "Continuous and discrete costs" include("Core7/mixed_costs.jl")
                @time @safetestset "Fully Out of Place adjoint sensitivity" include("Core7/adjoint_oop.jl")
                @time @safetestset "Differentiate LazyBuffer with ReverseDiff" include("Core7/lazybuffer.jl")
                # Core 7 was split off from Core 3 due to leaks in the testsets described here https://github.com/SciML/SciMLSensitivity.jl/pull/1024
            end
        end,
        "Core8" => function ()
            return @testset "Core 8" begin
                @time @safetestset "Adjoints through NonlinearProblem" include("Core8/parameter_initialization.jl")
                @time @safetestset "Initialization with MTK" include("Core8/desauty_dae_mwe.jl")
                @time @safetestset "MTK Forward Mode" include("Core8/mtk.jl")
                @time @safetestset "SCCNonlinearProblem" include("Core8/scc_nonlinearsolve.jl")
                @time @safetestset "EnzymeVJP Repeated Adjoint" include("Core8/enzyme_vjp_repeated_adjoint.jl")
            end
        end,
        "SDE1" => function ()
            return @testset "SDE 1" begin
                @time @safetestset "SDE Adjoint" include("SDE1/sde_stratonovich.jl")
                @time @safetestset "SDE Scalar Noise" include("SDE1/sde_scalar_stratonovich.jl")
                @time @safetestset "SDE Checkpointing" include("SDE1/sde_checkpointing.jl")
            end
        end,
        "SDE2" => function ()
            return @testset "SDE 2" begin
                @time @safetestset "SDE Non-Diagonal Noise" include("SDE2/sde_nondiag_stratonovich.jl")
            end
        end,
        "SDE3" => function ()
            return @testset "SDE 3" begin
                @time @safetestset "RODE Tests" include("SDE3/rode.jl")
                @time @safetestset "SDE Ito Conversion Tests" include("SDE3/sde_transformation_test.jl")
                @time @safetestset "SDE Ito Scalar Noise" include("SDE3/sde_scalar_ito.jl")
            end
        end,
        "Callbacks1" => function ()
            return @testset "Callbacks 1" begin
                @time @safetestset "Discrete Callbacks with ForwardDiffSensitivity" include("Callbacks1/forward_sensitivity_callback.jl")
                @time @safetestset "Discrete Callbacks with Adjoints" include("Callbacks1/discrete_callbacks.jl")
                @time @safetestset "SDE Callbacks" include("Callbacks1/SDE_callbacks.jl")
                @time @safetestset "Non-tracked callbacks" include("Callbacks1/non_tracked_callbacks.jl")
            end
        end,
        "Callbacks2" => function ()
            return @testset "Callbacks 2" begin
                @time @safetestset "Continuous vs. discrete Callbacks" include("Callbacks2/continuous_vs_discrete.jl")
                @time @safetestset "Continuous Callbacks with Adjoints" include("Callbacks2/continuous_callbacks.jl")
                @time @safetestset "VectorContinuousCallbacks with Adjoints" include("Callbacks2/vector_continuous_callbacks.jl")
            end
        end,
        "Shadowing" => function ()
            return @testset "Shadowing" begin
                @time @safetestset "Shadowing Tests" include("Shadowing/shadowing.jl")
            end
        end,
        "GPU" => (;
            env = joinpath(@__DIR__, "GPU"),
            body = function ()
                return @testset "GPU" begin
                    @time @safetestset "Standard DiffEqFlux GPU" include("GPU/diffeqflux_standard_gpu.jl")
                    @time @safetestset "Mixed GPU/CPU" include("GPU/mixed_gpu_cpu_adjoint.jl")
                end
            end,
        ),
    ),
    qa = function ()
        return @time @safetestset "Quality Assurance" include("QA/aqua.jl")
    end,
    all = [
        "Core1", "Core2", "Core3", "Core4", "Core5", "Core6", "Core7", "Core8",
        "SDE1", "SDE2", "SDE3",
    ],
    umbrellas = Dict("Downstream" => ["Core1", "Core3"]),
)
