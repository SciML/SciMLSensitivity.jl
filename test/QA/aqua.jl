using SciMLTesting, SciMLSensitivity, SciMLBase, Test

run_qa(
    SciMLSensitivity;
    explicit_imports = true,
    aqua_kwargs = (;
        ambiguities = (; recursive = false),
        piracies = (;
            treat_as_own = [
                SciMLBase._concrete_solve_adjoint,
                SciMLBase._concrete_solve_forward,
            ],
        ),
    ),
    ei_kwargs = (;
        # `BrownFullBasicInit`/`DefaultInit` are owned by DiffEqBase but re-exported
        # through OrdinaryDiffEqCore, which is where SciMLSensitivity imports them.
        # The rest are re-imported through the parent `SciMLSensitivity` by the
        # Mooncake extension (`using SciMLSensitivity: ...`), an intentional idiom.
        all_explicit_imports_via_owners = (;
            ignore = (
                :BrownFullBasicInit, :DefaultInit,
                # SciMLSensitivityMooncakeExt: re-imported through the parent module
                :DiffEqBase, :FunctionWrappersWrappers, :ODEFunction, :SciMLBase,
                :SciMLStructures, :Tunable, :canonicalize, :current_time,
                :isscimlstructure, :state_values, :unwrapped_f,
            ),
        ),
        # Non-public names of upstream deps imported explicitly here; ignore until
        # those packages mark them public (each grouped by its source package).
        all_explicit_imports_are_public = (;
            ignore = (
                # ChainRulesCore
                :AbstractTangent,
                # OrdinaryDiffEqCore
                :BrownFullBasicInit, :DefaultInit, :default_nlsolve, :has_autodiff,
                # SciMLBase
                :AbstractAdjointSensitivityAlgorithm, :AbstractDiffEqFunction,
                :AbstractForwardSensitivityAlgorithm, :AbstractNonlinearProblem,
                :AbstractODEFunction, :AbstractOptimizationProblem,
                :AbstractOverloadingSensitivityAlgorithm,
                :AbstractSecondOrderSensitivityAlgorithm, :AbstractSensitivityAlgorithm,
                :AbstractShadowingSensitivityAlgorithm, :AbstractTimeseriesSolution,
                :OverrideInit, :unwrapped_f,
                # SciMLStructures
                :Tunable, :canonicalize, :isscimlstructure,
                # SciMLSensitivityMooncakeExt: imports of the parent's own
                # (non-public) internals + names re-imported through the parent.
                :DiffEqBase, :FakeIntegrator, :FunctionWrappersWrappers,
                :MooncakeLoaded, :MooncakeVJP, :ODEFunction, :SciMLBase,
                :SciMLStructures, :SciMLStructuresCompatibilityError,
                :_init_originator_gradient, :convert_tspan, :current_time,
                :get_cb_paramjac_config, :get_paramjac_config,
                :has_continuous_callback, :mooncake_run_ad, :state_values,
            ),
        ),
        # Non-public names of upstream deps accessed qualified in the source; ignore
        # until those packages mark them public (each grouped by its source package).
        all_qualified_accesses_are_public = (;
            ignore = (
                # ArrayInterface
                :aos_to_soa, :ismutable, :parameterless_type, :restructure,
                # Base
                :(var"@pure"), :_nt_names, :diff_names,
                # DiffEqCallbacks
                :PeriodicCallbackAffect,
                # DiffEqNoiseProcess
                :vec_NoiseProcess,
                # Enzyme / EnzymeCore / EnzymeCore.EnzymeRules
                :EnzymeCore, :Mode, :inactive_type,
                # FiniteDiff
                :DerivativeCache, :GradientCache, :JacobianCache,
                :finite_difference_derivative!, :finite_difference_gradient!,
                :finite_difference_jacobian, :finite_difference_jacobian!,
                # ForwardDiff
                :Chunk, :DerivativeConfig, :Dual, :GradientConfig, :JacobianConfig,
                :Partials, :Tag, :construct_seeds, :derivative!, :gradient!, :jacobian,
                :jacobian!, :npartials, :partials, :pickchunksize, :value,
                # LinearSolve
                :needs_concrete_A,
                # OrdinaryDiffEqCore
                :alg_autodiff, :default_linear_interpolation,
                # ReverseDiff
                :GradientTape, :TrackedArray, :compile, :deriv, :forward_pass!,
                :gradient, :increment_deriv!, :input_hook, :output_hook, :pull_value!,
                :reverse_pass!, :unseed!, :value!,
                # SciMLBase
                :ADOriginator, :AbstractDAEProblem, :AbstractDDEProblem,
                :AbstractDiscreteProblem, :AbstractODEProblem, :AbstractRODEProblem,
                :AbstractSDDEProblem, :AbstractSDEProblem, :AbstractSciMLFunction,
                :AlgorithmInterpretation, :AutoSpecialize, :ChainRulesOriginator,
                :EnzymeOriginator, :FullSpecialize, :ImmutableNonlinearProblem,
                :MooncakeOriginator, :NullParameters, :OVERDETERMINED, :OverrideInit,
                :ParamJacobianWrapper, :ReverseDiffOriginator, :TrackerOriginator,
                :UDerivativeWrapper, :UJacobianWrapper, :Void,
                :_concrete_solve_adjoint, :_concrete_solve_forward, :alg_interpretation,
                :build_solution, :forwarddiffs_model, :forwarddiffs_model_time,
                :get_initial_values, :has_initialization_data, :has_jac, :has_observed,
                :has_paramjac, :has_vjp, :has_vjp_p, :initialization_status,
                :is_diagonal_noise, :sensitivity_solution, :specialization,
                # SciMLStructures
                :replace,
                # SparseArrays
                :AbstractSparseMatrixCSC,
                # Tracker
                :TrackedReal, :collect, :data, :forward,
                # Zygote
                :Buffer, :accum,
                # Mooncake (accessed in SciMLSensitivityMooncakeExt)
                :CoDual, :NoFData, :Tangent, :build_rrule,
                :tangent_to_primal!!, :zero_rdata,
                # Flagged only on Julia LTS (1.10); public on 1.11+:
                :Fix1, :Fix2, :depwarn,             # Base
                :Stratonovich, :Terminated,         # SciMLBase enum modules
                :children, :functor,                # Functors
            ),
        ),
    ),
)
