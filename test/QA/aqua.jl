using SciMLTesting, SciMLSensitivity, SciMLBase, Test

run_qa(
    SciMLSensitivity;
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
                :AbstractAdjointSensitivityAlgorithm,
                :AbstractForwardSensitivityAlgorithm, :AbstractOptimizationProblem,
                :AbstractOverloadingSensitivityAlgorithm,
                :AbstractSecondOrderSensitivityAlgorithm, :AbstractSensitivityAlgorithm,
                :AbstractShadowingSensitivityAlgorithm, :AbstractTimeseriesSolution,
                :unwrapped_f,
                # SciMLStructures
                :Tunable, :canonicalize, :isscimlstructure,
                # SciMLSensitivityMooncakeExt re-imports these internal/non-public
                # names through the parent module (`using/import SciMLSensitivity: ...`),
                # the intentional extension idiom. They are SciMLSensitivity internals
                # or deps re-exported by the parent, so they are not public in
                # SciMLSensitivity and never will be.
                :DiffEqBase, :FakeIntegrator, :FunctionWrappersWrappers, :MooncakeLoaded,
                :MooncakeVJP, :ODEFunction, :SciMLBase, :SciMLStructures,
                :SciMLStructuresCompatibilityError, :_init_originator_gradient,
                :convert_tspan, :current_time, :get_cb_paramjac_config,
                :get_paramjac_config, :has_continuous_callback, :mooncake_run_ad,
                :state_values,
                # SciMLSensitivityMooncakeExt: Mooncake's rule-writing API, explicitly
                # imported (`import Mooncake: ...`) for the native `MooncakeAdjoint`
                # `solve_up` rule. There is no public spelling for writing a Mooncake
                # rule (see the matching comment in all_qualified_accesses_are_public
                # below); ignore until Mooncake marks its rule API public.
                :(var"@is_primitive"), :(var"@zero_derivative"), :CoDual, :ForwardMode,
                :MinimalCtx, Symbol("frule!!"), Symbol("rrule!!"),
            ),
        ),
        # Non-public names of upstream deps accessed qualified in the source; ignore
        # until those packages mark them public (each grouped by its source package).
        all_qualified_accesses_are_public = (;
            ignore = (
                # ArrayInterface
                :parameterless_type,
                # Base
                :(var"@pure"), :_nt_names, :diff_names, :inferencebarrier, :structdiff,
                # Core
                :kwcall,
                # DiffEqBase (internal `solve_up`/`_solve_adjoint` rrule plumbing
                # used by SciMLSensitivityMooncakeExt)
                :AbstractDEProblem, :_solve_adjoint, :solve_up,
                # DiffEqCallbacks
                :PeriodicCallbackAffect,
                # DiffEqNoiseProcess
                :vec_NoiseProcess,
                # Enzyme
                :EnzymeCore,
                # EnzymeCore
                :Mode,
                # EnzymeCore.EnzymeRules
                :inactive_type, :inactive,
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
                # Mooncake (internal tangent/rrule API used by SciMLSensitivityMooncakeExt)
                :CoDual, :NoRData, :Tangent, :build_rrule, :fdata,
                :increment_and_get_rdata!, :instantiate, :lazy_zero_rdata,
                :mooncake_tangent, :primal, :rdata, :rrule!!, :tangent,
                :tangent_to_primal!!, :to_cr_tangent, :to_fwds, :tuple_map, :zero_tangent,
                # OrdinaryDiffEqCore
                :alg_autodiff, :default_linear_interpolation,
                # ReverseDiff
                :GradientTape, :TrackedArray, :compile, :deriv, :forward_pass!, :gradient,
                :increment_deriv!, :input_hook, :output_hook, :pull_value!, :reverse_pass!,
                :unseed!, :value, :value!,
                # SciMLBase
                :ADOriginator, :AbstractDAESolution, :AbstractRODEProblem,
                :AbstractSDDEProblem,
                :AlgorithmInterpretation, :ChainRulesOriginator, :EnzymeOriginator,
                :FullSpecialize, :ImmutableNonlinearProblem, :MooncakeOriginator,
                :OVERDETERMINED, :ParamJacobianWrapper, :ReverseDiffOriginator,
                :TrackerOriginator, :UDerivativeWrapper, :UJacobianWrapper, :Void,
                :_concrete_solve_adjoint, :_concrete_solve_forward, :alg_interpretation,
                :enable_interpolation_sensitivitymode,
                :has_initialization_data, :has_observed, :has_paramjac, :has_vjp_p,
                :initialization_status, :sensitivity_solution, :specialization,
                # SciMLSensitivity (own internal, accessed qualified from the extension)
                :automatic_sensealg_choice,
                # SciMLStructures
                :replace,
                # SparseArrays
                :AbstractSparseMatrixCSC,
                # Tracker
                :TrackedReal, :collect, :data, :forward,
                # Zygote
                :Buffer, :accum,
            ),
        ),
    ),
)
