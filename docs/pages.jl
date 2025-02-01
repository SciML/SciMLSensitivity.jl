pages = ["index.md",
    "getting_started.md",
    "Tutorials" => Any["tutorials/parameter_estimation_ode.md",
        "tutorials/direct_sensitivity.md",
        "tutorials/adjoint_continuous_functional.md",
        "tutorials/data_parallel.md",
        "tutorials/chaotic_ode.md",
        "Training Techniques and Tips" => Any["tutorials/training_tips/local_minima.md",
            "tutorials/training_tips/divergence.md",
            "tutorials/training_tips/multiple_nn.md"]],
    "Frequently Asked Questions (FAQ)" => "faq.md",
    "Examples" => Any[
        "Ordinary Differential Equations (ODEs)" => Any["examples/ode/exogenous_input.md",
            "examples/ode/prediction_error_method.md",
            "examples/ode/second_order_adjoints.md",
            "examples/ode/second_order_neural.md"],
        "Neural Ordinary Differential Equations (Neural ODE)" => Any[
            "examples/neural_ode/simplechains.md"],
        "Stochastic Differential Equations (SDEs)" => Any[
            "examples/sde/optimization_sde.md",
            "examples/sde/SDE_control.md"],
        "Delay Differential Equations (DDEs)" => Any["examples/dde/delay_diffeq.md"],
        "Partial Differential Equations (PDEs)" => Any["examples/pde/pde_constrained.md"],
        "Hybrid and Jump Equations" => Any["examples/hybrid_jump/hybrid_diffeq.md",
            "examples/hybrid_jump/bouncing_ball.md"],
        "Bayesian Estimation" => Any["examples/bayesian/turing_bayesian.md"],
        "Optimal and Model Predictive Control" => Any[
            "examples/optimal_control/optimal_control.md",
            "examples/optimal_control/feedback_control.md"]],
    "Manual and APIs" => Any[
        "manual/differential_equation_sensitivities.md",
        "manual/nonlinear_solve_sensitivities.md",
        "manual/direct_forward_sensitivity.md",
        "manual/direct_adjoint_sensitivities.md"],
    "Benchmarks" => "Benchmark.md",
    "Sensitivity Math Details" => "sensitivity_math.md"
]
