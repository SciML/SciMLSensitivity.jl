pages = ["Home" => "index.md",
    "Get Started" => "getting_started.md",
    "Tutorials" => [
        "Parameter estimation ode" => "tutorials/parameter_estimation_ode.md",
        "Training Techniques and Tips" => [
            "Local minima" => "tutorials/training_tips/local_minima.md",
            ]
        ],
    "Examples" => [
        "Ordinary Differential Equations (ODEs)" => [
            "Exogenous input"=>"examples/ode/exogenous_input.md",
            "Prediction error method" => "examples/ode/prediction_error_method.md",
            "Second order adjoints" => "examples/ode/second_order_adjoints.md",
            "second_order_neural" => "examples/ode/second_order_neural.md"
            ],
        "Neural Ordinary Differential Equations (Neural ODE)" => [
            "neural ode flux"=>"examples/neural_ode/neural_ode_flux.md",
            "Simple chains" => "examples/neural_ode/simplechains.md",
            "Minibatch" => "examples/neural_ode/minibatch.md"
            ],
        "Stochastic Differential Equations (SDEs)" => [
            "optimization sde"=>"examples/sde/optimization_sde.md",
            "SDE_control" => "examples/sde/SDE_control.md"
            ],
        "Delay Differential Equations (DDEs)" => [
            "delay diffeq"=>"examples/dde/delay_diffeq.md"
            ],
        "Partial Differential Equations (PDEs)" => [
            "pde constrained"=>"examples/pde/pde_constrained.md"
            ],
        "Hybrid and Jump Equations" => [
            "Hybrid diffeq"=>"examples/hybrid_jump/hybrid_diffeq.md",
            "Bouncing ball" => "examples/hybrid_jump/bouncing_ball.md"
            ],
        "Bayesian Estimation" => [
            "Turing Bayesian"=>"examples/bayesian/turing_bayesian.md"
            ],
        "Optimal and Model Predictive Control" => [
            "Optimal control"=>"examples/optimal_control/optimal_control.md",
            "Feedback control" => "examples/optimal_control/feedback_control.md"
            ]
        ],
    "Manual and APIs" => [
        "Differential equation sensitivities" => "manual/differential_equation_sensitivities.md",
        "Nonlinear solve sensitivities" => "manual/nonlinear_solve_sensitivities.md",
        "Direct forward sensitivity" => "manual/direct_forward_sensitivity.md",
        "Direct adjoint sensitivities" => "manual/direct_adjoint_sensitivities.md"
        ],
    "Benchmarks" => "Benchmark.md",
    "Sensitivity Math Details" => "sensitivity_math.md"
]