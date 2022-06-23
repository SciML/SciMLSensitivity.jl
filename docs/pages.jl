pages = [
        "DiffEqSensitivity.jl: Automatic Differentiation and Adjoints for (Differential) Equation Solvers" => "index.md",
        "Tutorials" => Any[
            "Differentiating Ordinary Differential Equations (ODE) Tutorials" => Any[
                "ad_examples/differentiating_ode.md",
                "ad_examples/direct_sensitivity.md",
                "ad_examples/adjoint_continuous_functional.md",
                "ad_examples/chaotic_ode.md",
            ],
            "Fitting Ordinary Differential Equation (ODE) Tutorials" => Any[
                "ode_fitting/optimization_ode.md",
                "ode_fitting/stiff_ode_fit.md",
                "ode_fitting/exogenous_input.md",
                "ode_fitting/data_parallel.md",
                "ode_fitting/prediction_error_method.md",
                "ode_fitting/second_order_adjoints.md",
                "ode_fitting/second_order_neural.md",
            ],
            "Training Techniques and Tips" => Any[
                "training_tips/local_minima.md",
                "training_tips/divergence.md",
                "training_tips/multiple_nn.md",
            ],
            "Neural Ordinary Differential Equation (Neural ODE) Tutorials" => Any[
                "neural_ode/neural_ode_flux.md",
                "neural_ode/mnist_neural_ode.md",
                "neural_ode/mnist_conv_neural_ode.md",
                "neural_ode/GPUs.md",
                "neural_ode/neural_gde.md",
                "neural_ode/minibatch.md",
            ],
            "Stochastic Differential Equation (SDE) Tutorials" => Any[
                "sde_fitting/optimization_sde.md",
            ],
            "Delay Differential Equation (DDE) Tutorials" => Any[
                "dde_fitting/delay_diffeq.md",
            ],
            "Differential-Algebraic Equation (DAE) Tutorials" => Any[
                "dae_fitting/physical_constraints.md",
            ],
            "Partial Differential Equation (PDE) Tutorials" => Any[
                "pde_fitting/pde_constrained.md",
            ],
            "Hybrid and Jump Equation Tutorials" => Any[
                "hybrid_jump_fitting/hybrid_diffeq.md",
                "hybrid_jump_fitting/bouncing_ball.md",
            ],
            "Bayesian Estimation Tutorials" => Any[
                "bayesian/turing_bayesian.md",
            ],
            "Optimal and Model Predictive Control Tutorials" => Any[
                "optimal_control/optimal_control.md",
                "optimal_control/feedback_control.md",
                "optimal_control/SDE_control.md",
            ],
        ],
        "Manual and APIs" => Any[
            "manual/differential_equation_sensitivities.md",
            "manual/nonlinear_solve_sensitivities.md",
            "manual/direct_forward_sensitivity.md",
            "manual/direct_adjoint_sensitivities.md",
        ],
        "Benchmarks" => "Benchmark.md",
        "Sensitivity Math Details" => "sensitivity_math.md",
    ]