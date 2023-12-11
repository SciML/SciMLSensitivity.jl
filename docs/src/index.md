# SciMLSensitivity: Automatic Differentiation and Adjoints for (Differential) Equation Solvers

SciMLSensitivity.jl is the automatic differentiation and adjoints system for the SciML
ecosystem. Also known as local sensitivity analysis, these methods allow for calculation
of fast derivatives of SciML problem types which are commonly used to analyze model
sensitivities, calibrate models to data, train neural ODEs, perform automated model
discovery via universal differential equations, and more. SciMLSensitivity.jl is
a high-level interface that pulls together all the tools with heuristics
and helper functions to make solving inverse problems and inferring models
as easy as possible without losing efficiency.

Thus, what SciMLSensitivity.jl provides is:

  - Automatic differentiation overloads for improving the performance and flexibility
    of AD calls over `solve`.
  - A lower level direct interface for defining forward sensitivity and adjoint problems
    to allow for minimal overhead and maximal performance.
  - A bunch of tutorials, documentation, and test cases for this combination
    with parameter estimation (data fitting / model calibration), neural network
    libraries and GPUs.

!!! note
    
    This documentation assumes familiarity with the solver packages for the respective problem
    types. If one is not familiar with the solver packages, please consult the documentation
    for pieces like [DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/),
    [NonlinearSolve.jl](https://docs.sciml.ai/NonlinearSolve/stable/),
    [LinearSolve.jl](https://docs.sciml.ai/LinearSolve/stable/), etc. first.

## Installation

To install SciMLSensitivity.jl, use the Julia package manager:

```julia
using Pkg
Pkg.add("SciMLSensitivity")
```

## High-Level Interface: `sensealg`

The highest level interface is provided by the function `solve`:

```julia
solve(prob, args...; sensealg = InterpolatingAdjoint(), checkpoints = sol.t, kwargs...)
```

`solve` is fully compatible with automatic differentiation libraries
like:

  - [Zygote.jl](https://fluxml.ai/Zygote.jl/stable/)
  - [ReverseDiff.jl](https://juliadiff.org/ReverseDiff.jl/)
  - [Tracker.jl](https://github.com/FluxML/Tracker.jl)
  - [ForwardDiff.jl](https://juliadiff.org/ForwardDiff.jl/stable/)

and will automatically replace any calculations of the solution's derivative
with a fast method. The keyword argument `sensealg` controls the dispatch to the
`AbstractSensitivityAlgorithm` used for the sensitivity calculation.
Note that `solve` in an AD context does not allow higher order
interpolations unless `sensealg=DiffEqBase.SensitivityADPassThrough()`
is used, i.e. going back to the AD mechanism.

!!! note
    
    The behavior of ForwardDiff.jl is different from the other automatic differentiation libraries mentioned above.
    The `sensealg` keyword is ignored. Instead, the differential equations are solved using `Dual` numbers for `u0` and `p`.
    If only `p` is perturbed in the sensitivity analysis, but not `u0`, the state is still implemented as a `Dual` number.
    ForwardDiff.jl will thus not dispatch into continuous forward nor adjoint sensitivity analysis even if a `sensealg` is provided.

## Equation Scope

SciMLSensitivity.jl supports all the equation types of the
[SciML Common Interface](https://docs.sciml.ai/SciMLBase/stable/), extending the problem
types by adding overloads for automatic differentiation to improve the performance
and flexibility of the differentiation system. This includes:

  - Linear systems (`LinearProblem`)
    
      + Direct methods for dense and sparse
      + Iterative solvers with preconditioning

  - Nonlinear Systems (`NonlinearProblem`)
    
      + Systems of nonlinear equations
      + Scalar bracketing systems
  - Integrals (quadrature) (`QuadratureProblem`)
  - Differential Equations
    
      + Discrete equations (function maps, discrete stochastic (Gillespie/Markov)
        simulations) (`DiscreteProblem`)
      + Ordinary differential equations (ODEs) (`ODEProblem`)
      + Split and Partitioned ODEs (Symplectic integrators, IMEX Methods) (`SplitODEProblem`)
      + Stochastic ordinary differential equations (SODEs or SDEs) (`SDEProblem`)
      + Stochastic differential-algebraic equations (SDAEs) (`SDEProblem` with mass matrices)
      + Random differential equations (RODEs or RDEs) (`RODEProblem`)
      + Differential algebraic equations (DAEs) (`DAEProblem` and `ODEProblem` with mass matrices)
      + Delay differential equations (DDEs) (`DDEProblem`)
      + Neutral, retarded, and algebraic delay differential equations (NDDEs, RDDEs, and DDAEs)
      + Stochastic delay differential equations (SDDEs) (`SDDEProblem`)
      + Experimental support for stochastic neutral, retarded, and algebraic delay differential equations (SNDDEs, SRDDEs, and SDDAEs)
      + Mixed discrete and continuous equations (Hybrid Equations, Jump Diffusions) (`DEProblem`s with callbacks)
  - Optimization (`OptimizationProblem`)
    
      + Nonlinear (constrained) optimization
  - (Stochastic/Delay/Differential-Algebraic) Partial Differential Equations (`PDESystem`)
    
      + Finite difference and finite volume methods
      + Interfaces to finite element methods
      + Physics-Informed Neural Networks (PINNs)
      + Integro-Differential Equations
      + Fractional Differential Equations

## SciMLSensitivity and Universal Differential Equations

SciMLSensitivity is for universal differential equations, where these can include
delays, physical constraints, stochasticity, events, and all other kinds of
interesting behavior that shows up in scientific simulations. Neural networks can
be all or part of the model. They can be around the differential equation,
in the cost function, or inside the differential equation. Neural networks
representing unknown portions of the model or functions can go anywhere you
have uncertainty in the form of the scientific simulator. Forward sensitivity
and adjoint equations are automatically generated with checkpointing and
stabilization to ensure it works for large stiff equations, while specializations
on static objects allows for high efficiency on small equations. For an overview
of the topic with applications, consult the paper
[Universal Differential Equations for Scientific Machine
Learning](https://arxiv.org/abs/2001.04385).

You can efficiently use the package for:

  - Parameter estimation of scientific models (ODEs, SDEs, DDEs, DAEs, etc.)
  - Neural ODEs, Neural SDE, Neural DAEs, Neural DDEs, etc.
  - Nonlinear optimal control, including training neural controllers
  - (Stiff) universal ordinary differential equations (universal ODEs)
  - Universal stochastic differential equations (universal SDEs)
  - Universal delay differential equations (universal DDEs)
  - Universal partial differential equations (universal PDEs)
  - Universal jump stochastic differential equations (universal jump diffusions)
  - Hybrid universal differential equations (universal DEs with event handling)

with high order, adaptive, implicit, GPU-accelerated, Newton-Krylov, etc.
methods. For examples, please refer to [the DiffEqFlux release blog
post](https://julialang.org/blog/2019/01/fluxdiffeq) (which we try to keep
updated for changes to the libraries). Additional demonstrations, like neural
PDEs and neural jump SDEs, can be found [at this blog
post](http://www.stochasticlifestyle.com/neural-jump-sdes-jump-diffusions-and-neural-pdes/)
(among many others!). All these features are only part of the advantage, as this library
[routinely benchmarks orders of magnitude faster than competing libraries like torchdiffeq](@ref Benchmarks).
Use with GPUs is highly optimized by
[recompiling the solvers to GPUs to remove all CPU-GPU data transfers](https://www.stochasticlifestyle.com/solving-systems-stochastic-pdes-using-gpus-julia/),
while use with CPUs uses specialized kernels for accelerating differential equation solves.

Many training techniques are supported by this package, including:

  - Optimize-then-discretize (backsolve adjoints, checkpointed adjoints, quadrature adjoints)

  - Discretize-then-optimize (forward and reverse mode discrete sensitivity analysis)
    
      + This is a generalization of [ANODE](https://arxiv.org/pdf/1902.10298.pdf) and
        [ANODEv2](https://arxiv.org/pdf/1906.04596.pdf) to all
        [DifferentialEquations.jl ODE solvers](https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/)
  - Hybrid approaches (adaptive time stepping + AD for adaptive discretize-then-optimize)
  - O(1) memory backprop of ODEs via BacksolveAdjoint, and Virtual Brownian Trees for O(1) backprop of SDEs
  - [Continuous adjoints for integral loss functions](@ref continuous_loss)
  - Probabilistic programming and variational inference on ODEs/SDEs/DAEs/DDEs/hybrid
    equations etc. is provided by integration with [Turing.jl](https://turinglang.org/v0.29/docs/using-turing/get-started)
    and [Gen.jl](https://github.com/probcomp/Gen.jl). Reproduce
    [variational loss functions](https://arxiv.org/abs/2001.01328) by plugging
    [composable libraries together](https://turinglang.org/v0.29/tutorials/09-variational-inference/).

all while mixing forward mode and reverse mode approaches as appropriate for the
most speed. For more details on the adjoint sensitivity analysis methods for
computing fast gradients, see the [adjoints details page](@ref sensitivity_diffeq).

With this package, you can explore various ways to integrate the two methodologies:

  - Neural networks can be defined where the “activations” are nonlinear functions
    described by differential equations
  - Neural networks can be defined where some layers are ODE solves
  - ODEs can be defined where some terms are neural networks
  - Cost functions on ODEs can define neural networks

## Note on Modularity and Composability with Solvers

Note that SciMLSensitivity.jl purely built on composable and modular infrastructure.
SciMLSensitivity provides high-level helper functions and documentation for the user, but the
code generation stack is modular and composes in many ways. For example, one can
use and swap out the ODE solver between any common interface compatible library, like:

  - Sundials.jl
  - OrdinaryDiffEq.jl
  - LSODA.jl
  - [IRKGaussLegendre.jl](https://github.com/mikelehu/IRKGaussLegendre.jl)
  - [SciPyDiffEq.jl](https://github.com/SciML/SciPyDiffEq.jl)
  - [… etc. many other choices!](https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/)

In addition, due to the composability of the system, none of the components are directly
tied to the Lux.jl machine learning framework. For example, you can [use SciMLSensitivity.jl
to generate TensorFlow graphs and train the neural network with TensorFlow.jl](https://youtu.be/n2MwJ1guGVQ?t=284),
[use PyTorch arrays via Torch.jl](https://github.com/FluxML/Torch.jl), and more all with
single line code changes by utilizing the underlying code generation. The tutorials shown here
are thus mostly a guide on how to use the ecosystem as a whole, only showing a small snippet
of the possible ways to compose the thousands of differentiable libraries together! Swap out
ODEs for SDEs, DDEs, DAEs, etc., put quadrature libraries or
[Tullio.jl](https://github.com/mcabbott/Tullio.jl) in the loss function, the world is your
oyster!

As a proof of composability, note that the implementation of Bayesian neural ODEs required
zero code changes to the library, and instead just relied on the composability with other
Julia packages.

## Contributing

  - Please refer to the
    [SciML ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://github.com/SciML/ColPrac/blob/master/README.md)
    for guidance on PRs, issues, and other matters relating to contributing to SciML.

  - See the [SciML Style Guide](https://github.com/SciML/SciMLStyle) for common coding practices and other style decisions.
  - There are a few community forums:
    
      + The #diffeq-bridged and #sciml-bridged channels in the
        [Julia Slack](https://julialang.org/slack/)
      + The #diffeq-bridged and #sciml-bridged channels in the
        [Julia Zulip](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
      + On the [Julia Discourse forums](https://discourse.julialang.org)
      + See also [SciML Community page](https://sciml.ai/community/)

## Citation

If you use SciMLSensitivity.jl or are influenced by its ideas, please cite:

```
@article{rackauckas2020universal,
  title={Universal differential equations for scientific machine learning},
  author={Rackauckas, Christopher and Ma, Yingbo and Martensen, Julius and Warner, Collin and Zubov, Kirill and Supekar, Rohit and Skinner, Dominic and Ramadhan, Ali},
  journal={arXiv preprint arXiv:2001.04385},
  year={2020}
}
```

## Reproducibility

```@raw html
<details><summary>The documentation of this SciML package was built using these direct dependencies,</summary>
```

```@example
using Pkg # hide
Pkg.status() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>and using this machine and Julia version.</summary>
```

```@example
using InteractiveUtils # hide
versioninfo() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>A more complete overview of all dependencies and their versions is also provided.</summary>
```

```@example
using Pkg # hide
Pkg.status(; mode = PKGMODE_MANIFEST) # hide
```

```@raw html
</details>
```

```@eval
using TOML
using Markdown
version = TOML.parse(read("../../Project.toml", String))["version"]
name = TOML.parse(read("../../Project.toml", String))["name"]
link_manifest = "https://github.com/SciML/" * name * ".jl/tree/gh-pages/v" * version *
                "/assets/Manifest.toml"
link_project = "https://github.com/SciML/" * name * ".jl/tree/gh-pages/v" * version *
               "/assets/Project.toml"
Markdown.parse("""You can also download the
[manifest]($link_manifest)
file and the
[project]($link_project)
file.
""")
```
