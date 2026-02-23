function SensitivityAlg(args...; kwargs...)
    return @error("The SensitivityAlg choice mechanism was completely overhauled. Please consult the local sensitivity documentation for more information")
end

"""
```julia
ForwardSensitivity{CS, AD, FDT} <: AbstractForwardSensitivityAlgorithm{CS, AD, FDT}
```

An implementation of continuous forward sensitivity analysis for propagating
derivatives by solving the extended ODE. When used within adjoint differentiation
(i.e. via Zygote), this will cause forward differentiation of the `solve` call
within the reverse-mode automatic differentiation environment.

## Constructor

```julia
ForwardSensitivity(;
    chunk_size = 0, autodiff = true,
    diff_type = Val{:central},
    autojacvec = autodiff,
    autojacmat = false)
```

## Keyword Arguments

  - `autodiff`: Use automatic differentiation in the internal sensitivity algorithm
    computations. Default is `true`.
  - `chunk_size`: Chunk size for forward mode differentiation if full Jacobians are
    built (`autojacvec=false` and `autodiff=true`). Default is `0` for automatic
    choice of chunk size.
  - `autojacvec`: Calculate the Jacobian-vector product via automatic
    differentiation with special seeding.
  - `diff_type`: The method used by FiniteDiff.jl for constructing the Jacobian
    if the full Jacobian is required with `autodiff=false`.

Further details:

  - If `autodiff=true` and `autojacvec=true`, then the one chunk `J*v` forward-mode
    directional derivative calculation trick is used to compute the product without
    constructing the Jacobian (via ForwardDiff.jl).
  - If `autodiff=false` and `autojacvec=true`, then the numerical direction derivative
    trick `(f(x+epsilon*v)-f(x))/epsilon` is used to compute `J*v` without constructing
    the Jacobian.
  - If `autodiff=true` and `autojacvec=false`, then the Jacobian is constructed via
    chunked forward-mode automatic differentiation (via ForwardDiff.jl).
  - If `autodiff=false` and `autojacvec=false`, then the Jacobian is constructed via
    finite differences via FiniteDiff.jl.

## SciMLProblem Support

This `sensealg` only supports `ODEProblem`s without callbacks (events).
"""
struct ForwardSensitivity{CS, AD, FDT} <: AbstractForwardSensitivityAlgorithm{CS, AD, FDT}
    autojacvec::Bool
    autojacmat::Bool
end
function ForwardSensitivity(;
        chunk_size = 0, autodiff = true,
        diff_type = Val{:central},
        autojacvec = autodiff,
        autojacmat = false
    )
    autojacvec && autojacmat &&
        error("Choose either Jacobian matrix products or Jacobian vector products,
                autojacmat and autojacvec cannot both be true")
    return ForwardSensitivity{chunk_size, autodiff, diff_type}(autojacvec, autojacmat)
end

"""
```julia
ForwardDiffSensitivity{CS, CTS} <: AbstractForwardSensitivityAlgorithm{CS, Nothing, Nothing}
```

An implementation of discrete forward sensitivity analysis through ForwardDiff.jl.
When used within adjoint differentiation (i.e. via Zygote), this will cause forward
differentiation of the `solve` call within the reverse-mode automatic differentiation
environment.

## Constructor

```julia
ForwardDiffSensitivity(; chunk_size = 0, convert_tspan = nothing)
```

## Keyword Arguments

  - `chunk_size`: the chunk size used by ForwardDiff for computing the Jacobian, i.e. the
    number of simultaneous columns computed.
  - `convert_tspan`: whether to convert time to also be `Dual` valued. By default this is
    `nothing` which will only convert if callbacks are found. Conversion is required in order
    to accurately differentiate callbacks (hybrid equations).

## SciMLProblem Support

This `sensealg` supports any `SciMLProblem`s, provided that the solver algorithms is
`SciMLBase.isautodifferentiable`. Note that `ForwardDiffSensitivity` can
accurately differentiate code with callbacks only when `convert_tspan=true`.
"""
struct ForwardDiffSensitivity{CS, CTS} <:
    AbstractForwardSensitivityAlgorithm{CS, Nothing, Nothing} end
function ForwardDiffSensitivity(; chunk_size = 0, convert_tspan = nothing)
    return ForwardDiffSensitivity{chunk_size, convert_tspan}()
end

"""
```julia
BacksolveAdjoint{CS, AD, FDT, VJP} <: AbstractAdjointSensitivityAlgorithm{CS, AD, FDT}
```

An implementation of adjoint sensitivity analysis using a backwards solution of the ODE.
By default, this algorithm will use the values from the forward pass to perturb the
backwards solution to the correct spot, allowing reduced memory (O(1) memory). Checkpointing
stabilization is included for additional numerical stability over the naive implementation.

## Constructor

```julia
BacksolveAdjoint(; chunk_size = 0, autodiff = true,
    diff_type = Val{:central},
    autojacvec = nothing,
    checkpointing = true, noisemixing = false)
```

## Keyword Arguments

  - `autodiff`: Use automatic differentiation for constructing the Jacobian
    if the Jacobian needs to be constructed.  Defaults to `true`.

  - `chunk_size`: Chunk size for forward-mode differentiation if full Jacobians are
    built (`autojacvec=false` and `autodiff=true`). Default is `0` for automatic
    choice of chunk size.
  - `diff_type`: The method used by FiniteDiff.jl for constructing the Jacobian
    if the full Jacobian is required with `autodiff=false`.
  - `autojacvec`: Calculate the vector-Jacobian product (`J'*v`) via automatic
    differentiation with special seeding. The total set of choices are:

      + `nothing`: uses an automatic algorithm to automatically choose the vjp.
        This is the default and recommended for most users.
      + `false`: the Jacobian is constructed via FiniteDiff.jl
      + `true`: the Jacobian is constructed via ForwardDiff.jl
      + `TrackerVJP`: Uses Tracker.jl for the vjp.
      + `ZygoteVJP`: Uses Zygote.jl for the vjp.
      + `EnzymeVJP`: Uses Enzyme.jl for the vjp.
      + `ReactantVJP`: Uses Reactant.jl-compiled Enzyme.jl for the vjp.
        Requires `using Reactant`.
      + `ReverseDiffVJP(compile=false)`: Uses ReverseDiff.jl for the vjp. `compile`
        is a boolean for whether to precompile the tape, which should only be done
        if there are no branches (`if` or `while` statements) in the `f` function.
  - `checkpointing`: whether checkpointing is enabled for the reverse pass. Defaults
    to `true`.
  - `noisemixing`: Handle noise processes that are not of the form `du[i] = f(u[i])`.
    For example, to compute the sensitivities of an SDE with diagonal diffusion

    ```julia
    function g_mixing!(du, u, p, t)
        du[1] = p[3] * u[1] + p[4] * u[2]
        du[2] = p[3] * u[1] + p[4] * u[2]
        nothing
    end
    ```

    correctly, `noisemixing=true` must be enabled. The default is `false`.

For more details on the vjp choices, please consult the sensitivity algorithms
documentation page or the docstrings of the vjp types.

## Applicability of Backsolve and Caution

When `BacksolveAdjoint` is applicable, it is a fast method, and requires the least memory.
However, one must be cautious because not all ODEs are stable under backwards integration
by the majority of ODE solvers. An example of such an equation is the Lorenz equation.
Notice that if one solves the Lorenz equation forward and then in reverse with any
adaptive time step and non-reversible integrator, then the backwards solution diverges
from the forward solution. As a quick demonstration:

```julia
using Sundials
function lorenz(du, u, p, t)
    du[1] = 10.0 * (u[2] - u[1])
    du[2] = u[1] * (28.0 - u[3]) - u[2]
    du[3] = u[1] * u[2] - (8 / 3) * u[3]
end
u0 = [1.0; 0.0; 0.0]
tspan = (0.0, 100.0)
prob = ODEProblem(lorenz, u0, tspan)
sol = solve(prob, Tsit5(), reltol = 1e-12, abstol = 1e-12)
prob2 = ODEProblem(lorenz, sol.u[end], (100.0, 0.0))
sol = solve(prob, Tsit5(), reltol = 1e-12, abstol = 1e-12)
@show sol.u[end] - u0 #[-3.22091, -1.49394, 21.3435]
```

Thus, one should check the stability of the backsolve on their type of problem before
enabling this method. Additionally, using checkpointing with backsolve can be a
low memory way to stabilize it.

For more details on this topic, see
[Stiff Neural Ordinary Differential Equations](https://aip.scitation.org/doi/10.1063/5.0060697).

## Checkpointing

To improve the numerical stability of the reverse pass, `BacksolveAdjoint` includes a checkpointing
feature. If `sol.u` is a time series, then whenever a time `sol.t` is hit while reversing, a callback
will replace the reversing ODE portion with `sol.u[i]`. This nudges the solution back onto the appropriate
trajectory and reduces the numerical caused by drift.

## SciMLProblem Support

This `sensealg` only supports `ODEProblem`s, `SDEProblem`s, and `RODEProblem`s. This `sensealg` supports
callback functions (events).

### Disclaimer for `SDEProblem`s

The runtime of this algorithm is in O(n^2) for diagonal-noise SDEs until issue #854 is solved.

## References

ODE:
Rackauckas, C. and Ma, Y. and Martensen, J. and Warner, C. and Zubov, K. and Supekar,
R. and Skinner, D. and Ramadhana, A. and Edelman, A., Universal Differential Equations
for Scientific Machine Learning,	arXiv:2001.04385

Hindmarsh, A. C. and Brown, P. N. and Grant, K. E. and Lee, S. L. and Serban, R.
and Shumaker, D. E. and Woodward, C. S., SUNDIALS: Suite of nonlinear and
differential/algebraic equation solvers, ACM Transactions on Mathematical
Software (TOMS), 31, pp:363–396 (2005)

Chen, R.T.Q. and Rubanova, Y. and Bettencourt, J. and Duvenaud, D. K.,
Neural ordinary differential equations. In Advances in neural information processing
systems, pp. 6571–6583 (2018)

Pontryagin, L. S. and Mishchenko, E.F. and Boltyanskii, V.G. and Gamkrelidze, R.V.
The mathematical theory of optimal processes. Routledge, (1962)

Rackauckas, C. and Ma, Y. and Dixit, V. and Guo, X. and Innes, M. and Revels, J.
and Nyberg, J. and Ivaturi, V., A comparison of automatic differentiation and
continuous sensitivity analysis for derivatives of differential equation solutions,
arXiv:1812.01892

DAE:
Cao, Y. and Li, S. and Petzold, L. and Serban, R., Adjoint sensitivity analysis
for differential-algebraic equations: The adjoint DAE system and its numerical
solution, SIAM journal on scientific computing 24 pp: 1076-1089 (2003)

SDE:
Gobet, E. and Munos, R., Sensitivity Analysis Using Ito-Malliavin Calculus and
Martingales, and Application to Stochastic Optimal Control,
SIAM Journal on control and optimization, 43, pp. 1676-1713 (2005)

Li, X. and Wong, T.-K. L.and Chen, R. T. Q. and Duvenaud, D.,
Scalable Gradients for Stochastic Differential Equations,
PMLR 108, pp. 3870-3882 (2020), http://proceedings.mlr.press/v108/li20i.html
"""
struct BacksolveAdjoint{CS, AD, FDT, VJP} <:
    AbstractAdjointSensitivityAlgorithm{CS, AD, FDT}
    autojacvec::VJP
    checkpointing::Bool
    noisemixing::Bool
end
Base.@pure function BacksolveAdjoint(;
        chunk_size = 0, autodiff = true,
        diff_type = Val{:central},
        autojacvec = nothing,
        checkpointing = true, noisemixing = false
    )
    BacksolveAdjoint{chunk_size, autodiff, diff_type, typeof(autojacvec)}(
        autojacvec,
        checkpointing,
        noisemixing
    )
end

function setvjp(sensealg::BacksolveAdjoint{CS, AD, FDT}, vjp) where {CS, AD, FDT}
    return BacksolveAdjoint{CS, AD, FDT, typeof(vjp)}(
        vjp, sensealg.checkpointing,
        sensealg.noisemixing
    )
end

"""
```julia
InterpolatingAdjoint{CS, AD, FDT, VJP} <: AbstractAdjointSensitivityAlgorithm{CS, AD, FDT}
```

An implementation of adjoint sensitivity analysis which uses the interpolation of
the forward solution for the reverse solve vector-Jacobian products. By
default it requires, a dense solution of the forward pass and will internally
ignore saving arguments during the gradient calculation. When checkpointing is
enabled, it will only require the memory to interpolate between checkpoints.

## Constructor

```julia
InterpolatingAdjoint(; chunk_size = 0, autodiff = true,
    diff_type = Val{:central},
    autojacvec = nothing,
    checkpointing = false, noisemixing = false)
```

## Keyword Arguments

  - `autodiff`: Use automatic differentiation for constructing the Jacobian
    if the Jacobian needs to be constructed.  Defaults to `true`.

  - `chunk_size`: Chunk size for forward-mode differentiation if full Jacobians are
    built (`autojacvec=false` and `autodiff=true`). Default is `0` for automatic
    choice of chunk size.
  - `diff_type`: The method used by FiniteDiff.jl for constructing the Jacobian
    if the full Jacobian is required with `autodiff=false`.
  - `autojacvec`: Calculate the vector-Jacobian product (`J'*v`) via automatic
    differentiation with special seeding. The total set of choices are:

      + `nothing`: uses an automatic algorithm to automatically choose the vjp.
        This is the default and recommended for most users.
      + `false`: the Jacobian is constructed via FiniteDiff.jl
      + `true`: the Jacobian is constructed via ForwardDiff.jl
      + `TrackerVJP`: Uses Tracker.jl for the vjp.
      + `ZygoteVJP`: Uses Zygote.jl for the vjp.
      + `EnzymeVJP`: Uses Enzyme.jl for the vjp.
      + `ReactantVJP`: Uses Reactant.jl-compiled Enzyme.jl for the vjp.
        Requires `using Reactant`.
      + `ReverseDiffVJP(compile=false)`: Uses ReverseDiff.jl for the vjp. `compile`
        is a boolean for whether to precompile the tape, which should only be done
        if there are no branches (`if` or `while` statements) in the `f` function.
  - `checkpointing`: whether checkpointing is enabled for the reverse pass. Defaults
    to `false`.
  - `noisemixing`: Handle noise processes that are not of the form `du[i] = f(u[i])`.
    For example, to compute the sensitivities of an SDE with diagonal diffusion

    ```julia
    function g_mixing!(du, u, p, t)
        du[1] = p[3] * u[1] + p[4] * u[2]
        du[2] = p[3] * u[1] + p[4] * u[2]
        nothing
    end
    ```

    correctly, `noisemixing=true` must be enabled. The default is `false`.

For more details on the vjp choices, please consult the sensitivity algorithms
documentation page or the docstrings of the vjp types.

## Checkpointing

To reduce the memory usage of the reverse pass, `InterpolatingAdjoint` includes a checkpointing
feature. If `sol` is `dense`, checkpointing is ignored and the continuous solution is used for
calculating `u(t)` at arbitrary time points. If `checkpointing=true` and `sol` is not `dense`,
then dense intervals between `sol.t[i]` and `sol.t[i+1]` are reconstructed on-demand for calculating
`u(t)` at arbitrary time points. This reduces the total memory requirement to only the cost of
holding the dense solution over the largest time interval (in terms of number of required steps).
The total compute cost is no more than double the original forward compute cost.

## SciMLProblem Support

This `sensealg` only supports `ODEProblem`s, `SDEProblem`s, and `RODEProblem`s. This `sensealg`
supports callbacks (events).

### Disclaimer for `SDEProblem`s

The runtime of this algorithm is in O(n^2) for diagonal-noise SDEs until issue #854 is solved.

## References

Rackauckas, C. and Ma, Y. and Martensen, J. and Warner, C. and Zubov, K. and Supekar,
R. and Skinner, D. and Ramadhana, A. and Edelman, A., Universal Differential Equations
for Scientific Machine Learning,	arXiv:2001.04385

Hindmarsh, A. C. and Brown, P. N. and Grant, K. E. and Lee, S. L. and Serban, R.
and Shumaker, D. E. and Woodward, C. S., SUNDIALS: Suite of nonlinear and
differential/algebraic equation solvers, ACM Transactions on Mathematical
Software (TOMS), 31, pp:363–396 (2005)

Rackauckas, C. and Ma, Y. and Dixit, V. and Guo, X. and Innes, M. and Revels, J.
and Nyberg, J. and Ivaturi, V., A comparison of automatic differentiation and
continuous sensitivity analysis for derivatives of differential equation solutions,
arXiv:1812.01892
"""
struct InterpolatingAdjoint{CS, AD, FDT, VJP} <:
    AbstractAdjointSensitivityAlgorithm{CS, AD, FDT}
    autojacvec::VJP
    checkpointing::Bool
    noisemixing::Bool
end
Base.@pure function InterpolatingAdjoint(;
        chunk_size = 0, autodiff = true,
        diff_type = Val{:central},
        autojacvec = nothing,
        checkpointing = false, noisemixing = false
    )
    InterpolatingAdjoint{chunk_size, autodiff, diff_type, typeof(autojacvec)}(
        autojacvec,
        checkpointing,
        noisemixing
    )
end

function setvjp(
        sensealg::InterpolatingAdjoint{CS, AD, FDT},
        vjp
    ) where {CS, AD, FDT}
    return InterpolatingAdjoint{CS, AD, FDT, typeof(vjp)}(
        vjp, sensealg.checkpointing,
        sensealg.noisemixing
    )
end

"""
```julia
QuadratureAdjoint{CS, AD, FDT, VJP} <: AbstractAdjointSensitivityAlgorithm{CS, AD, FDT}
```

An implementation of adjoint sensitivity analysis which develops a full
continuous solution of the reverse solve in order to perform a post-ODE
quadrature. This method requires the dense solution and will ignore
saving arguments during the gradient calculation. The tolerances in the
constructor control the inner quadrature.

This method is O(n^3 + p) for stiff / implicit equations (as opposed to the
O((n+p)^3) scaling of BacksolveAdjoint and InterpolatingAdjoint), and thus
is much more compute efficient. However, it requires holding a dense reverse
pass and is thus memory intensive.

## Constructor

```julia
QuadratureAdjoint(; chunk_size = 0, autodiff = true,
    diff_type = Val{:central},
    autojacvec = nothing, abstol = 1e-6,
    reltol = 1e-3)
```

## Keyword Arguments

  - `autodiff`: Use automatic differentiation for constructing the Jacobian
    if the Jacobian needs to be constructed.  Defaults to `true`.

  - `chunk_size`: Chunk size for forward-mode differentiation if full Jacobians are
    built (`autojacvec=false` and `autodiff=true`). Default is `0` for automatic
    choice of chunk size.
  - `diff_type`: The method used by FiniteDiff.jl for constructing the Jacobian
    if the full Jacobian is required with `autodiff=false`.
  - `autojacvec`: Calculate the vector-Jacobian product (`J'*v`) via automatic
    differentiation with special seeding. The total set of choices are:

      + `nothing`: uses an automatic algorithm to automatically choose the vjp.
        This is the default and recommended for most users.
      + `false`: the Jacobian is constructed via FiniteDiff.jl
      + `true`: the Jacobian is constructed via ForwardDiff.jl
      + `TrackerVJP`: Uses Tracker.jl for the vjp.
      + `ZygoteVJP`: Uses Zygote.jl for the vjp.
      + `EnzymeVJP`: Uses Enzyme.jl for the vjp.
      + `ReactantVJP`: Uses Reactant.jl-compiled Enzyme.jl for the vjp.
        Requires `using Reactant`.
      + `ReverseDiffVJP(compile=false)`: Uses ReverseDiff.jl for the vjp. `compile`
        is a boolean for whether to precompile the tape, which should only be done
        if there are no branches (`if` or `while` statements) in the `f` function.
  - `abstol`: absolute tolerance for the quadrature calculation
  - `reltol`: relative tolerance for the quadrature calculation

For more details on the vjp choices, please consult the sensitivity algorithms
documentation page or the docstrings of the vjp types.

## SciMLProblem Support

This `sensealg` only supports `ODEProblem`s. This `sensealg` supports events (callbacks).

## References

Rackauckas, C. and Ma, Y. and Martensen, J. and Warner, C. and Zubov, K. and Supekar,
R. and Skinner, D. and Ramadhana, A. and Edelman, A., Universal Differential Equations
for Scientific Machine Learning,	arXiv:2001.04385

Hindmarsh, A. C. and Brown, P. N. and Grant, K. E. and Lee, S. L. and Serban, R.
and Shumaker, D. E. and Woodward, C. S., SUNDIALS: Suite of nonlinear and
differential/algebraic equation solvers, ACM Transactions on Mathematical
Software (TOMS), 31, pp:363–396 (2005)

Rackauckas, C. and Ma, Y. and Dixit, V. and Guo, X. and Innes, M. and Revels, J.
and Nyberg, J. and Ivaturi, V., A comparison of automatic differentiation and
continuous sensitivity analysis for derivatives of differential equation solutions,
arXiv:1812.01892

Kim, S., Ji, W., Deng, S., Ma, Y., & Rackauckas, C. (2021). Stiff neural ordinary
differential equations. Chaos: An Interdisciplinary Journal of Nonlinear Science, 31(9), 093122.
"""
struct QuadratureAdjoint{CS, AD, FDT, VJP} <:
    AbstractAdjointSensitivityAlgorithm{CS, AD, FDT}
    autojacvec::VJP
    abstol::Float64
    reltol::Float64
end
Base.@pure function QuadratureAdjoint(;
        chunk_size = 0, autodiff = true,
        diff_type = Val{:central},
        autojacvec = nothing, abstol = 1.0e-6,
        reltol = 1.0e-3
    )
    QuadratureAdjoint{chunk_size, autodiff, diff_type, typeof(autojacvec)}(
        autojacvec,
        abstol, reltol
    )
end

function setvjp(sensealg::QuadratureAdjoint{CS, AD, FDT}, vjp) where {CS, AD, FDT}
    return QuadratureAdjoint{CS, AD, FDT, typeof(vjp)}(
        vjp, sensealg.abstol,
        sensealg.reltol
    )
end

"""
```julia
GaussAdjoint{CS, AD, FDT, VJP} <: AbstractAdjointSensitivityAlgorithm{CS, AD, FDT}
```

An implementation of adjoint sensitivity analysis which solves the quadrature
during the reverse solve with a callback, thus not requiring a dense adjoint
solution. This method requires the dense forward solution and will ignore
saving arguments during the gradient calculation. The tolerances in the
constructor control the inner quadrature.

This method is O(n^3 + p) for stiff / implicit equations (as opposed to the
O((n+p)^3) scaling of BacksolveAdjoint and InterpolatingAdjoint), and thus
is much more compute efficient. It also does not requires holding a dense reverse
pass and is thus memory efficient.

## Constructor

```julia
GaussAdjoint(; chunk_size = 0, autodiff = true,
    diff_type = Val{:central},
    autojacvec = nothing,
    checkpointing = false)
```

## Keyword Arguments

  - `autodiff`: Use automatic differentiation for constructing the Jacobian
    if the Jacobian needs to be constructed.  Defaults to `true`.

  - `chunk_size`: Chunk size for forward-mode differentiation if full Jacobians are
    built (`autojacvec=false` and `autodiff=true`). Default is `0` for automatic
    choice of chunk size.
  - `diff_type`: The method used by FiniteDiff.jl for constructing the Jacobian
    if the full Jacobian is required with `autodiff=false`.
  - `autojacvec`: Calculate the vector-Jacobian product (`J'*v`) via automatic
    differentiation with special seeding. The total set of choices are:

      + `nothing`: uses an automatic algorithm to automatically choose the vjp.
        This is the default and recommended for most users.
      + `false`: the Jacobian is constructed via FiniteDiff.jl
      + `true`: the Jacobian is constructed via ForwardDiff.jl
      + `TrackerVJP`: Uses Tracker.jl for the vjp.
      + `ZygoteVJP`: Uses Zygote.jl for the vjp.
      + `EnzymeVJP`: Uses Enzyme.jl for the vjp.
      + `ReactantVJP`: Uses Reactant.jl-compiled Enzyme.jl for the vjp.
        Requires `using Reactant`.
      + `ReverseDiffVJP(compile=false)`: Uses ReverseDiff.jl for the vjp. `compile`
        is a boolean for whether to precompile the tape, which should only be done
        if there are no branches (`if` or `while` statements) in the `f` function.
  - `checkpointing`: whether checkpointing is enabled for the reverse pass. Defaults
    to `false`.

For more details on the vjp choices, please consult the sensitivity algorithms
documentation page or the docstrings of the vjp types.

## SciMLProblem Support

This `sensealg` only supports `ODEProblem`s. This `sensealg` supports events (callbacks).

## References

Rackauckas, C. and Ma, Y. and Martensen, J. and Warner, C. and Zubov, K. and Supekar,
R. and Skinner, D. and Ramadhana, A. and Edelman, A., Universal Differential Equations
for Scientific Machine Learning,	arXiv:2001.04385

Hindmarsh, A. C. and Brown, P. N. and Grant, K. E. and Lee, S. L. and Serban, R.
and Shumaker, D. E. and Woodward, C. S., SUNDIALS: Suite of nonlinear and
differential/algebraic equation solvers, ACM Transactions on Mathematical
Software (TOMS), 31, pp:363–396 (2005)

Rackauckas, C. and Ma, Y. and Dixit, V. and Guo, X. and Innes, M. and Revels, J.
and Nyberg, J. and Ivaturi, V., A comparison of automatic differentiation and
continuous sensitivity analysis for derivatives of differential equation solutions,
arXiv:1812.01892

Kim, S., Ji, W., Deng, S., Ma, Y., & Rackauckas, C. (2021). Stiff neural ordinary
differential equations. Chaos: An Interdisciplinary Journal of Nonlinear Science, 31(9), 093122.
"""
struct GaussAdjoint{CS, AD, FDT, VJP} <:
    AbstractAdjointSensitivityAlgorithm{CS, AD, FDT}
    autojacvec::VJP
    checkpointing::Bool
end
Base.@pure function GaussAdjoint(;
        chunk_size = 0, autodiff = true,
        diff_type = Val{:central},
        autojacvec = nothing,
        checkpointing = false
    )
    GaussAdjoint{chunk_size, autodiff, diff_type, typeof(autojacvec)}(
        autojacvec, checkpointing
    )
end

function setvjp(sensealg::GaussAdjoint{CS, AD, FDT}, vjp) where {CS, AD, FDT}
    return GaussAdjoint{CS, AD, FDT, typeof(vjp)}(vjp, sensealg.checkpointing)
end

"""
```julia
GaussKronrodAdjoint{CS, AD, FDT, VJP} <: AbstractAdjointSensitivityAlgorithm{CS, AD, FDT}
```

Uses Gauss-Kronrod quadrature instead of Gauss quadrature, to achieve
error control.

This method is O(n^3 + p) for stiff / implicit equations (as opposed to the
O((n+p)^3) scaling of BacksolveAdjoint and InterpolatingAdjoint), and thus
is much more compute efficient. It also does not requires holding a dense reverse
pass and is thus memory efficient.

## Constructor

```julia
GaussKronrodAdjoint(; chunk_size = 0, autodiff = true,
    diff_type = Val{:central},
    autojacvec = nothing,
    checkpointing = false)
```

## Keyword Arguments

  - `autodiff`: Use automatic differentiation for constructing the Jacobian
    if the Jacobian needs to be constructed.  Defaults to `true`.

  - `chunk_size`: Chunk size for forward-mode differentiation if full Jacobians are
    built (`autojacvec=false` and `autodiff=true`). Default is `0` for automatic
    choice of chunk size.
  - `diff_type`: The method used by FiniteDiff.jl for constructing the Jacobian
    if the full Jacobian is required with `autodiff=false`.
  - `autojacvec`: Calculate the vector-Jacobian product (`J'*v`) via automatic
    differentiation with special seeding. The total set of choices are:

      + `nothing`: uses an automatic algorithm to automatically choose the vjp.
        This is the default and recommended for most users.
      + `false`: the Jacobian is constructed via FiniteDiff.jl
      + `true`: the Jacobian is constructed via ForwardDiff.jl
      + `TrackerVJP`: Uses Tracker.jl for the vjp.
      + `ZygoteVJP`: Uses Zygote.jl for the vjp.
      + `EnzymeVJP`: Uses Enzyme.jl for the vjp.
      + `ReactantVJP`: Uses Reactant.jl-compiled Enzyme.jl for the vjp.
        Requires `using Reactant`.
      + `ReverseDiffVJP(compile=false)`: Uses ReverseDiff.jl for the vjp. `compile`
        is a boolean for whether to precompile the tape, which should only be done
        if there are no branches (`if` or `while` statements) in the `f` function.
  - `checkpointing`: whether checkpointing is enabled for the reverse pass. Defaults
    to `false`.

For more details on the vjp choices, please consult the sensitivity algorithms
documentation page or the docstrings of the vjp types.

## SciMLProblem Support

This `sensealg` only supports `ODEProblem`s. This `sensealg` supports events (callbacks).

## References

Rackauckas, C. and Ma, Y. and Martensen, J. and Warner, C. and Zubov, K. and Supekar,
R. and Skinner, D. and Ramadhana, A. and Edelman, A., Universal Differential Equations
for Scientific Machine Learning,	arXiv:2001.04385

Hindmarsh, A. C. and Brown, P. N. and Grant, K. E. and Lee, S. L. and Serban, R.
and Shumaker, D. E. and Woodward, C. S., SUNDIALS: Suite of nonlinear and
differential/algebraic equation solvers, ACM Transactions on Mathematical
Software (TOMS), 31, pp:363–396 (2005)

Rackauckas, C. and Ma, Y. and Dixit, V. and Guo, X. and Innes, M. and Revels, J.
and Nyberg, J. and Ivaturi, V., A comparison of automatic differentiation and
continuous sensitivity analysis for derivatives of differential equation solutions,
arXiv:1812.01892

Kim, S., Ji, W., Deng, S., Ma, Y., & Rackauckas, C. (2021). Stiff neural ordinary
differential equations. Chaos: An Interdisciplinary Journal of Nonlinear Science, 31(9), 093122.
"""
struct GaussKronrodAdjoint{CS, AD, FDT, VJP} <:
    AbstractAdjointSensitivityAlgorithm{CS, AD, FDT}
    autojacvec::VJP
    checkpointing::Bool
end
Base.@pure function GaussKronrodAdjoint(;
        chunk_size = 0, autodiff = true,
        diff_type = Val{:central},
        autojacvec = nothing,
        checkpointing = false
    )
    GaussKronrodAdjoint{chunk_size, autodiff, diff_type, typeof(autojacvec)}(
        autojacvec, checkpointing
    )
end

function setvjp(sensealg::GaussKronrodAdjoint{CS, AD, FDT}, vjp) where {
        CS, AD, FDT,
    }
    return GaussKronrodAdjoint{CS, AD, FDT, typeof(vjp)}(vjp, sensealg.checkpointing)
end

# Supertype of gauss methods, internal
AbstractGAdjoint = Union{GaussAdjoint, GaussKronrodAdjoint}

"""
```julia
TrackerAdjoint <: AbstractAdjointSensitivityAlgorithm{nothing, true, nothing}
```

An implementation of discrete adjoint sensitivity analysis
using the Tracker.jl tracing-based AD. Supports in-place functions through
an Array of Structs formulation, and supports out of place through struct of
arrays.

## Constructor

```julia
TrackerAdjoint()
```

## SciMLProblem Support

This `sensealg` supports any `DEProblem` if the algorithm is `SciMLBase.isautodifferentiable`
Compatible with a limited subset of `AbstractArray` types for `u0`, including `CuArrays`.

!!! warning

    TrackerAdjoint is incompatible with Stiff ODE solvers using forward-mode automatic
    differentiation for the Jacobians. Thus, for example, `TRBDF2()` will error. Instead,
    use `autodiff=AutoFiniteDiff()`, i.e. `TRBDF2(autodiff=AutoFiniteDiff())`. This will only remove the
    forward-mode automatic differentiation of the Jacobian construction, not the reverse-mode
    AD usage, and thus performance will still be nearly the same, though Jacobian accuracy
    may suffer which could cause more steps to be required.
"""
struct TrackerAdjoint <: AbstractAdjointSensitivityAlgorithm{nothing, true, nothing} end

"""
```julia
MooncakeAdjoint <: AbstractAdjointSensitivityAlgorithm{nothing, true, nothing}
```

An implementation of discrete adjoint sensitivity analysis
using the Mooncake.jl direct differentiation.

!!! warning

    This is currently experimental and supports only explicit solvers. It will
    support all solvers in the future.

## Constructor

```julia
MooncakeAdjoint()
```

## SciMLProblem Support

This `sensealg` supports any `DEProblem` if the algorithm is `SciMLBase.isautodifferentiable`
"""
struct MooncakeAdjoint <: AbstractAdjointSensitivityAlgorithm{nothing, true, nothing} end

"""
```julia
ReverseDiffAdjoint <: AbstractAdjointSensitivityAlgorithm{nothing, true, nothing}
```

An implementation of discrete adjoint sensitivity analysis using the ReverseDiff.jl
tracing-based AD. Supports in-place functions through an Array of Structs formulation,
and supports out of place through struct of arrays.

## Constructor

```julia
ReverseDiffAdjoint()
```

## SciMLProblem Support

This `sensealg` supports any `DEProblem` if the algorithm is `SciMLBase.isautodifferentiable`.
Requires that the state variables are CPU-based `Array` types.
"""
struct ReverseDiffAdjoint <: AbstractAdjointSensitivityAlgorithm{nothing, true, nothing} end

"""
ZygoteAdjoint <: AbstractAdjointSensitivityAlgorithm{nothing,true,nothing}

An implementation of discrete adjoint sensitivity analysis
using the Zygote.jl source-to-source AD directly on the differential equation
solver.

!!! warning

    This is only supports SimpleDiffEq.jl solvers due to limitations of Enzyme.

## Constructor

```julia
ZygoteAdjoint()
```

## SciMLProblem Support

Currently fails on almost every solver.
"""
struct ZygoteAdjoint <: AbstractAdjointSensitivityAlgorithm{nothing, true, nothing} end

"""
EnzymeAdjoint <: AbstractAdjointSensitivityAlgorithm{nothing,true,nothing}

An implementation of discrete adjoint sensitivity analysis
using the Enzyme.jl source-to-source AD directly on the differential equation
solver.

!!! warning

    This is currently experimental and supports only explicit solvers. It will
    support all solvers in the future.

## Constructor

```julia
EnzymeAdjoint(mode = nothing)
```

## Arguments

  - `mode::M` determines the autodiff mode (forward or reverse). It can be:

      + an object subtyping `EnzymeCore.Mode` (like `EnzymeCore.Forward` or `EnzymeCore.Reverse`) if a specific mode is required
      + `nothing` to choose the best mode automatically

## SciMLProblem Support

Currently fails on almost every solver.
"""
struct EnzymeAdjoint{M <: Union{Nothing, Enzyme.EnzymeCore.Mode}} <:
    AbstractAdjointSensitivityAlgorithm{nothing, true, nothing}
    mode::M
    EnzymeAdjoint(mode = nothing) = new{typeof(mode)}(mode)
end

"""
```julia
ForwardLSS{CS, AD, FDT, RType, gType} <: AbstractShadowingSensitivityAlgorithm{CS, AD, FDT}
```

An implementation of the discrete, forward-mode
[least squares shadowing](https://arxiv.org/abs/1204.0159) (LSS) method. LSS replaces
the ill-conditioned initial value problem (`ODEProblem`) for chaotic systems by a
well-conditioned least-squares problem. This allows for computing sensitivities of
long-time averaged quantities with respect to the parameters of the `ODEProblem`. The
computational cost of LSS scales as (number of states x number of time steps). Converges
to the correct sensitivity at a rate of `T^(-1/2)`, where `T` is the time of the trajectory.
See `NILSS()` and `NILSAS()` for a more efficient non-intrusive formulation.

## Constructor

```julia
ForwardLSS(;
    chunk_size = 0, autodiff = true,
    diff_type = Val{:central},
    LSSregularizer = TimeDilation(10.0, 0.0, 0.0),
    g = nothing)
```

## Keyword Arguments

  - `autodiff`: Use automatic differentiation for constructing the Jacobian
    if the Jacobian needs to be constructed.  Defaults to `true`.

  - `chunk_size`: Chunk size for forward-mode differentiation if full Jacobians are
    built (`autojacvec=false` and `autodiff=true`). Default is `0` for automatic
    choice of chunk size.
  - `diff_type`: The method used by FiniteDiff.jl for constructing the Jacobian
    if the full Jacobian is required with `autodiff=false`.
  - `LSSregularizer`: Using `LSSregularizer`, one can choose between three different
    regularization routines. The default choice is `TimeDilation(10.0,0.0,0.0)`.

      + `CosWindowing()`: cos windowing of the time grid, i.e. the time grid (saved
        time steps) is transformed using a cosine.
      + `Cos2Windowing()`: cos^2 windowing of the time grid.
      + `TimeDilation(alpha::Number,t0skip::Number,t1skip::Number)`: Corresponds to
        a time dilation. `alpha` controls the weight. `t0skip` and `t1skip` indicate
        the times truncated at the beginning and end of the trajectory, respectively.
  - `g`: instantaneous objective function of the long-time averaged objective.

## SciMLProblem Support

This `sensealg` only supports `ODEProblem`s. This `sensealg` does not support
events (callbacks). This `sensealg` assumes that the objective is a long-time averaged
quantity and ergodic, i.e. the time evolution of the system behaves qualitatively the
same over infinite time independent of the specified initial conditions, such that only
the sensitivity with respect to the parameters is of interest.

## References

Wang, Q., Hu, R., and Blonigan, P. Least squares shadowing sensitivity analysis of
chaotic limit cycle oscillations. Journal of Computational Physics, 267, 210-224 (2014).

Wang, Q., Convergence of the Least Squares Shadowing Method for Computing Derivative of Ergodic
Averages, SIAM Journal on Numerical Analysis, 52, 156–170 (2014).

Blonigan, P., Gomez, S., Wang, Q., Least Squares Shadowing for sensitivity analysis of turbulent
fluid flows, in: 52nd Aerospace Sciences Meeting, 1–24 (2014).
"""
struct ForwardLSS{CS, AD, FDT, RType, gType} <:
    AbstractShadowingSensitivityAlgorithm{CS, AD, FDT}
    LSSregularizer::RType
    g::gType
end
Base.@pure function ForwardLSS(;
        chunk_size = 0, autodiff = true,
        diff_type = Val{:central},
        LSSregularizer = TimeDilation(10.0, 0.0, 0.0),
        g = nothing
    )
    ForwardLSS{chunk_size, autodiff, diff_type, typeof(LSSregularizer), typeof(g)}(
        LSSregularizer,
        g
    )
end

"""
```julia
AdjointLSS{CS, AD, FDT, RType, gType} <: AbstractShadowingSensitivityAlgorithm{CS, AD, FDT}
```

An implementation of the discrete, adjoint-mode
[least square shadowing](https://arxiv.org/abs/1204.0159) method. LSS replaces
the ill-conditioned initial value problem (`ODEProblem`) for chaotic systems by a
well-conditioned least-squares problem. This allows for computing sensitivities of
long-time averaged quantities with respect to the parameters of the `ODEProblem`. The
computational cost of LSS scales as (number of states x number of time steps). Converges
to the correct sensitivity at a rate of `T^(-1/2)`, where `T` is the time of the trajectory.
See `NILSS()` and `NILSAS()` for a more efficient non-intrusive formulation.

## Constructor

```julia
AdjointLSS(;
    chunk_size = 0, autodiff = true,
    diff_type = Val{:central},
    LSSRegularizer = TimeDilation(10.0, 0.0, 0.0),
    g = nothing)
```

## Keyword Arguments

  - `autodiff`: Use automatic differentiation for constructing the Jacobian
    if the Jacobian needs to be constructed.  Defaults to `true`.

  - `chunk_size`: Chunk size for forward-mode differentiation if full Jacobians are
    built (`autojacvec=false` and `autodiff=true`). Default is `0` for automatic
    choice of chunk size.
  - `diff_type`: The method used by FiniteDiff.jl for constructing the Jacobian
    if the full Jacobian is required with `autodiff=false`.
  - `LSSregularizer`: Using `LSSregularizer`, one can choose between different
    regularization routines. The default choice is `TimeDilation(10.0,0.0,0.0)`.

      + `TimeDilation(alpha::Number,t0skip::Number,t1skip::Number)`: Corresponds to
        a time dilation. `alpha` controls the weight. `t0skip` and `t1skip` indicate
        the times truncated at the beginning and end of the trajectory, respectively.
        The default value for `t0skip` and `t1skip` is `zero(alpha)`.
  - `g`: instantaneous objective function of the long-time averaged objective.

## SciMLProblem Support

This `sensealg` only supports `ODEProblem`s. This `sensealg` does not support
events (callbacks). This `sensealg` assumes that the objective is a long-time averaged
quantity and ergodic, i.e. the time evolution of the system behaves qualitatively the
same over infinite time independent of the specified initial conditions, such that only
the sensitivity with respect to the parameters is of interest.

## References

Wang, Q., Hu, R., and Blonigan, P. Least squares shadowing sensitivity analysis of
chaotic limit cycle oscillations. Journal of Computational Physics, 267, 210-224 (2014).
"""
struct AdjointLSS{CS, AD, FDT, RType, gType} <:
    AbstractShadowingSensitivityAlgorithm{CS, AD, FDT}
    LSSregularizer::RType
    g::gType
end
Base.@pure function AdjointLSS(;
        chunk_size = 0, autodiff = true,
        diff_type = Val{:central},
        LSSregularizer = TimeDilation(10.0, 0.0, 0.0),
        g = nothing
    )
    AdjointLSS{chunk_size, autodiff, diff_type, typeof(LSSregularizer), typeof(g)}(
        LSSregularizer,
        g
    )
end

abstract type AbstractLSSregularizer end
abstract type AbstractCosWindowing <: AbstractLSSregularizer end
struct CosWindowing <: AbstractCosWindowing end
struct Cos2Windowing <: AbstractCosWindowing end

"""
```julia
TimeDilation{T1 <: Number} <: AbstractLSSregularizer
```

A regularization method for `LSS`. See `?LSS` for
additional information and other methods.

## Constructor

```julia
TimeDilation(alpha;
    t0skip = zero(alpha),
    t1skip = zero(alpha))
```
"""
struct TimeDilation{T1 <: Number} <: AbstractLSSregularizer
    alpha::T1 # alpha: weight of the time dilation term in LSS.
    t0skip::T1
    t1skip::T1
end
function TimeDilation(alpha, t0skip = zero(alpha), t1skip = zero(alpha))
    return TimeDilation{typeof(alpha)}(alpha, t0skip, t1skip)
end
"""
```
struct NILSS{CS,AD,FDT,RNG,nType,gType} <: AbstractShadowingSensitivityAlgorithm{CS,AD,FDT}
```

An implementation of the forward-mode, continuous
[non-intrusive least squares shadowing](https://arxiv.org/abs/1611.00880) method. `NILSS`
allows for computing sensitivities of long-time averaged quantities with respect to the
parameters of an `ODEProblem` by constraining the computation to the unstable subspace.
`NILSS` employs the continuous-time `ForwardSensitivity` method as tangent solver. To
avoid an exponential blow-up of the (homogeneous and inhomogeneous) tangent solutions,
the trajectory should be divided into sufficiently small segments, where the tangent solutions
are rescaled on the interfaces. The computational and memory cost of NILSS scale with
the number of unstable (positive) Lyapunov exponents (instead of the number of states, as
in the LSS method). `NILSS` avoids the explicit construction of the Jacobian at each time
step, and thus should generally be preferred (for large system sizes) over `ForwardLSS`.

## Constructor

```julia
NILSS(nseg, nstep; nus = nothing,
    rng = Xorshifts.Xoroshiro128Plus(rand(UInt64)),
    chunk_size = 0, autodiff = true,
    diff_type = Val{:central},
    autojacvec = autodiff,
    g = nothing)
```

## Arguments

  - `nseg`: Number of segments on full time interval on the attractor.
  - `nstep`: number of steps on each segment.

## Keyword Arguments

  - `nus`: Dimension of the unstable subspace. Default is `nothing`. `nus` must be
    smaller or equal to the state dimension (`length(u0)`). With the default choice,
    `nus = length(u0) - 1` will be set at compile time.
  - `rng`: (Pseudo) random number generator. Used for initializing the homogeneous
    tangent states (`w`). Default is `Xorshifts.Xoroshiro128Plus(rand(UInt64))`.
  - `autodiff`: Use automatic differentiation in the internal sensitivity algorithm
    computations. Default is `true`.
  - `chunk_size`: Chunk size for forward mode differentiation if full Jacobians are
    built (`autojacvec=false` and `autodiff=true`). Default is `0` for automatic
    choice of chunk size.
  - `autojacvec`: Calculate the Jacobian-vector product via automatic
    differentiation with special seeding.
  - `diff_type`: The method used by FiniteDiff.jl for constructing the Jacobian
    if the full Jacobian is required with `autodiff=false`.
  - `g`: instantaneous objective function of the long-time averaged objective.

## SciMLProblem Support

This `sensealg` only supports `ODEProblem`s. This `sensealg` does not support
events (callbacks). This `sensealg` assumes that the objective is a long-time averaged
quantity and ergodic, i.e. the time evolution of the system behaves qualitatively the
same over infinite time independent of the specified initial conditions, such that only
the sensitivity with respect to the parameters is of interest.

## References

Ni, A., Blonigan, P. J., Chater, M., Wang, Q., Zhang, Z., Sensitivity analy-
sis on chaotic dynamical system by Non-Intrusive Least Square Shadowing
(NI-LSS), in: 46th AIAA Fluid Dynamics Conference, AIAA AVIATION Forum (AIAA 2016-4399),
American Institute of Aeronautics and Astronautics, 1–16 (2016).

Ni, A., and Wang, Q. Sensitivity analysis on chaotic dynamical systems by Non-Intrusive
Least Squares Shadowing (NILSS). Journal of Computational Physics 347, 56-77 (2017).
"""
struct NILSS{CS, AD, FDT, RNG, nType, gType} <:
    AbstractShadowingSensitivityAlgorithm{CS, AD, FDT}
    rng::RNG
    nseg::Int
    nstep::Int
    nus::nType
    autojacvec::Bool
    g::gType
end
Base.@pure function NILSS(
        nseg, nstep; nus = nothing,
        rng = Xorshifts.Xoroshiro128Plus(rand(UInt64)),
        chunk_size = 0, autodiff = true,
        diff_type = Val{:central},
        autojacvec = autodiff,
        g = nothing
    )
    NILSS{chunk_size, autodiff, diff_type, typeof(rng), typeof(nus), typeof(g)}(
        rng, nseg,
        nstep, nus,
        autojacvec,
        g
    )
end

"""
```julia
NILSAS{CS, AD, FDT, RNG, SENSE, gType} <: AbstractShadowingSensitivityAlgorithm{CS, AD, FDT}
```

An implementation of the adjoint-mode, continuous
[non-intrusive adjoint least squares shadowing](https://arxiv.org/abs/1801.08674) method.
`NILSAS` allows for computing sensitivities of long-time averaged quantities with respect
to the parameters of an `ODEProblem` by constraining the computation to the unstable subspace.
`NILSAS` employs SciMLSensitivity.jl's continuous adjoint sensitivity methods on each segment
to compute (homogeneous and inhomogeneous) adjoint solutions. To avoid an exponential blow-up
of the adjoint solutions, the trajectory should be divided into sufficiently small segments,
where the adjoint solutions are rescaled on the interfaces. The computational and memory cost
of NILSAS scale with the number of unstable, adjoint Lyapunov exponents (instead of the number
of states as in the LSS method). `NILSAS` avoids the explicit construction of the Jacobian at
each time step, and thus should generally be preferred (for large system sizes) over `AdjointLSS`.
`NILSAS` is favorable over `NILSS` for many parameters because NILSAS computes the gradient
with respect to multiple parameters with negligible additional cost.

## Constructor

```julia
NILSAS(nseg, nstep, M = nothing; rng = Xorshifts.Xoroshiro128Plus(rand(UInt64)),
    adjoint_sensealg = BacksolveAdjoint(autojacvec = ReverseDiffVJP()),
    chunk_size = 0, autodiff = true,
    diff_type = Val{:central},
    g = nothing)
```

## Arguments

  - `nseg`: Number of segments on full time interval on the attractor.
  - `nstep`: number of steps on each segment.
  - `M`: number of homogeneous adjoint solutions. This number must be bigger or equal
    than the number of (positive, adjoint) Lyapunov exponents. Default is `nothing`.

## Keyword Arguments

  - `rng`: (Pseudo) random number generator. Used for initializing the terminate
    conditions of the homogeneous adjoint states (`w`). Default is `Xorshifts.Xoroshiro128Plus(rand(UInt64))`.

  - `adjoint_sensealg`: Continuous adjoint sensitivity method to compute homogeneous
    and inhomogeneous adjoint solutions on each segment. Default is `BacksolveAdjoint(autojacvec=ReverseDiffVJP())`.

      + `autojacvec`: Calculate the vector-Jacobian product (`J'*v`) via automatic
        differentiation with special seeding. The default is `true`. The total set
        of choices are:

          * `false`: the Jacobian is constructed via FiniteDiff.jl
          * `true`: the Jacobian is constructed via ForwardDiff.jl
          * `TrackerVJP`: Uses Tracker.jl for the vjp.
          * `ZygoteVJP`: Uses Zygote.jl for the vjp.
          * `EnzymeVJP`: Uses Enzyme.jl for the vjp.
          * `ReactantVJP`: Uses Reactant.jl-compiled Enzyme.jl for the vjp.
            Requires `using Reactant`.
          * `ReverseDiffVJP(compile=false)`: Uses ReverseDiff.jl for the vjp. `compile`
            is a boolean for whether to precompile the tape, which should only be done
            if there are no branches (`if` or `while` statements) in the `f` function.
  - `autodiff`: Use automatic differentiation for constructing the Jacobian
    if the Jacobian needs to be constructed.  Defaults to `true`.
  - `chunk_size`: Chunk size for forward-mode differentiation if full Jacobians are
    built (`autojacvec=false` and `autodiff=true`). Default is `0` for automatic
    choice of chunk size.
  - `diff_type`: The method used by FiniteDiff.jl for constructing the Jacobian
    if the full Jacobian is required with `autodiff=false`.
  - `g`: instantaneous objective function of the long-time averaged objective.

## SciMLProblem Support

This `sensealg` only supports `ODEProblem`s. This `sensealg` does not support
events (callbacks). This `sensealg` assumes that the objective is a long-time averaged
quantity and ergodic, i.e. the time evolution of the system behaves qualitatively the
same over infinite time independent of the specified initial conditions, such that only
the sensitivity with respect to the parameters is of interest.

## References

Ni, A., and Talnikar, C., Adjoint sensitivity analysis on chaotic dynamical systems
by Non-Intrusive Least Squares Adjoint Shadowing (NILSAS). Journal of Computational
Physics 395, 690-709 (2019).
"""
struct NILSAS{CS, AD, FDT, RNG, SENSE, gType} <:
    AbstractShadowingSensitivityAlgorithm{CS, AD, FDT}
    rng::RNG
    adjoint_sensealg::SENSE
    M::Int
    nseg::Int
    nstep::Int
    g::gType
end
Base.@pure function NILSAS(
        nseg, nstep, M = nothing;
        rng = Xorshifts.Xoroshiro128Plus(rand(UInt64)),
        adjoint_sensealg = BacksolveAdjoint(autojacvec = ReverseDiffVJP()),
        chunk_size = 0, autodiff = true,
        diff_type = Val{:central},
        g = nothing
    )

    # integer dimension of the unstable subspace
    M === nothing &&
        error("Please provide an `M` with `M >= nus + 1`, where nus is the number of unstable covariant Lyapunov vectors.")

    NILSAS{
        chunk_size,
        autodiff,
        diff_type,
        typeof(rng),
        typeof(adjoint_sensealg),
        typeof(g),
    }(
        rng, adjoint_sensealg, M,
        nseg, nstep, g
    )
end

"""
```julia
SteadyStateAdjoint{CS, AD, FDT, VJP, LS} <: AbstractAdjointSensitivityAlgorithm{CS, AD, FDT}
```

An implementation of the adjoint differentiation of a nonlinear solve. Uses the
implicit function theorem to directly compute the derivative of the solution to
``f(u,p) = 0`` with respect to `p`.

## Constructor

```julia
SteadyStateAdjoint(; chunk_size = 0, autodiff = true,
    diff_type = Val{:central},
    autojacvec = autodiff, linsolve = nothing)
```

## Keyword Arguments

  - `autodiff`: Use automatic differentiation for constructing the Jacobian
    if the Jacobian needs to be constructed.  Defaults to `true`.

  - `chunk_size`: Chunk size for forward-mode differentiation if full Jacobians are
    built (`autojacvec=false` and `autodiff=true`). Default is `0` for automatic
    choice of chunk size.
  - `diff_type`: The method used by FiniteDiff.jl for constructing the Jacobian
    if the full Jacobian is required with `autodiff=false`.
  - `autojacvec`: Calculate the vector-Jacobian product (`J'*v`) via automatic
    differentiation with special seeding. The total set of choices are:

      + `nothing`: uses an automatic algorithm to automatically choose the vjp.
        This is the default and recommended for most users.
      + `false`: the Jacobian is constructed via FiniteDiff.jl
      + `true`: the Jacobian is constructed via ForwardDiff.jl
      + `TrackerVJP`: Uses Tracker.jl for the vjp.
      + `ZygoteVJP`: Uses Zygote.jl for the vjp.
      + `EnzymeVJP`: Uses Enzyme.jl for the vjp.
      + `ReactantVJP`: Uses Reactant.jl-compiled Enzyme.jl for the vjp.
        Requires `using Reactant`.
      + `ReverseDiffVJP(compile=false)`: Uses ReverseDiff.jl for the vjp. `compile`
        is a boolean for whether to precompile the tape, which should only be done
        if there are no branches (`if` or `while` statements) in the `f` function.
  - `linsolve`: the linear solver used in the adjoint solve. Defaults to `nothing`,
    which uses a polyalgorithm to choose an efficient
    algorithm automatically.
  - `linsolve_kwargs`: keyword arguments to be passed to the linear solver.

For more details on the vjp choices, please consult the sensitivity algorithms
documentation page or the docstrings of the vjp types.

## References

Johnson, S. G., Notes on Adjoint Methods for 18.336, Online at
http://math.mit.edu/stevenj/18.336/adjoint.pdf (2007)
"""
struct SteadyStateAdjoint{CS, AD, FDT, VJP, LS, LK} <:
    AbstractAdjointSensitivityAlgorithm{CS, AD, FDT}
    autojacvec::VJP
    linsolve::LS
    linsolve_kwargs::LK
end

Base.@pure function SteadyStateAdjoint(;
        chunk_size = 0, autodiff = true,
        diff_type = Val{:central}, autojacvec = nothing, linsolve = nothing,
        linsolve_kwargs = (;)
    )
    return SteadyStateAdjoint{
        chunk_size, autodiff, diff_type, typeof(autojacvec),
        typeof(linsolve), typeof(linsolve_kwargs),
    }(autojacvec, linsolve, linsolve_kwargs)
end
function setvjp(
        sensealg::SteadyStateAdjoint{CS, AD, FDT, VJP, LS, LK},
        vjp
    ) where {CS, AD, FDT, VJP, LS, LK}
    return SteadyStateAdjoint{CS, AD, FDT, typeof(vjp), LS, LK}(
        vjp, sensealg.linsolve,
        sensealg.linsolve_kwargs
    )
end

abstract type VJPChoice end

"""
```julia
ZygoteVJP <: VJPChoice
```

Uses Zygote.jl to compute vector-Jacobian products. Tends to be the fastest VJP method if the
ODE/DAE/SDE/DDE is written with mostly vectorized  functions (like neural networks and other
layers from Flux.jl) and the `f` function is given out-of-place. If the `f` function is
in-place, then `Zygote.Buffer` arrays are used internally, which can greatly reduce the
performance of the VJP method.

## Constructor

```julia
ZygoteVJP(; allow_nothing = false)
```

Keyword arguments:

  - `allow_nothing`: whether `nothing`s should be implicitly converted to zeros. In Zygote,
    the derivative of a function with respect to `p` which does not use `p` in any possible
    calculation is given a derivative of `nothing` instead of zero. By default, this `nothing`
    is caught in order to throw an informative error message about a potentially unintentional
    misdefined function. However, if this was intentional, setting `allow_nothing=true` will
    remove the error message.
"""
struct ZygoteVJP <: VJPChoice
    allow_nothing::Bool
end
ZygoteVJP(; allow_nothing = false) = ZygoteVJP(allow_nothing)

"""
```julia
EnzymeVJP <: VJPChoice
```

Uses Enzyme.jl to compute vector-Jacobian products. Is the fastest VJP whenever applicable,
though Enzyme.jl currently has low coverage over the Julia programming language, for example
restricting the user's defined `f` function to not do things like require garbage collection
or calls to BLAS/LAPACK. However, mutation is supported, meaning that in-place `f` with
fully mutating non-allocating code will work with Enzyme (provided no high-level calls to C
like BLAS/LAPACK are used) and this will be the most efficient adjoint implementation.

## Constructor

```julia
EnzymeVJP(; chunksize = 0, mode = EnzymeCore.Reverse)
```

## Keyword Arguments

  - `chunksize`: the default chunk size for the temporary variables inside the vjp's right
    hand side definition. This is used for compatibility with ODE solves that default to using
    ForwardDiff.jl for the Jacobian of the stiff ODE solve, such as OrdinaryDiffEq.jl. This
    should be set to the maximum chunksize that can occur during an integration to preallocate
    the `DualCaches` for PreallocationTools.jl. It defaults to 0, using `ForwardDiff.pickchunksize`
    but could be decreased if this value is known to be lower to conserve memory.
  - `mode`: the parameterized Enzyme mode, default set to EnzymeCore.Reverse. Alternatively one
    may want to pass Enzyme.set_runtime_activity(Enzyme.Reverse)
"""
struct EnzymeVJP{Mode <: Enzyme.ReverseMode} <: VJPChoice
    chunksize::Int
    mode::Mode
end

EnzymeVJP(; chunksize = 0, mode = Enzyme.Reverse) = EnzymeVJP(chunksize, mode)

"""
```julia
TrackerVJP <: VJPChoice
```

Uses Tracker.jl to compute the vector-Jacobian products. If `f` is in-place,
then it uses a array of structs formulation to do scalarized reverse mode,
while if `f` is out-of-place then it uses an array-based reverse mode.

Not as efficient as `ReverseDiffVJP`, but supports GPUs when doing array-based
reverse mode.

## Constructor

```julia
TrackerVJP(; allow_nothing = false)
```

Keyword arguments:

  - `allow_nothing`: whether non-tracked values should be implicitly converted to zeros. In Tracker,
    the derivative of a function with respect to `p` which does not use `p` in any possible
    calculation is given an untracked return instead of zero. By default, this `nothing` Trackedness
    is caught in order to throw an informative error message about a potentially unintentional
    misdefined function. However, if this was intentional, setting `allow_nothing=true` will
    remove the error message.
"""
struct TrackerVJP <: VJPChoice
    allow_nothing::Bool
end
TrackerVJP(; allow_nothing = false) = TrackerVJP(allow_nothing)

"""
```julia
ReverseDiffVJP{compile} <: VJPChoice
```

Uses ReverseDiff.jl to compute the vector-Jacobian products. If `f` is in-place,
then it uses a array of structs formulation to do scalarized reverse mode,
while if `f` is out-of-place, then it uses an array-based reverse mode.

Usually, the fastest when scalarized operations exist in the f function
(like in scientific machine learning applications like Universal Differential Equations)
and the boolean compilation is enabled (i.e. ReverseDiffVJP(true)), if EnzymeVJP fails on
a given choice of `f`.

Does not support GPUs (CuArrays).

## Constructor

```julia
ReverseDiffVJP(compile = false)
```

## Keyword Arguments

  - `compile`: Whether to cache the compilation of the reverse tape. This heavily increases
    the performance of the method, but requires that the `f` function of the ODE/DAE/SDE/DDE
    has no branching.
"""
struct ReverseDiffVJP{compile} <: VJPChoice
    ReverseDiffVJP(compile = false) = new{compile}()
end

"""
```julia
MooncakeVJP <: VJPChoice
```

Uses Mooncake.jl to compute the vector-Jacobian products.

Does not support GPUs (CuArrays).

## Constructor

```julia
MooncakeVJP()
```
"""
struct MooncakeVJP <: VJPChoice end

"""
```julia
ReactantVJP <: VJPChoice
```

Uses Reactant.jl to compile Enzyme.jl's reverse-mode automatic differentiation into
XLA/HLO for hardware-accelerated vector-Jacobian product computation. The entire
Enzyme reverse pass is compiled by Reactant and can execute on CPU, GPU, or TPU.

Requires `using Reactant` to be loaded.

## Constructor

```julia
ReactantVJP(; allow_scalar = false)
```

## Keyword Arguments

- `allow_scalar`: If `true`, wraps Reactant compilation in `Reactant.@allowscalar`
  to permit scalar indexing during tracing. Required for ODE functions that use
  scalar indexing (e.g. `du[1] = ...`). Defaults to `false`.
"""
struct ReactantVJP <: VJPChoice
    allow_scalar::Bool
    ReactantVJP(; allow_scalar = false) = new(allow_scalar)
end

@inline convert_tspan(::ForwardDiffSensitivity{CS, CTS}) where {CS, CTS} = CTS
@inline convert_tspan(::Any) = nothing
@inline function alg_autodiff(
        alg::AbstractSensitivityAlgorithm{
            CS,
            AD,
            FDT,
        }
    ) where {
        CS,
        AD,
        FDT,
    }
    return AD
end
@inline function get_chunksize(
        alg::AbstractSensitivityAlgorithm{
            CS,
            AD,
            FDT,
        }
    ) where {
        CS,
        AD,
        FDT,
    }
    return CS
end
@inline function diff_type(
        alg::AbstractSensitivityAlgorithm{
            CS,
            AD,
            FDT,
        }
    ) where {
        CS,
        AD,
        FDT,
    }
    return FDT
end
@inline function get_jacvec(alg::AbstractSensitivityAlgorithm)
    return alg.autojacvec isa Bool ? alg.autojacvec : true
end
@inline function get_jacmat(alg::AbstractSensitivityAlgorithm)
    return alg.autojacmat isa Bool ? alg.autojacmat : true
end
@inline ischeckpointing(alg::AbstractSensitivityAlgorithm, sol = nothing) = false
@inline ischeckpointing(alg::InterpolatingAdjoint) = alg.checkpointing
@inline ischeckpointing(alg::InterpolatingAdjoint, sol) = alg.checkpointing || !sol.dense
@inline ischeckpointing(alg::GaussAdjoint) = alg.checkpointing
@inline ischeckpointing(alg::GaussAdjoint, sol) = alg.checkpointing || !sol.dense
@inline ischeckpointing(alg::GaussKronrodAdjoint) = alg.checkpointing
@inline ischeckpointing(alg::GaussKronrodAdjoint, sol) = alg.checkpointing || !sol.dense
@inline ischeckpointing(alg::BacksolveAdjoint, sol = nothing) = alg.checkpointing

@inline isnoisemixing(alg::AbstractSensitivityAlgorithm) = false
@inline isnoisemixing(alg::InterpolatingAdjoint) = alg.noisemixing
@inline isnoisemixing(alg::BacksolveAdjoint) = alg.noisemixing

@inline compile_tape(vjp::ReverseDiffVJP{compile}) where {compile} = compile
@inline compile_tape(autojacvec::Bool) = false

"""
    supports_functor_params(sensealg) -> Bool

Return `true` if the sensitivity algorithm supports Functors.jl parameter structs
without requiring conversion to an AbstractArray or SciMLStructures interface.

Only `GaussAdjoint` and `GaussKronrodAdjoint` support this, because they compute
parameter gradients via callbacks that work with structured types through `fmap`.
`QuadratureAdjoint` does not support functor params because it uses `quadgk` which
requires flat array types.
"""
supports_functor_params(::AbstractSensitivityAlgorithm) = false
supports_functor_params(::GaussAdjoint) = true
supports_functor_params(::GaussKronrodAdjoint) = true

"""
    supports_structured_vjp(autojacvec) -> Bool

Return `true` if the VJP backend can natively differentiate through structured
(non-array) parameter types like NamedTuples from Functors.jl.

When `false`, Functors.jl parameter structs are not supported and an informative
error will be thrown.
"""
supports_structured_vjp(::ZygoteVJP) = true
supports_structured_vjp(::EnzymeVJP) = true
supports_structured_vjp(::MooncakeVJP) = true
supports_structured_vjp(::ReactantVJP) = true
supports_structured_vjp(::ReverseDiffVJP) = false
supports_structured_vjp(::Bool) = false
supports_structured_vjp(::Nothing) = false

"""
```julia
ForwardDiffOverAdjoint{A} <: AbstractSecondOrderSensitivityAlgorithm{nothing, true, nothing}
```

ForwardDiff.jl over a choice of `sensealg` method for the adjoint.

## Constructor

```julia
ForwardDiffOverAdjoint(sensealg)
```

## SciMLProblem Support

This supports any SciMLProblem that the `sensealg` choice supports, provided the solver algorithm
is `SciMLBase.isautodifferentiable`.

## References

Hindmarsh, A. C. and Brown, P. N. and Grant, K. E. and Lee, S. L. and Serban, R.
and Shumaker, D. E. and Woodward, C. S., SUNDIALS: Suite of nonlinear and
differential/algebraic equation solvers, ACM Transactions on Mathematical
Software (TOMS), 31, pp:363–396 (2005)
"""
struct ForwardDiffOverAdjoint{A} <:
    AbstractSecondOrderSensitivityAlgorithm{nothing, true, nothing}
    adjalg::A
end

function get_autodiff_from_vjp(::ReverseDiffVJP{compile}) where {compile}
    return AutoReverseDiff(; compile)
end
get_autodiff_from_vjp(::ZygoteVJP) = AutoZygote()
get_autodiff_from_vjp(::EnzymeVJP) = AutoEnzyme()
get_autodiff_from_vjp(::MooncakeVJP) = AutoMooncake()
get_autodiff_from_vjp(::ReactantVJP) = AutoEnzyme()
get_autodiff_from_vjp(::TrackerVJP) = AutoTracker()
get_autodiff_from_vjp(::Nothing) = AutoZygote()
get_autodiff_from_vjp(b::Bool) = ifelse(b, AutoForwardDiff(), AutoFiniteDiff())
