SensitivityAlg(args...;kwargs...) = @error("The SensitivtyAlg choice mechanism was completely overhauled. Please consult the local sensitivity documentation for more information")

abstract type AbstractForwardSensitivityAlgorithm{CS,AD,FDT} <: DiffEqBase.AbstractSensitivityAlgorithm{CS,AD,FDT} end
abstract type AbstractAdjointSensitivityAlgorithm{CS,AD,FDT} <: DiffEqBase.AbstractSensitivityAlgorithm{CS,AD,FDT} end
abstract type AbstractSecondOrderSensitivityAlgorithm{CS,AD,FDT} <: DiffEqBase.AbstractSensitivityAlgorithm{CS,AD,FDT} end
abstract type AbstractShadowingSensitivityAlgorithm{CS,AD,FDT} <: DiffEqBase.AbstractSensitivityAlgorithm{CS,AD,FDT} end

"""
ForwardSensitivity{CS,AD,FDT} <: AbstractForwardSensitivityAlgorithm{CS,AD,FDT}

An implementation of continuous forward sensitivity analysis for propagating 
derivatives by solving the extended ODE. When used within adjoint differentiation
(i.e. via Zygote), this will cause forward differentiation of the `solve` call
within the reverse-mode automatic differentiation environment.

## Constructor

```julia
function ForwardSensitivity(;
                            chunk_size=0,autodiff=true,
                            diff_type=Val{:central},
                            autojacvec=autodiff,
                            autojacmat=false)
```

## Keyword Arguments

* `autodiff`: Use automatic differentiation in the internal sensitivity algorithm
  computations. Default is `true`.
* `chunk_size`: Chunk size for forward mode differentiation if full Jacobians are
  built (`autojacvec=false` and `autodiff=true`). Default is `0` for automatic
  choice of chunk size.
* `autojacvec`: Calculate the Jacobian-vector product via automatic
  differentiation with special seeding.
* `diff_type`: The method used by FiniteDiff.jl for constructing the Jacobian
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
struct ForwardSensitivity{CS,AD,FDT} <: AbstractForwardSensitivityAlgorithm{CS,AD,FDT}
  autojacvec::Bool
  autojacmat::Bool
end
Base.@pure function ForwardSensitivity(;
                                       chunk_size=0,autodiff=true,
                                       diff_type=Val{:central},
                                       autojacvec=autodiff,
                                       autojacmat=false)
  autojacvec && autojacmat && error("Choose either Jacobian matrix products or Jacobian vector products,
                                      autojacmat and autojacvec cannot both be true")
  ForwardSensitivity{chunk_size,autodiff,diff_type}(autojacvec,autojacmat)
end

"""
ForwardDiffSensitivity{CS,CTS} <: AbstractForwardSensitivityAlgorithm{CS,Nothing,Nothing}

An implementation of discrete forward sensitivity analysis through ForwardDiff.jl.
When used within adjoint differentiation (i.e. via Zygote), this will cause forward 
differentiation of the `solve` call within the reverse-mode automatic differentiation 
environment.

## Constructor 

```julia
ForwardDiffSensitivity(;chunk_size=0,convert_tspan=nothing)
```

## Keyword Arguments

* `chunk_size`: the chunk size used by ForwardDiff for computing the Jacobian, i.e. the
  number of simultaneous columns computed.
* `convert_tspan`: whether to convert time to also be `Dual` valued. By default this is
  `nothing` which will only convert if callbacks are found. Conversion is required in order
  to accurately differentiate callbacks (hybrid equations).

## SciMLProblem Support

This `sensealg` supports any `SciMLProblem`s, provided that the solver algorithms is
`SciMLBase.isautodifferentiable`. Note that `ForwardDiffSensitivity` can 
accurately differentiate code with callbacks only when `convert_tspan=true`.
"""
struct ForwardDiffSensitivity{CS,CTS} <: AbstractForwardSensitivityAlgorithm{CS,Nothing,Nothing}
end
Base.@pure function ForwardDiffSensitivity(;chunk_size=0,convert_tspan=nothing)
  ForwardDiffSensitivity{chunk_size,convert_tspan}()
end

"""
BacksolveAdjoint{CS,AD,FDT,VJP,NOISE} <: AbstractAdjointSensitivityAlgorithm{CS,AD,FDT}

An implementation of adjoint sensitivity analysis using a backwards solution of the ODE. 
By default this algorithm will use the values from the forward pass to perturb the 
backwards solution to the correct spot, allowing reduced memory (O(1) memory). Checkpointing
stabilization is included for additional numerical stability over the naive implementation.

## Constructor

```julia
BacksolveAdjoint(;chunk_size=0,autodiff=true,
                  diff_type=Val{:central},
                  autojacvec=autodiff,
                  checkpointing=true, noise=true, noisemixing=false)
```

## Keyword Arguments

* `autodiff`: Use automatic differentiation for constructing the Jacobian
  if the Jacobian needs to be constructed.  Defaults to `true`.
* `chunk_size`: Chunk size for forward-mode differentiation if full Jacobians are
  built (`autojacvec=false` and `autodiff=true`). Default is `0` for automatic
  choice of chunk size.
* `diff_type`: The method used by FiniteDiff.jl for constructing the Jacobian
  if the full Jacobian is required with `autodiff=false`.
* `autojacvec`: Calculate the vector-Jacobian product (`J'*v`) via automatic
  differentiation with special seeding. The default is `true`. The total set 
  of choices are:
    - `false`: the Jacobian is constructed via FiniteDiff.jl
    - `true`: the Jacobian is constructed via ForwardDiff.jl
    - `TrackerVJP`: Uses Tracker.jl for the vjp.
    - `ZygoteVJP`: Uses Zygote.jl for the vjp.
    - `EnzymeVJP`: Uses Enzyme.jl for the vjp.
    - `ReverseDiffVJP(compile=false)`: Uses ReverseDiff.jl for the vjp. `compile`
      is a boolean for whether to precompile the tape, which should only be done
      if there are no branches (`if` or `while` statements) in the `f` function.
* `checkpointing`: whether checkpointing is enabled for the reverse pass. Defaults
  to `true`.
* `noise`: Calculate the vector-Jacobian product (`J'*v`) of the diffusion term 
  of an SDE via automatic differentiation with special seeding. The default is `true`. 
  The total set of choices are:
    - `false`: the Jacobian is constructed via FiniteDiff.jl
    - `true`: the Jacobian is constructed via ForwardDiff.jl
    - `DiffEqSensitivity.ZygoteNoise()`: Uses Zygote.jl for the vjp.
    - `DiffEqSensitivity.ReverseDiffNoise(compile=false)`: Uses ReverseDiff.jl for 
      the vjp. `compile` is a boolean for whether to precompile the tape, which 
      should only be done if there are no branches (`if` or `while` statements) in 
      the `f` function.
* `noisemixing`: Handle noise processes that are not of the form `du[i] = f(u[i])`. 
  For example, to compute the sensitivities of an SDE with diagonal diffusion
  ```julia
  function g_mixing!(du,u,p,t)
    du[1] = p[3]*u[1] + p[4]*u[2]
    du[2] = p[3]*u[1] + p[4]*u[2]
    nothing
  end
  ```
  correctly, `noisemixing=true` must be enabled. The default is `false`.

For more details on the vjp choices, please consult the sensitivity algorithms
documentation page or the docstrings of the vjp types.

## Applicability of Backsolve and Caution

When `BacksolveAdjoint` is applicable, it is a fast method and requires the least memory.
However, one must be cautious because not all ODEs are stable under backwards integration
by the majority of ODE solvers. An example of such an equation is the Lorenz equation.
Notice that if one solves the Lorenz equation forward and then in reverse with any
adaptive time step and non-reversible integrator, then the backwards solution diverges
from the forward solution. As a quick demonstration:

```julia
using Sundials
function lorenz(du,u,p,t)
 du[1] = 10.0*(u[2]-u[1])
 du[2] = u[1]*(28.0-u[3]) - u[2]
 du[3] = u[1]*u[2] - (8/3)*u[3]
end
u0 = [1.0;0.0;0.0]
tspan = (0.0,100.0)
prob = ODEProblem(lorenz,u0,tspan)
sol = solve(prob,Tsit5(),reltol=1e-12,abstol=1e-12)
prob2 = ODEProblem(lorenz,sol[end],(100.0,0.0))
sol = solve(prob,Tsit5(),reltol=1e-12,abstol=1e-12)
@show sol[end]-u0 #[-3.22091, -1.49394, 21.3435]
```

Thus one should check the stability of the backsolve on their type of problem before
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
struct BacksolveAdjoint{CS,AD,FDT,VJP,NOISE} <: AbstractAdjointSensitivityAlgorithm{CS,AD,FDT}
  autojacvec::VJP
  checkpointing::Bool
  noise::NOISE
  noisemixing::Bool
end
Base.@pure function BacksolveAdjoint(;chunk_size=0,autodiff=true,
                                      diff_type=Val{:central},
                                      autojacvec=autodiff,
                                      checkpointing=true, noise=true,noisemixing=false)
  BacksolveAdjoint{chunk_size,autodiff,diff_type,typeof(autojacvec),typeof(noise)}(autojacvec,checkpointing,noise,noisemixing)
end

"""
InterpolatingAdjoint{CS,AD,FDT,VJP,NOISE} <: AbstractAdjointSensitivityAlgorithm{CS,AD,FDT}

An implementation of adjoint sensitivity analysis which uses the interpolation of
the forward solution for the reverse solve vector-Jacobian products. By
default it requires a dense solution of the forward pass and will internally
ignore saving arguments during the gradient calculation. When checkpointing is
enabled it will only require the memory to interpolate between checkpoints.

## Constructor

```julia
function InterpolatingAdjoint(;chunk_size=0,autodiff=true,
                               diff_type=Val{:central},
                               autojacvec=autodiff,
                               checkpointing=false, noise=true, noisemixing=false)
```

## Keyword Arguments

* `autodiff`: Use automatic differentiation for constructing the Jacobian
  if the Jacobian needs to be constructed.  Defaults to `true`.
* `chunk_size`: Chunk size for forward-mode differentiation if full Jacobians are
  built (`autojacvec=false` and `autodiff=true`). Default is `0` for automatic
  choice of chunk size.
* `diff_type`: The method used by FiniteDiff.jl for constructing the Jacobian
  if the full Jacobian is required with `autodiff=false`.
* `autojacvec`: Calculate the vector-Jacobian product (`J'*v`) via automatic
  differentiation with special seeding. The default is `true`. The total set 
  of choices are:
    - `false`: the Jacobian is constructed via FiniteDiff.jl
    - `true`: the Jacobian is constructed via ForwardDiff.jl
    - `TrackerVJP`: Uses Tracker.jl for the vjp.
    - `ZygoteVJP`: Uses Zygote.jl for the vjp.
    - `EnzymeVJP`: Uses Enzyme.jl for the vjp.
    - `ReverseDiffVJP(compile=false)`: Uses ReverseDiff.jl for the vjp. `compile`
      is a boolean for whether to precompile the tape, which should only be done
      if there are no branches (`if` or `while` statements) in the `f` function.
* `checkpointing`: whether checkpointing is enabled for the reverse pass. Defaults
  to `true`.
* `noise`: Calculate the vector-Jacobian product (`J'*v`) of the diffusion term 
  of an SDE via automatic differentiation with special seeding. The default is `true`. 
  The total set of choices are:
    - `false`: the Jacobian is constructed via FiniteDiff.jl
    - `true`: the Jacobian is constructed via ForwardDiff.jl
    - `DiffEqSensitivity.ZygoteNoise()`: Uses Zygote.jl for the vjp.
    - `DiffEqSensitivity.ReverseDiffNoise(compile=false)`: Uses ReverseDiff.jl for 
      the vjp. `compile` is a boolean for whether to precompile the tape, which 
      should only be done if there are no branches (`if` or `while` statements) in 
      the `f` function.
* `noisemixing`: Handle noise processes that are not of the form `du[i] = f(u[i])`. 
  For example, to compute the sensitivities of an SDE with diagonal diffusion
  ```julia
  function g_mixing!(du,u,p,t)
    du[1] = p[3]*u[1] + p[4]*u[2]
    du[2] = p[3]*u[1] + p[4]*u[2]
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
struct InterpolatingAdjoint{CS,AD,FDT,VJP,NOISE} <: AbstractAdjointSensitivityAlgorithm{CS,AD,FDT}
  autojacvec::VJP
  checkpointing::Bool
  noise::NOISE
  noisemixing::Bool
end
Base.@pure function InterpolatingAdjoint(;chunk_size=0,autodiff=true,
                                         diff_type=Val{:central},
                                         autojacvec=autodiff,
                                         checkpointing=false, noise=true,noisemixing=false)
  InterpolatingAdjoint{chunk_size,autodiff,diff_type,typeof(autojacvec),typeof(noise)}(autojacvec,checkpointing,noise,noisemixing)
end

"""
QuadratureAdjoint{CS,AD,FDT,VJP} <: AbstractAdjointSensitivityAlgorithm{CS,AD,FDT}

An implementation of adjoint sensitivity analysis which develops a full
continuous solution of the reverse solve in order to perform a post-ODE
quadrature. This method requires the the dense solution and will ignore
saving arguments during the gradient calculation. The tolerances in the
constructor control the inner quadrature. The inner quadrature uses a
ReverseDiff vjp if autojacvec, and `compile=false` by default but can
compile the tape under the same circumstances as `ReverseDiffVJP`.

This method is O(n^3 + p) for stiff / implicit equations (as opposed to the
O((n+p)^3) scaling of BacksolveAdjoint and InterpolatingAdjoint), and thus
is much more compute efficient. However, it requires holding a dense reverse
pass and is thus memory intensive.

## Constructor

```julia
function QuadratureAdjoint(;chunk_size=0,autodiff=true,
                            diff_type=Val{:central},
                            autojacvec=autodiff,abstol=1e-6,
                            reltol=1e-3,compile=false)
```

## Keyword Arguments

* `autodiff`: Use automatic differentiation for constructing the Jacobian
  if the Jacobian needs to be constructed.  Defaults to `true`.
* `chunk_size`: Chunk size for forward-mode differentiation if full Jacobians are
  built (`autojacvec=false` and `autodiff=true`). Default is `0` for automatic
  choice of chunk size.
* `diff_type`: The method used by FiniteDiff.jl for constructing the Jacobian
  if the full Jacobian is required with `autodiff=false`.
* `autojacvec`: Calculate the vector-Jacobian product (`J'*v`) via automatic
  differentiation with special seeding. The default is `true`. The total set 
  of choices are:
    - `false`: the Jacobian is constructed via FiniteDiff.jl
    - `true`: the Jacobian is constructed via ForwardDiff.jl
    - `TrackerVJP`: Uses Tracker.jl for the vjp.
    - `ZygoteVJP`: Uses Zygote.jl for the vjp.
    - `EnzymeVJP`: Uses Enzyme.jl for the vjp.
    - `ReverseDiffVJP(compile=false)`: Uses ReverseDiff.jl for the vjp. `compile`
      is a boolean for whether to precompile the tape, which should only be done
      if there are no branches (`if` or `while` statements) in the `f` function.
* `abstol`: absolute tolerance for the quadrature calculation
* `reltol`: relative tolerance for the quadrature calculation
* `compile`: whether to compile the vjp calculation for the integrand calculation.
  See `ReverseDiffVJP` for more details.

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
struct QuadratureAdjoint{CS,AD,FDT,VJP} <: AbstractAdjointSensitivityAlgorithm{CS,AD,FDT}
  autojacvec::VJP
  abstol::Float64
  reltol::Float64
  compile::Bool
end
Base.@pure function QuadratureAdjoint(;chunk_size=0,autodiff=true,
                                         diff_type=Val{:central},
                                         autojacvec=autodiff,abstol=1e-6,
                                         reltol=1e-3,compile=false)
  QuadratureAdjoint{chunk_size,autodiff,diff_type,typeof(autojacvec)}(autojacvec,abstol,reltol,compile)
end

"""
TrackerAdjoint <: AbstractAdjointSensitivityAlgorithm{nothing,true,nothing}

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
"""
struct TrackerAdjoint <: AbstractAdjointSensitivityAlgorithm{nothing,true,nothing} end

"""
ReverseDiffAdjoint <: AbstractAdjointSensitivityAlgorithm{nothing,true,nothing}

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
struct ReverseDiffAdjoint <: AbstractAdjointSensitivityAlgorithm{nothing,true,nothing} end

"""
ZygoteAdjoint <: AbstractAdjointSensitivityAlgorithm{nothing,true,nothing}

An implementation of discrete adjoint sensitivity analysis
using the Zygote.jl source-to-source AD directly on the differential equation
solver. 

## Constructor

```julia
ZygoteAdjoint()
```

## SciMLProblem Support

Currently fails on almost every solver.
"""
struct ZygoteAdjoint <: AbstractAdjointSensitivityAlgorithm{nothing,true,nothing} end

"""
ForwardLSS{CS,AD,FDT,aType,tType,gType} <: AbstractShadowingSensitivityAlgorithm{CS,AD,FDT}

An implementation of the forward
[least square shadowing](https://arxiv.org/abs/1204.0159) method. For `alpha`,
one can choose between two different windowing options, `CosWindowing` (default)
and `Cos2Windowing`, and `alpha::Number` which corresponds to the weight of the
time dilation term in `ForwardLSS`.

## Constructor

```julia
ForwardLSS(;
          chunk_size=0,autodiff=true,
          diff_type=Val{:central},
          alpha=CosWindowing(),
          t0skip=0.0,t1skip=0.0,
          g=nothing)
```

## Keyword Arguments

## SciMLProblem Support

## References

Wang, Q., Hu, R., and Blonigan, P. Least squares shadowing sensitivity analysis of
chaotic limit cycle oscillations. Journal of Computational Physics, 267, 210-224 (2014).
"""
struct ForwardLSS{CS,AD,FDT,aType,tType,gType} <: AbstractShadowingSensitivityAlgorithm{CS,AD,FDT}
  alpha::aType # alpha: weight of the time dilation term in LSS.
  t0skip::tType
  t1skip::tType
  g::gType
end
Base.@pure function ForwardLSS(;
                                chunk_size=0,autodiff=true,
                                diff_type=Val{:central},
                                alpha=CosWindowing(),
                                t0skip=0.0,t1skip=0.0,
                                g=nothing)
                           
  ForwardLSS{chunk_size,autodiff,diff_type,typeof(alpha),typeof(t0skip),typeof(g)}(alpha,t0skip,t1skip,g)
end

"""
AdjointLSS{CS,AD,FDT,aType,tType,gType} <: AbstractShadowingSensitivityAlgorithm{CS,AD,FDT}

An implementation of the forward
[least square shadowing](https://arxiv.org/abs/1204.0159) method. For `alpha`,
one can choose between two different windowing options, `CosWindowing` (default)
and `Cos2Windowing`, and `alpha::Number` which corresponds to the weight of the
time dilation term in `ForwardLSS`.

## Constructor

```julia
ForwardLSS(;
          chunk_size=0,autodiff=true,
          diff_type=Val{:central},
          alpha=CosWindowing(),
          t0skip=0.0,t1skip=0.0,
          g=nothing)
```

## Keyword Arguments

## SciMLProblem Support

## References

Wang, Q., Hu, R., and Blonigan, P. Least squares shadowing sensitivity analysis of
chaotic limit cycle oscillations. Journal of Computational Physics, 267, 210-224 (2014).
"""
struct AdjointLSS{CS,AD,FDT,aType,tType,gType} <: AbstractShadowingSensitivityAlgorithm{CS,AD,FDT}
  alpha::aType # alpha: weight of the time dilation term in LSS.
  t0skip::tType
  t1skip::tType
  g::gType
end
Base.@pure function AdjointLSS(;
                                chunk_size=0,autodiff=true,
                                diff_type=Val{:central},
                                alpha=10.0,
                                t0skip=0.0,t1skip=0.0,
                                g=nothing)
  AdjointLSS{chunk_size,autodiff,diff_type,typeof(alpha),typeof(t0skip),typeof(g)}(alpha,t0skip,t1skip,g)
end

abstract type WindowingChoice end
struct CosWindowing <: WindowingChoice end
struct Cos2Windowing <: WindowingChoice end

"""
struct NILSS{CS,AD,FDT,RNG,gType} <: AbstractShadowingSensitivityAlgorithm{CS,AD,FDT}

An implementation of the adjoint-mode
[least square shadowing](https://arxiv.org/abs/1204.0159) method. `alpha`
controls the weight of the time dilation term in `AdjointLSS`.

## Constructor

```julia
NILSS(nseg, nstep; rng = Xorshifts.Xoroshiro128Plus(rand(UInt64)),
                                chunk_size=0,autodiff=true,
                                diff_type=Val{:central},
                                autojacvec=autodiff,
                                g=nothing
                                )
```

## Keyword Arguments

## SciMLProblem Support

## References

Ni, A., and Wang, Q. Sensitivity analysis on chaotic dynamical systems by Non-Intrusive
Least Squares Shadowing (NILSS). Journal of Computational Physics 347, 56-77 (2017).
"""
struct NILSS{CS,AD,FDT,RNG,gType} <: AbstractShadowingSensitivityAlgorithm{CS,AD,FDT}
  rng::RNG
  nseg::Int
  nstep::Int
  autojacvec::Bool
  g::gType
end
Base.@pure function NILSS(nseg, nstep; rng = Xorshifts.Xoroshiro128Plus(rand(UInt64)),
                                chunk_size=0,autodiff=true,
                                diff_type=Val{:central},
                                autojacvec=autodiff,
                                g=nothing
                                )
  NILSS{chunk_size,autodiff,diff_type,typeof(rng),typeof(g)}(rng, nseg, nstep, autojacvec,g)
end

"""
NILSAS{CS,AD,FDT,RNG,SENSE,gType} <: AbstractShadowingSensitivityAlgorithm{CS,AD,FDT}

An implementation of the [non-intrusive least squares shadowing (NILSS)](https://arxiv.org/abs/1611.00880)
method. `nseg` is the number of segments. `nstep` is the number of steps per
segment.

## Constructor

```julia
NILSAS(nseg, nstep, M=nothing; rng = Xorshifts.Xoroshiro128Plus(rand(UInt64)),
                                adjoint_sensealg = BacksolveAdjoint(),
                                chunk_size=0,autodiff=true,
                                diff_type=Val{:central},
                                autojacvec=autodiff,
                                g=nothing
                                )
```

## Keyword Arguments

## SciMLProblem Support

## References

Ni, A., and Talnikar, C., Adjoint sensitivity analysis on chaotic dynamical systems 
by Non-Intrusive Least Squares Adjoint Shadowing (NILSAS). Journal of Computational 
Physics 395, 690-709 (2019).
"""
struct NILSAS{CS,AD,FDT,RNG,SENSE,gType} <: AbstractShadowingSensitivityAlgorithm{CS,AD,FDT}
  rng::RNG
  adjoint_sensealg::SENSE
  M::Int
  nseg::Int
  nstep::Int
  autojacvec::Bool
  g::gType
end
Base.@pure function NILSAS(nseg, nstep, M=nothing; rng = Xorshifts.Xoroshiro128Plus(rand(UInt64)),
                                adjoint_sensealg = BacksolveAdjoint(),
                                chunk_size=0,autodiff=true,
                                diff_type=Val{:central},
                                autojacvec=autodiff,
                                g=nothing
                                )
  # integer dimension of the unstable subspace
  M === nothing && error("Please provide an `M` with `M >= nus + 1`, where nus is the number of unstable covariant Lyapunov vectors.")

  NILSAS{chunk_size,autodiff,diff_type,typeof(rng),typeof(adjoint_sensealg),typeof(g)}(rng, adjoint_sensealg, M, 
    nseg, nstep, autojacvec, g)
end

"""
SteadyStateAdjoint{CS,AD,FDT,VJP,LS} <: AbstractAdjointSensitivityAlgorithm{CS,AD,FDT}

An implementation of the adjoint differentiation of a nonlinear solve. Uses the
implicit function theorem to directly compute the derivative of the solution to
``f(u,p) = 0`` with respect to `p`.

## Constructor

```julia
SteadyStateAdjoint(;chunk_size = 0, autodiff = true, 
                    diff_type = Val{:central},
                    autojacvec = autodiff, linsolve = nothing)
```

## Keyword Arguments

## Keyword Arguments

* `autodiff`: Use automatic differentiation for constructing the Jacobian
  if the Jacobian needs to be constructed.  Defaults to `true`.
* `chunk_size`: Chunk size for forward-mode differentiation if full Jacobians are
  built (`autojacvec=false` and `autodiff=true`). Default is `0` for automatic
  choice of chunk size.
* `diff_type`: The method used by FiniteDiff.jl for constructing the Jacobian
  if the full Jacobian is required with `autodiff=false`.
* `autojacvec`: Calculate the vector-Jacobian product (`J'*v`) via automatic
  differentiation with special seeding. The default is `true`. The total set 
  of choices are:
    - `false`: the Jacobian is constructed via FiniteDiff.jl
    - `true`: the Jacobian is constructed via ForwardDiff.jl
    - `TrackerVJP`: Uses Tracker.jl for the vjp.
    - `ZygoteVJP`: Uses Zygote.jl for the vjp.
    - `EnzymeVJP`: Uses Enzyme.jl for the vjp.
    - `ReverseDiffVJP(compile=false)`: Uses ReverseDiff.jl for the vjp. `compile`
      is a boolean for whether to precompile the tape, which should only be done
      if there are no branches (`if` or `while` statements) in the `f` function.
* `linsolve`: the linear solver used in the adjoint solve. Defaults to `nothing`,
  which uses a polyalgorithm to attempt to automatically choose an efficient 
  algorithm.

For more details on the vjp choices, please consult the sensitivity algorithms
documentation page or the docstrings of the vjp types.

## References

Johnson, S. G., Notes on Adjoint Methods for 18.336, Online at
http://math.mit.edu/stevenj/18.336/adjoint.pdf (2007)
"""
struct SteadyStateAdjoint{CS,AD,FDT,VJP,LS} <: AbstractAdjointSensitivityAlgorithm{CS,AD,FDT}
  autojacvec::VJP
  linsolve::LS
end

Base.@pure function SteadyStateAdjoint(;chunk_size = 0, autodiff = true, diff_type = Val{:central},
                                        autojacvec = autodiff, linsolve = nothing)
  SteadyStateAdjoint{chunk_size,autodiff,diff_type,typeof(autojacvec),typeof(linsolve)}(autojacvec,linsolve)
end

abstract type VJPChoice end

"""
ZygoteVJP <: VJPChoice

Uses Zygote.jl to compute vector-Jacobian products. Tends to be the fastest VJP method if the 
ODE/DAE/SDE/DDE is written with mostly vectorized  functions (like neural networks and other 
layers from Flux.jl) and the `f` functions is given out-of-place. If the `f` function is 
in-place, then `Zygote.Buffer` arrays are used internally which can greatly reduce the 
performance of the VJP method. 

## Constructor

```julia
ZygoteVJP(compile=false)
```
"""
struct ZygoteVJP <: VJPChoice end

"""
EnzymeVJP <: VJPChoice

Uses Enzyme.jl to compute vector-Jacobian products. Is the fastest VJP whenever applicable,
though Enzyme.jl currently has low coverage over the Julia programming language, for example
restricting the user's defined `f` function to not do things like require garbage collection
or calls to BLAS/LAPACK. However, mutation is supported, meaning that in-place `f` with 
fully mutating non-allocating code will work with Enzyme (provided no high level calls to C
like BLAS/LAPACK are used) and this will be the most efficient adjoint implementation.

## Constructor

```julia
EnzymeVJP(compile=false)
```
"""
struct EnzymeVJP <: VJPChoice end

"""
TrackerVJP <: VJPChoice

Uses Tracker.jl to compute the vector-Jacobian products. If `f` is in-place,
then it uses a array of structs formulation to do scalarized reverse mode, 
while if `f` is out-of-place then it uses an array-based reverse mode.

Not as efficient as `ReverseDiffVJP`, but supports GPUs when doing array-based
reverse mode.

## Constructor

```julia
TrackerVJP(compile=false)
```
"""
struct TrackerVJP <: VJPChoice end

"""
ReverseDiffVJP{compile} <: VJPChoice

Uses ReverseDiff.jl to compute the vector-Jacobian products. If `f` is in-place,
then it uses a array of structs formulation to do scalarized reverse mode, 
while if `f` is out-of-place then it uses an array-based reverse mode.

Usually the fastest when scalarized operations exist in the f function 
(like in scientific machine learning applications like Universal Differential Equations) 
and the boolean compilation is enabled (i.e. ReverseDiffVJP(true)), if EnzymeVJP fails on
a given choice of `f`.

Does not support GPUs (CuArrays).

## Constructor

```julia
ReverseDiffVJP(compile=false)
```

## Keyword Arguments

* `compile`: Whether to cache the compilation of the reverse tape. This heavily increases
  the performance of the method but requires that the `f` function of the ODE/DAE/SDE/DDE 
  has no branching. 
"""
struct ReverseDiffVJP{compile} <: VJPChoice
  ReverseDiffVJP(compile=false) = new{compile}()
end

abstract type NoiseChoice end

"""
ZygoteNoise <: NoiseChoice

Uses Zygote.jl to compute vector-Jacobian products for the noise term (for SDE adjoints only). 
Tends to be the fastest VJP method if the ODE/DAE/SDE/DDE is written with mostly vectorized 
functions (like neural networks and other layers from Flux.jl) and the `f` functions is given 
out-of-place. If the `f` function is in-place, then `Zygote.Buffer` arrays are used 
internally which can greatly reduce the performance of the VJP method. 

## Constructor

```julia
ZygoteNoise()
```
"""
struct ZygoteNoise <: NoiseChoice end

"""
ReverseDiffNoise{compile} <: NoiseChoice

Uses ReverseDiff.jl to compute the vector-Jacobian products for the noise
term differentiation (for SDE adjoints only). If `f` is in-place,
then it uses a array of structs formulation to do scalarized reverse mode, 
while if `f` is out-of-place then it uses an array-based reverse mode.

Usually the fastest when scalarized operations exist in the f function 
(like in scientific machine learning applications like Universal Differential Equations) 
and the boolean compilation is enabled (i.e. ReverseDiffVJP(true)), if EnzymeVJP fails on
a given choice of `f`.

Does not support GPUs (CuArrays).

## Constructor

```julia
ReverseDiffNoise(compile=false)
```

## Keyword Arguments

* `compile`: Whether to cache the compilation of the reverse tape. This heavily increases
  the performance of the method but requires that the `f` function of the ODE/DAE/SDE/DDE 
  has no branching. 
"""
struct ReverseDiffNoise{compile} <: NoiseChoice
  ReverseDiffNoise(compile=false) = new{compile}()
end

@inline convert_tspan(::ForwardDiffSensitivity{CS,CTS}) where {CS,CTS} = CTS
@inline convert_tspan(::Any) = nothing
@inline alg_autodiff(alg::DiffEqBase.AbstractSensitivityAlgorithm{CS,AD,FDT}) where {CS,AD,FDT} = AD
@inline get_chunksize(alg::DiffEqBase.AbstractSensitivityAlgorithm{CS,AD,FDT}) where {CS,AD,FDT} = CS
@inline diff_type(alg::DiffEqBase.AbstractSensitivityAlgorithm{CS,AD,FDT}) where {CS,AD,FDT} = FDT
@inline function get_jacvec(alg::DiffEqBase.AbstractSensitivityAlgorithm)
  alg.autojacvec isa Bool ? alg.autojacvec : true
end
@inline function get_jacmat(alg::DiffEqBase.AbstractSensitivityAlgorithm)
  alg.autojacmat isa Bool ? alg.autojacmat : true
end
@inline ischeckpointing(alg::DiffEqBase.AbstractSensitivityAlgorithm, sol=nothing) = false
@inline ischeckpointing(alg::InterpolatingAdjoint) = alg.checkpointing
@inline ischeckpointing(alg::InterpolatingAdjoint, sol) = alg.checkpointing || !sol.dense
@inline ischeckpointing(alg::BacksolveAdjoint, sol=nothing) = alg.checkpointing

@inline isnoise(alg::DiffEqBase.AbstractSensitivityAlgorithm) = false
@inline isnoise(alg::InterpolatingAdjoint) = alg.noise
@inline isnoise(alg::BacksolveAdjoint) = alg.noise

@inline isnoisemixing(alg::DiffEqBase.AbstractSensitivityAlgorithm) = false
@inline isnoisemixing(alg::InterpolatingAdjoint) = alg.noisemixing
@inline isnoisemixing(alg::BacksolveAdjoint) = alg.noisemixing

@inline compile_tape(vjp::ReverseDiffVJP{compile}) where compile = compile
@inline compile_tape(noise::ReverseDiffNoise{compile}) where compile = compile
@inline compile_tape(autojacvec::Bool) = false
@inline compile_tape(sensealg::QuadratureAdjoint) = sensealg.compile

"""
ForwardDiffOverAdjoint{A} <: AbstractSecondOrderSensitivityAlgorithm{nothing,true,nothing}

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
struct ForwardDiffOverAdjoint{A} <: AbstractSecondOrderSensitivityAlgorithm{nothing,true,nothing}
  adjalg::A
end
