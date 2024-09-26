## Direct calls

const ADJOINT_PARAMETER_COMPATIBILITY_MESSAGE = """
                                                Adjoint sensitivity analysis functionality requires being able to solve
                                                a differential equation defined by the parameter struct `p`. Thus while
                                                DifferentialEquations.jl can support any parameter struct type, usage
                                                with adjoint sensitivity analysis requires that `p` could be a valid
                                                type for being the initial condition `u0` of an array. This means that
                                                many simple types, such as `Tuple`s and `NamedTuple`s, will work as
                                                parameters in normal contexts but will fail during adjoint differentiation.
                                                To work around this issue for complicated cases like nested structs, look
                                                into defining `p` using `AbstractArray` libraries such as RecursiveArrayTools.jl
                                                or ComponentArrays.jl so that `p` is an `AbstractArray` with a concrete element type.
                                                """

struct AdjointSensitivityParameterCompatibilityError <: Exception end

function Base.showerror(io::IO, e::AdjointSensitivityParameterCompatibilityError)
    print(io, ADJOINT_PARAMETER_COMPATIBILITY_MESSAGE)
end

@doc doc"""
```julia
adjoint_sensitivities(sol,alg;t=nothing,
                            dgdu_discrete = nothing, dgdp_discrete = nothing,
                            dgdu_continuous = nothing, dgdp_continuous = nothing,
                            g=nothing,
                            abstol=1e-6,reltol=1e-3,
                            checkpoints=sol.t,
                            corfunc_analytical=nothing,
                            callback = nothing,
                            sensealg=InterpolatingAdjoint(),
                            kwargs...)
```

Adjoint sensitivity analysis is used to find the gradient of the solution
with respect to some functional of the solution. Often, this is used
in an optimization problem to return the gradient with respect to some cost
function. It is equivalent to "backpropagation" or reverse-mode automatic
differentiation of a differential equation.

Using `adjoint_sensitivities` directly lets you do three things. First, it can
allow you to be more efficient, since the sensitivity calculation can be done
directly on a cost function, avoiding the overhead of building the derivative
of the full concretized solution. It can also allow you to be more efficient
by directly controlling the forward solve that is then reversed over. Lastly,
it allows one to define a continuous cost function on the continuous solution,
instead of just at discrete data points.

!!! warning

      Adjoint sensitivity analysis functionality requires being able to solve
      a differential equation defined by the parameter struct `p`. Even though
      DifferentialEquations.jl can support any parameter struct type, usage
      with adjoint sensitivity analysis requires that `p` could be a valid
      type for being the initial condition `u0` of an array. This means that
      many simple types, such as `Tuple`s and `NamedTuple`s, will work as
      parameters in normal contexts but will fail during adjoint differentiation.
      To work around this issue for complicated cases like nested structs, look
      into defining `p` using `AbstractArray` libraries such as RecursiveArrayTools.jl
      or ComponentArrays.jl so that `p` is an `AbstractArray` with a concrete element type.

!!! warning

      Non-checkpointed InterpolatingAdjoint and QuadratureAdjoint sensealgs
      require that the forward solution `sol(t)` has an accurate dense
      solution unless checkpointing is used. This means that you should
      not use `solve(prob,alg,saveat=ts)` unless checkpointing. If specific
      saving is required, one should solve dense `solve(prob,alg)`, use the
      solution in the adjoint, and then `sol(ts)` interpolate.

## Mathematical Definition

Adjoint sensitivity analysis finds the gradient of a cost function ``G`` defined by the
infinitesimal cost over the whole time period ``(t_{0}, T)``, given by the equation:

```math
G(u,p)=G(u(\cdot,p))=\int_{t_{0}}^{T}g(u(t,p),p,t)dt
```

It does so by solving the adjoint problem:

```math
\frac{d\lambda^{\star}}{dt}=g_{u}(u(t,p),p)-\lambda^{\star}(t)f_{u}(t,u(t,p),p),\thinspace\thinspace\thinspace\lambda^{\star}(T)=0
```

and obtaining the sensitivities through the integral:

```math
\frac{dG}{dp}=\int_{t_{0}}^{T}\lambda^{\star}(t)f_{p}(t)+g_{p}(t)dt+\lambda^{\star}(t_{0})u_{p}(t_{0})
```

As defined, that cost function only has non-zero values over nontrivial intervals. However, often
one may want to include in the cost function loss values at discrete points, for example, matching
the data at time points `t`. In this case, terms of `g` can be represented by Dirac delta functions,
which are then applied to the corresponding ``\lambda^\star`` and ``\frac{dG}{dp}`` equations.

For more information, see [Sensitivity Math Details](@ref sensitivity_math).

## Positional Arguments

- `sol`: the solution from the forward pass of the ODE. Note that if not using a checkpointing
  sensitivity algorithm, then it's assumed that the (dense) interpolation of the forward solution is
  of sufficient accuracy for recreating the solution at any time point.
- `alg`: the algorithm (i.e., DiffEq solver) to use for the solution of the adjoint problem.

## Keyword Arguments

- `t`: the time points at which the discrete cost function is to be evaluated.
  This argument is only required if discrete cost functions are declared.
- `g`: the continuous instantaneous cost ``g(u,p,t)`` at a given time point
  represented by a Julia function `g(u,p,t)`. This argument is only required
  if there is a continuous instantaneous cost contribution.
- `dgdu_discrete`: the partial derivative ``g_u`` evaluated at the discrete
  (Dirac delta) times. If discrete cost values are given, then `dgdu_discrete`
  is required.
- `dgdp_discrete`: the partial derivative ``g_p`` evaluated at the discrete
  (Dirac delta) times. If discrete cost values are given, then `dgdp_discrete`
  is not required and is assumed to be zero.
- `dgdu_continuous`: the partial derivative ``g_u`` evaluated at times
  not corresponding to terms with an associated Dirac delta. If `g` is given,
  then this term is not required and will be approximated by numerical or (forward-mode) automatic
  differentiation (via the `autodiff` keyword argument in the `sensealg`)
  if this term is not given by the user.
- `dgdp_continuous`: the partial derivative ``g_p`` evaluated at times
  not corresponding to terms with an associated Dirac delta. If `g` is given,
  then this term is not required and will be approximated by numerical or (forward-mode) automatic
  differentiation (via the `autojacvec` keyword argument in the `sensealg`)
  if this term is not given by the user.
- `abstol`: the absolute tolerance of the adjoint solve. Defaults to `1e-3`
- `reltol`: the relative tolerance of the adjoint solve. Defaults to `1e-3`
- `checkpoints`: the values to use for the checkpoints of the reverse solve, if the
  adjoint `sensealg` has `checkpointing = true`. Defaults to `sol.t`, i.e. the
  saved points in the `sol`.
- `corfunc_analytical`: the function corresponding to the conversion from an Ito to a Stratanovich definition of an SDE, i.e.
For sensitivity analysis of an SDE in the Ito sense ``dX = a(X,t)dt + b(X,t)dW_t`` with conversion term ``- 1/2 b_X b``, `corfunc_analytical` denotes `b_X b``.
Only used if the `sol.prob isa SDEProblem`. If not given, this is
  computed using automatic differentiation. Note that this inside of the reverse solve SDE then implies automatic
  differentiation of a function being automatic differentiated, and nested higher order automatic differentiation
  has more restrictions on the function plus some performance disadvantages.
- `callback`: callback functions to be used in the adjoint solve. Defaults to
  `nothing`.
- `sensealg`: the choice for what adjoint method to use for the reverse solve.
  Defaults to `InterpolatingAdjoint()`. See the
  [sensitivity algorithms](@ref sensitivity_diffeq) page for more details.
- `kwargs`: any extra keyword arguments passed to the adjoint solve.

## Detailed Description

For discrete adjoints where the cost functions only depend on parameters through
the ODE solve itself (for example, parameter estimation with L2 loss), use:

```julia
du0,dp = adjoint_sensitivities(sol,alg;t=ts,dgdu_discrete=dg,
                               sensealg=InterpolatingAdjoint(),
                               checkpoints=sol.t,kwargs...)
```

where `alg` is the ODE algorithm to solve the adjoint problem, `dgdu_discrete` is the
jump function, `sensealg` is the sensitivity algorithm, and `ts` are the time points
for data. `dg` is given by:

```julia
dg(out,u,p,t,i)
```

which is the in-place gradient of the cost functional `g` at time point `ts[i]`
with `u=u(t)`.

For continuous functionals, the form is:

```julia
du0,dp = adjoint_sensitivities(sol,alg;dgdu_continuous=dgdu,g=g,
                               dgdp_continuous = dgdp,
                               sensealg=InterpolatingAdjoint(),
                               checkpoints=sol.t,kwargs...)
```

for the cost functional

```julia
g(u,p,t)
```

with in-place gradient

```julia
dgdu(out,u,p,t)
dgdp(out,u,p,t)
```

If the gradient is omitted, i.e.

```julia
du0,dp = adjoint_sensitivities(sol,alg;g=g,kwargs...)
```

then we assume `dgdp` is zero and `dgdu` will be computed automatically using ForwardDiff or finite
differencing, depending on the `autodiff` setting in the `AbstractSensitivityAlgorithm`.
Note that the keyword arguments are passed to the internal ODE solver for
solving the adjoint problem.

!!! note

    Mixing discrete and continuous terms in the cost function is allowed

## Examples

### Example discrete adjoints on a cost function

In this example, we will show solving for the adjoint sensitivities of a discrete
cost functional. First, let's solve the ODE and get a high quality continuous
solution:

```julia
function f(du,u,p,t)
  du[1] = dx = p[1]*u[1] - p[2]*u[1]*u[2]
  du[2] = dy = -p[3]*u[2] + u[1]*u[2]
end

p = [1.5,1.0,3.0]
prob = ODEProblem(f,[1.0;1.0],(0.0,10.0),p)
sol = solve(prob,Vern9(),abstol=1e-10,reltol=1e-10)
```

Now let's calculate the sensitivity of the ``\ell_2`` error against 1 at evenly spaced
points in time, that is:

```math
L(u,p,t)=\sum_{i=1}^{n}\frac{\Vert1-u(t_{i},p)\Vert^{2}}{2}
```

for ``t_i = 0.5i``. This is the assumption that the data is `data[i]=1.0`.
For this function, notice we have that:

```math
\begin{aligned}
dg_{1}&=-1+u_{1} \\
dg_{2}&=-1+u_{2} \\
& \quad \vdots
\end{aligned}
```

and thus:

```julia
dg(out,u,p,t,i) = (out.=-1.0.+u)
```

Also, we can omit `dgdp` because the cost function doesn't dependent on `p`. If we had data, we'd just replace `1.0` with `data[i]`. To get the adjoint
sensitivities, call:

```julia
ts = 0:0.5:10
res = adjoint_sensitivities(sol,Vern9();t=ts,dg_discrete=dg,abstol=1e-14,
                            reltol=1e-14)
```

This is super high accuracy. As always, there's a tradeoff between accuracy
and computation time. We can check this almost exactly matches the
autodifferentiation and numerical differentiation results:

```julia
using ForwardDiff,Calculus,Tracker
function G(p)
  tmp_prob = remake(prob,u0=convert.(eltype(p),prob.u0),p=p)
  sol = solve(tmp_prob,Vern9(),abstol=1e-14,reltol=1e-14,saveat=ts,
              sensealg=SensitivityADPassThrough())
  A = convert(Array,sol)
  sum(((1 .- A).^2)./2)
end
G([1.5,1.0,3.0])
res2 = ForwardDiff.gradient(G,[1.5,1.0,3.0])
res3 = Calculus.gradient(G,[1.5,1.0,3.0])
res4 = Tracker.gradient(G,[1.5,1.0,3.0])
res5 = ReverseDiff.gradient(G,[1.5,1.0,3.0])
```

and see this gives the same values.

### Example controlling adjoint method choices and checkpointing

In the previous examples, all calculations were done using the interpolating
method. This maximizes speed, but at a cost of requiring a dense `sol`. If it
is not possible to hold a dense forward solution in memory, then one can use
checkpointing. For example:

```julia
ts = [0.0,0.2,0.5,0.7]
sol = solve(prob,Vern9(),saveat=ts)
```

Creates a non-dense solution with checkpoints at `[0.0,0.2,0.5,0.7]`. Now we
can do:

```julia
res = adjoint_sensitivities(sol,Vern9();t=ts,dg_discrete=dg,
                            sensealg=InterpolatingAdjoint(checkpointing=true))
```

When grabbing a Jacobian value during the backwards solution, it will no longer
interpolate to get the value. Instead, it will start a forward solution at the
nearest checkpoint to build local interpolants in a way that conserves memory.
By default, the checkpoints are at `sol.t`, but we can override this:

```julia
res = adjoint_sensitivities(sol,Vern9();t=ts,dg_discrte=dg,
                            sensealg=InterpolatingAdjoint(checkpointing=true),
                            checkpoints = [0.0,0.5])
```

### Example continuous adjoints on an energy functional

In this case, we'd like to calculate the adjoint sensitivity of the scalar energy
functional:

```math
G(u,p)=\int_{0}^{T}\frac{\sum_{i=1}^{n}u_{i}^{2}(t)}{2}dt
```

which is:

```julia
g(u,p,t) = (sum(u).^2) ./ 2
```

Notice that the gradient of this function with respect to the state `u` is:

```julia
function dg(out,u,p,t)
  out[1]= u[1] + u[2]
  out[2]= u[1] + u[2]
end
```

To get the adjoint sensitivities, we call:

```julia
res = adjoint_sensitivities(sol,Vern9();dg_continuous=dg,g=g,abstol=1e-8,
                                 reltol=1e-8,iabstol=1e-8,ireltol=1e-8)
```

Notice that we can check this against autodifferentiation and numerical
differentiation as follows:

```julia
using QuadGK
function G(p)
  tmp_prob = remake(prob,p=p)
  sol = solve(tmp_prob,Vern9(),abstol=1e-14,reltol=1e-14)
  res,err = quadgk((t)-> (sum(sol(t)).^2)./2,0.0,10.0,atol=1e-14,rtol=1e-10)
  res
end
res2 = ForwardDiff.gradient(G,[1.5,1.0,3.0])
res3 = Calculus.gradient(G,[1.5,1.0,3.0])
```
"""
function adjoint_sensitivities(sol, args...;
        sensealg = InterpolatingAdjoint(),
        verbose = true, kwargs...)
    p = SymbolicIndexingInterface.parameter_values(sol)
    if !(p === nothing || p isa SciMLBase.NullParameters)
        if !isscimlstructure(p) && !isfunctor(p)
            throw(SciMLStructuresCompatibilityError())
        end
    end

    prob = sol.prob
    if isscimlstructure(sol.prob.p)
        tunables, repack, aliases = canonicalize(Tunable(), p)
    else
        tunables, repack, aliases = p, identity, false
    end

    if hasfield(typeof(sensealg), :autojacvec) && sensealg.autojacvec === nothing
        if haskey(kwargs, :callback)
            has_cb = kwargs[:callback] !== nothing
        else
            has_cb = false
        end
        if !has_cb
            _sensealg = if isinplace(sol.prob)
                setvjp(
                    sensealg, inplace_vjp(prob, state_values(prob), p, verbose, repack))
            else
                setvjp(sensealg, ZygoteVJP())
            end
        else
            _sensealg = setvjp(sensealg, ReverseDiffVJP())
        end

        return try
            _adjoint_sensitivities(sol, _sensealg, args...; verbose, kwargs...)
        catch e
            verbose &&
                @warn "Automatic AD choice of autojacvec failed in ODE adjoint, failing back to ODE adjoint + numerical vjp"
            _adjoint_sensitivities(sol, setvjp(sensealg, false), args...; verbose,
                kwargs...)
        end
    else
        return _adjoint_sensitivities(sol, sensealg, args...; verbose, kwargs...)
    end
end

function _adjoint_sensitivities(sol, sensealg, alg;
        t = nothing,
        dgdu_discrete = nothing, dgdp_discrete = nothing,
        dgdu_continuous = nothing, dgdp_continuous = nothing,
        g = nothing,
        abstol = 1e-6, reltol = 1e-3,
        checkpoints = current_time(sol),
        corfunc_analytical = nothing,
        callback = nothing,
        kwargs...)
    mtkp = SymbolicIndexingInterface.parameter_values(sol)
    if !(mtkp isa Union{Nothing, SciMLBase.NullParameters, AbstractArray}) ||
       (mtkp isa AbstractArray && !Base.isconcretetype(eltype(mtkp)))
        throw(AdjointSensitivityParameterCompatibilityError())
    end
    rcb = nothing
    if sol.prob isa ODEProblem
        adj_prob, rcb = ODEAdjointProblem(sol, sensealg, alg, t, dgdu_discrete,
            dgdp_discrete,
            dgdu_continuous, dgdp_continuous, g, Val(true);
            checkpoints = checkpoints,
            callback = callback,
            abstol = abstol, reltol = reltol, kwargs...)

    elseif sol.prob isa SDEProblem
        adj_prob = SDEAdjointProblem(sol, sensealg, alg, t, dgdu_discrete, dgdp_discrete,
            dgdu_continuous, dgdp_continuous, g;
            checkpoints = checkpoints,
            callback = callback,
            abstol = abstol, reltol = reltol,
            corfunc_analytical = corfunc_analytical)
    elseif sol.prob isa RODEProblem
        adj_prob = RODEAdjointProblem(sol, sensealg, alg, t, dgdu_discrete, dgdp_discrete,
            dgdu_continuous, dgdp_continuous, g;
            checkpoints = checkpoints,
            callback = callback,
            abstol = abstol, reltol = reltol,
            corfunc_analytical = corfunc_analytical)
    else
        error("Continuous adjoint sensitivities are only supported for ODE/SDE/RODE problems.")
    end

    tstops = ischeckpointing(sensealg, sol) ? checkpoints : similar(current_time(sol), 0)
    adj_sol = solve(adj_prob, alg;
        save_everystep = false, save_start = false, saveat = eltype(state_values(sol, 1))[],
        tstops = tstops, abstol = abstol, reltol = reltol, kwargs...)

    if mtkp === nothing || mtkp isa SciMLBase.NullParameters
        tunables, repack = mtkp, identity
    else
        tunables, _, _ = canonicalize(Tunable(), mtkp)
    end
    prob = sol.prob
    l = mtkp === nothing || mtkp === SciMLBase.NullParameters() ? 0 : length(tunables)
    du0 = state_values(adj_sol)[end][1:length(state_values(prob))]

    if eltype(mtkp) <: real(eltype(state_values(adj_sol)[end]))
        dp = real.(state_values(adj_sol)[end][(1:l) .+ length(state_values(prob))])'
    elseif mtkp === nothing || mtkp === SciMLBase.NullParameters()
        dp = nothing
    else
        dp = state_values(adj_sol)[end][(1:l) .+ length(state_values(prob))]'
    end

    if rcb !== nothing && !isempty(rcb.Δλas)
        S = adj_prob.f.f
        iλ = similar(rcb.λ, length(state_values(sol, 1)))
        out = zero(dp')
        yy = similar(rcb.y)
        for (Δλa, tt) in rcb.Δλas
            iλ .= zero(eltype(iλ))
            (; algevar_idxs) = rcb.diffcache
            iλ[algevar_idxs] .= Δλa
            sol(yy, tt)
            vecjacobian!(nothing, yy, iλ, mtkp, tt, S, dgrad = out)
            dp .+= out'
        end
    end

    du0, dp
end

function _adjoint_sensitivities(sol, sensealg::SteadyStateAdjoint, alg;
        dgdu = nothing, dgdp = nothing, g = nothing,
        abstol = 1e-6, reltol = 1e-3,
        kwargs...)
    SteadyStateAdjointProblem(sol, sensealg, alg, dgdu, dgdp, g; kwargs...)
end

@doc doc"""
H = second_order_sensitivities(loss,prob,alg,args...;
                               sensealg=ForwardDiffOverAdjoint(InterpolatingAdjoint(autojacvec=ReverseDiffVJP())),
                               kwargs...)

Second order sensitivity analysis is used for the fast calculation of Hessian
matrices.

!!! warning

      Adjoint sensitivity analysis functionality requires being able to solve
      a differential equation defined by the parameter struct `p`. Even though
      DifferentialEquations.jl can support any parameter struct type, usage
      with adjoint sensitivity analysis requires that `p` could be a valid
      type for being the initial condition `u0` of an array. This means that
      many simple types, such as `Tuple`s and `NamedTuple`s, will work as
      parameters in normal contexts but will fail during adjoint differentiation.
      To work around this issue for complicated cases like nested structs, look
      into defining `p` using `AbstractArray` libraries such as RecursiveArrayTools.jl
      or ComponentArrays.jl so that `p` is an `AbstractArray` with a concrete element type.

### Example second order sensitivity analysis calculation

```julia
using SciMLSensitivity, OrdinaryDiffEq, ForwardDiff
using Test

function lotka!(du,u,p,t)
  du[1] = dx = p[1]*u[1] - p[2]*u[1]*u[2]
  du[2] = dy = -p[3]*u[2] + p[4]*u[1]*u[2]
end

p = [1.5,1.0,3.0,1.0]; u0 = [1.0;1.0]
prob = ODEProblem(lotka!,u0,(0.0,10.0),p)
loss(sol) = sum(sol)
v = ones(4)

H  = second_order_sensitivities(loss,prob,Vern9(),saveat=0.1,abstol=1e-12,reltol=1e-12)
```

## Arguments

The arguments for this function match `adjoint_sensitivities`. The only notable difference
is `sensealg` which requires a second order sensitivity algorithm, of which currently the
only choice is `ForwardDiffOverAdjoint` which uses forward-over-reverse to mix a forward-mode
sensitivity analysis with an adjoint sensitivity analysis for a faster computation than either
double forward or double reverse. `ForwardDiffOverAdjoint`'s positional argument just accepts
a first order sensitivity algorithm.
"""
function second_order_sensitivities(loss, prob, alg, args...;
        sensealg = ForwardDiffOverAdjoint(InterpolatingAdjoint(autojacvec = ReverseDiffVJP())),
        kwargs...)
    _second_order_sensitivities(loss, prob, alg, sensealg, args...; kwargs...)
end

@doc doc"""
Hv = second_order_sensitivity_product(loss,v,prob,alg,args...;
                               sensealg=ForwardDiffOverAdjoint(InterpolatingAdjoint(autojacvec=ReverseDiffVJP())),
                               kwargs...)

Second order sensitivity analysis product is used for the fast calculation of
Hessian-vector products ``Hv`` without requiring the construction of the Hessian
matrix.

!!! warning

      Adjoint sensitivity analysis functionality requires being able to solve
      a differential equation defined by the parameter struct `p`. Even though
      DifferentialEquations.jl can support any parameter struct type, usage
      with adjoint sensitivity analysis requires that `p` could be a valid
      type for being the initial condition `u0` of an array. This means that
      many simple types, such as `Tuple`s and `NamedTuple`s, will work as
      parameters in normal contexts but will fail during adjoint differentiation.
      To work around this issue for complicated cases like nested structs, look
      into defining `p` using `AbstractArray` libraries such as RecursiveArrayTools.jl
      or ComponentArrays.jl so that `p` is an `AbstractArray` with a concrete element type.

### Example second order sensitivity analysis calculation

```julia
using SciMLSensitivity, OrdinaryDiffEq, ForwardDiff
using Test

function lotka!(du,u,p,t)
  du[1] = dx = p[1]*u[1] - p[2]*u[1]*u[2]
  du[2] = dy = -p[3]*u[2] + p[4]*u[1]*u[2]
end

p = [1.5,1.0,3.0,1.0]; u0 = [1.0;1.0]
prob = ODEProblem(lotka!,u0,(0.0,10.0),p)
loss(sol) = sum(sol)
v = ones(4)

Hv = second_order_sensitivity_product(loss,v,prob,Vern9(),saveat=0.1,abstol=1e-12,reltol=1e-12)
```

## Arguments

The arguments for this function match `adjoint_sensitivities`. The only notable difference
is `sensealg` which requires a second order sensitivity algorithm, of which currently the
only choice is `ForwardDiffOverAdjoint` which uses forward-over-reverse to mix a forward-mode
sensitivity analysis with an adjoint sensitivity analysis for a faster computation than either
double forward or double reverse. `ForwardDiffOverAdjoint`'s positional argument just accepts
a first order sensitivity algorithm.
"""
function second_order_sensitivity_product(loss, v, prob, alg, args...;
        sensealg = ForwardDiffOverAdjoint(InterpolatingAdjoint(autojacvec = ReverseDiffVJP())),
        kwargs...)
    _second_order_sensitivity_product(loss, v, prob, alg, sensealg, args...; kwargs...)
end
