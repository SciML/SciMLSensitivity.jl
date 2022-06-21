## Direct calls

const ADJOINT_PARAMETER_COMPATABILITY_MESSAGE =
"""
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
  print(io, ADJOINT_PARAMETER_COMPATABILITY_MESSAGE)
end

@doc doc"""
adjoint_sensitivities(sol,alg,g,t=nothing,dg=nothing;
                            abstol=1e-6,reltol=1e-3,
                            checkpoints=sol.t,
                            corfunc_analytical=nothing,
                            callback = nothing,
                            sensealg=InterpolatingAdjoint(),
                            kwargs...)

Adjoint sensitivity analysis is used to find the gradient of the solution
with respect to some functional of the solution. In many cases this is used
in an optimization problem to return the gradient with respect to some cost
function. It is equivalent to "backpropagation" or reverse-mode automatic
differentiation of a differential equation.

Using `adjoint_sensitivities` directly let's you do three things. One it can
allow you to be more efficient, since the sensitivity calculation can be done
directly on a cost function, avoiding the overhead of building the derivative
of the full concretized solution. It can also allow you to be more efficient
by directly controlling the forward solve that is then reversed over. Lastly,
it allows one to define a continuous cost function on the continuous solution,
instead of just at discrete data points.

!!! warning

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

!!! warning

      Non-checkpointed InterpolatingAdjoint and QuadratureAdjoint sensealgs
      require that the forward solution `sol(t)` has an accurate dense
      solution unless checkpointing is used. This means that you should
      not use `solve(prob,alg,saveat=ts)` unless checkpointing. If specific
      saving is required, one should solve dense `solve(prob,alg)`, use the
      solution in the adjoint, and then `sol(ts)` interpolate.

### Syntax

There are two forms. For discrete adjoints, the form is:

```julia
du0,dp = adjoint_sensitivities(sol,alg,dg,ts;sensealg=InterpolatingAdjoint(),
                               checkpoints=sol.t,kwargs...)
```

where `alg` is the ODE algorithm to solve the adjoint problem, `dg` is the jump
function, `sensealg` is the sensitivity algorithm, and `ts` is the time points
for data. `dg` is given by:

```julia
dg(out,u,p,t,i)
```

which is the in-place gradient of the cost functional `g` at time point `ts[i]`
with `u=u(t)`.

For continuous functionals, the form is:

```julia
du0,dp = adjoint_sensitivities(sol,alg,g,nothing,(dgdu,dgdp);sensealg=InterpolatingAdjoint(),
                               checkpoints=sol.t,,kwargs...)
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
du0,dp = adjoint_sensitivities(sol,alg,g,nothing;kwargs...)
```

then we assume `dgdp` is zero and `dgdu` will be computed automatically using ForwardDiff or finite
differencing, depending on the `autodiff` setting in the `AbstractSensitivityAlgorithm`.
Note that the keyword arguments are passed to the internal ODE solver for
solving the adjoint problem.

### Example discrete adjoints on a cost function

In this example we will show solving for the adjoint sensitivities of a discrete
cost functional. First let's solve the ODE and get a high quality continuous
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
dg_{1}&=1-u_{1} \\
dg_{2}&=1-u_{2} \\
& \quad \vdots
\end{aligned}
```

and thus:

```julia
dg(out,u,p,t,i) = (out.=1.0.-u)
```

Also, we can omit `dgdp`, because the cost function doesn't dependent on `p`. If we had data, we'd just replace `1.0` with `data[i]`. To get the adjoint
sensitivities, call:

```julia
ts = 0:0.5:10
res = adjoint_sensitivities(sol,Vern9(),dg,ts,abstol=1e-14,
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
method. This maximizes speed but at a cost of requiring a dense `sol`. If it
is not possible to hold a dense forward solution in memory, then one can use
checkpointing. For example:

```julia
ts = [0.0,0.2,0.5,0.7]
sol = solve(prob,Vern9(),saveat=ts)
```

Creates a non-dense solution with checkpoints at `[0.0,0.2,0.5,0.7]`. Now we
can do:

```julia
res = adjoint_sensitivities(sol,Vern9(),dg,ts,
                            sensealg=InterpolatingAdjoint(checkpointing=true))
```

When grabbing a Jacobian value during the backwards solution, it will no longer
interpolate to get the value. Instead, it will start a forward solution at the
nearest checkpoint to build local interpolants in a way that conserves memory.
By default the checkpoints are at `sol.t`, but we can override this:

```julia
res = adjoint_sensitivities(sol,Vern9(),dg,ts,
                            sensealg=InterpolatingAdjoint(checkpointing=true),
                            checkpoints = [0.0,0.5])
```

### Example continuous adjoints on an energy functional

In this case we'd like to calculate the adjoint sensitivity of the scalar energy
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
res = adjoint_sensitivities(sol,Vern9(),g,nothing,dg,abstol=1e-8,
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
function adjoint_sensitivities(sol,args...;
                                  sensealg=InterpolatingAdjoint(),
                                  verbose=true,kwargs...)
  if hasfield(typeof(sensealg),:autojacvec) && sensealg.autojacvec === nothing
    if haskey(kwargs, :callback)
      has_cb = kwargs[:callback] !== nothing
    else
      has_cb = false
    end
    if !has_cb
      _sensealg = if isinplace(sol.prob)
        setvjp(sensealg,inplace_vjp(sol.prob,sol.prob.u0,sol.prob.p,verbose))
      else
        setvjp(sensealg,ZygoteVJP())
      end
    else
      _sensealg = setvjp(sensealg, ReverseDiffVJP())
    end

    return try
      _adjoint_sensitivities(sol,_sensealg,args...;verbose,kwargs...)
    catch e
      verbose && @warn "Automatic AD choice of autojacvec failed in ODE adjoint, failing back to ODE adjoint + numerical vjp"
      _adjoint_sensitivities(sol,setvjp(sensealg,false),args...;verbose,kwargs...)
    end
  else
    return _adjoint_sensitivities(sol,sensealg,args...;verbose,kwargs...)
  end
end

function _adjoint_sensitivities(sol,sensealg,alg,g,t=nothing,dg=nothing;
                                   abstol=1e-6,reltol=1e-3,
                                   checkpoints=sol.t,
                                   corfunc_analytical=nothing,
                                   callback = nothing,
                                   kwargs...)

  if !(typeof(sol.prob.p) <: Union{Nothing,SciMLBase.NullParameters,AbstractArray}) || (sol.prob.p isa AbstractArray && !Base.isconcretetype(eltype(sol.prob.p)))
    throw(AdjointSensitivityParameterCompatibilityError())
  end

  if sol.prob isa ODEProblem
    adj_prob = ODEAdjointProblem(sol,sensealg,g,t,dg; checkpoints=checkpoints,
                                 callback = callback,
                                 abstol=abstol,reltol=reltol, kwargs...)

  elseif sol.prob isa SDEProblem
    adj_prob = SDEAdjointProblem(sol,sensealg,g,t,dg,checkpoints=checkpoints,
                                 callback = callback,
                                 abstol=abstol,reltol=reltol,
                                 corfunc_analytical=corfunc_analytical)
  elseif sol.prob isa RODEProblem
    adj_prob = RODEAdjointProblem(sol,sensealg,g,t,dg,checkpoints=checkpoints,
                                callback = callback,
                                abstol=abstol,reltol=reltol,
                                corfunc_analytical=corfunc_analytical)
  else
    error("Continuous adjoint sensitivities are only supported for ODE/SDE/RODE problems.")
  end

  tstops = ischeckpointing(sensealg, sol) ? checkpoints : similar(sol.t, 0)
  adj_sol = solve(adj_prob,alg;
                  save_everystep=false,save_start=false,saveat=eltype(sol[1])[],
                  tstops=tstops,abstol=abstol,reltol=reltol,kwargs...)

  p = sol.prob.p
  l = p === nothing || p === DiffEqBase.NullParameters() ? 0 : length(sol.prob.p)
  du0 = adj_sol[end][1:length(sol.prob.u0)]

  if eltype(sol.prob.p) <: real(eltype(adj_sol[end]))
    dp = real.(adj_sol[end][(1:l) .+ length(sol.prob.u0)])'
  elseif p === nothing || p === DiffEqBase.NullParameters()
    dp = nothing
  else
    dp = adj_sol[end][(1:l) .+ length(sol.prob.u0)]'
  end

  du0,dp
end

function _adjoint_sensitivities(sol,sensealg::SteadyStateAdjoint,alg,g,dg=nothing;
                                   abstol=1e-6,reltol=1e-3,
                                   kwargs...)
  SteadyStateAdjointProblem(sol,sensealg,g,dg;kwargs...)
end

function _adjoint_sensitivities(sol,sensealg::SteadyStateAdjoint,alg;
                                   g=nothing,dg=nothing,
                                   abstol=1e-6,reltol=1e-3,
                                   kwargs...)
  SteadyStateAdjointProblem(sol,sensealg,g,dg;kwargs...)
end

@doc doc"""
H = second_order_sensitivities(loss,prob,alg,args...;
                               sensealg=ForwardDiffOverAdjoint(InterpolatingAdjoint(autojacvec=ReverseDiffVJP())),
                               kwargs...)

Second order sensitivity analysis is used for the fast calculation of Hessian
matrices.

!!! warning

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

### Example second order sensitivity analysis calculation

```julia
using DiffEqSensitivity, OrdinaryDiffEq, ForwardDiff
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
function second_order_sensitivities(loss,prob,alg,args...;
                                    sensealg=ForwardDiffOverAdjoint(InterpolatingAdjoint(autojacvec=ReverseDiffVJP())),
                                    kwargs...)
  _second_order_sensitivities(loss,prob,alg,sensealg,args...;kwargs...)
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
      a differential equation defined by the parameter struct `p`. Thus while
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
using DiffEqSensitivity, OrdinaryDiffEq, ForwardDiff
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
function second_order_sensitivity_product(loss,v,prob,alg,args...;
                                          sensealg=ForwardDiffOverAdjoint(InterpolatingAdjoint(autojacvec=ReverseDiffVJP())),
                                          kwargs...)
  _second_order_sensitivity_product(loss,v,prob,alg,sensealg,args...;kwargs...)
end
