# [Sensitivity analysis for chaotic systems (shadowing methods)](@id shadowing_methods)

Let us define the instantaneous objective ``g(u,p)`` which depends on the state `u`
and the parameter `p` of the differential equation. Then, if the objective is a
long-time average quantity

```math
\langle g \rangle_∞ = \lim_{T \rightarrow ∞} \langle g \rangle_T,
```

where

```math
\langle g \rangle_T = \frac{1}{T} \int_0^T g(u,p) \text{d}t,
```
under the assumption of ergodicity, ``\langle g \rangle_∞`` only depends on `p`.

In the case of chaotic systems, the trajectories diverge with ``O(1)`` error]. This
can be seen, for instance, when solving the [Lorenz system](https://en.wikipedia.org/wiki/Lorenz_system) at
`1e-14` tolerances with 9th order integrators and a small machine-epsilon perturbation:

```@example chaosode
using OrdinaryDiffEq, DiffEqSensitivity, Zygote

function lorenz!(du, u, p, t)
  du[1] = 10 * (u[2] - u[1])
  du[2] = u[1] * (p[1] - u[3]) - u[2]
  du[3] = u[1] * u[2] - (8 // 3) * u[3]
end

p = [28.0]
tspan = (0.0, 100.0)
u0 = [1.0, 0.0, 0.0]
prob = ODEProblem(lorenz!, u0, tspan, p)
sol = solve(prob, Vern9(), abstol = 1e-14, reltol = 1e-14)
sol2 = solve(prob, Vern9(), abstol = 1e-14 + eps(Float64), reltol = 1e-14)
```
![Chaotic behavior of the Lorenz system](../assets/chaos_eps_pert.png)

More formally, such chaotic behavior can be analyzed using tools from
[uncertainty quantification](@ref uncertainty_quantification).
This effect of diverging trajectories is known as the butterfly effect and can be
formulated as "most (small) perturbations on initial conditions or parameters lead
to new trajectories diverging exponentially fast from the original trajectory".

The latter statement can be roughly translated to the level of sensitivity calculation
as follows: "For most initial conditions, the (homogeneous) tangent solutions grow
exponentially fast."

To compute derivatives of an objective ``\langle g \rangle_∞`` with respect to the
parameters `p` of a chaotic systems, one thus encounters that "traditional" forward
and adjoint sensitivity methods diverge because the tangent space diverges with a
rate given by the Lyapunov exponent. Taking the average of these derivative can then
also fail, i.e., one finds that the average derivative is not the derivative of
the average.

Although numerically computed chaotic trajectories diverge from the true/original
trajectory, the [shadowing theorem](http://mathworld.wolfram.com/ShadowingTheorem.html) guarantees that there exists an errorless trajectory
with a slightly different initial condition that stays near ("shadows") the numerically
computed one, see, e.g, the [blog post](https://frankschae.github.io/post/shadowing/) or the [non-intrusive least squares shadowing paper](https://arxiv.org/abs/1611.00880) for more details.
Essentially, the idea is to replace the ill-conditioned ODE by a well-conditioned
optimization problem. Shadowing methods use the shadowing theorem within a renormalization
procedure to distill the long-time effect from the joint observation of the long-time
and the butterfly effect. This allows us to accurately compute derivatives w.r.t.
the long-time average quantities.

The following `sensealg` choices exist

- `ForwardLSS(;LSSregularizer=TimeDilation(10.0,0.0,0.0),g=nothing,ADKwargs...)`:
  An implementation of the forward [least square shadowing](https://arxiv.org/abs/1204.0159) method.
  For `LSSregularizer`, one can choose between two different windowing options,
  `TimeDilation` (default) with weight `10.0` and `CosWindowing`, and `Cos2Windowing`.
- `AdjointLSS(;LSSRegularizer=TimeDilation(10.0, 0.0, 0.0),g=nothing,ADKwargs...)`: An
  implementation of the adjoint-mode [least square shadowing](https://arxiv.org/abs/1204.0159)
  method. `10.0` controls the weight of the time dilation term in `AdjointLSS`.
- `NILSS(nseg,nstep;nus=nothing,rng=Xorshifts.Xoroshiro128Plus(rand(UInt64)),g=nothing,ADKwargs...)`:  
  An implementation of the [non-intrusive least squares shadowing (NILSS)](https://arxiv.org/abs/1611.00880)
  method. Here, `nseg` is the number of segments, `nstep` is the number of steps per
  segment, and `nus` is the number of unstable Lyapunov exponents.
- `NILSAS(nseg,nstep,M=nothing;rng =Xorshifts.Xoroshiro128Plus(rand(UInt64)),
          adjoint_sensealg=BacksolveAdjoint(autojacvec=ReverseDiffVJP()),g=nothing,ADKwargs...)`:  
  An implementation of the [non-intrusive least squares adjoint shadowing (NILSAS)](https://arxiv.org/abs/1801.08674)
  method. `nseg` is the number of segments. `nstep` is the number of steps per
  segment, `M >= nus + 1` has to be provided, where `nus` is the number of unstable
  covariant Lyapunov vectors.

Recommendation: Since the computational and memory costs of `NILSS()` scale with
the number of positive (unstable) Lyapunov, it is typically less expensive than
`ForwardLSS()`. `AdjointLSS()` and `NILSAS()` are favorable for a large number
of system parameters.

As an example, for the Lorenz system with `g(u,p,t) = u[3]`, i.e., the ``z`` coordinate,
as the instantaneous objective, we can use the direct interface by passing `ForwardLSS`
as the `sensealg`:

```@example chaosode
function lorenz!(du,u,p,t)
  du[1] = p[1]*(u[2]-u[1])
  du[2] = u[1]*(p[2]-u[3]) - u[2]
  du[3] = u[1]*u[2] - p[3]*u[3]
end

p = [10.0, 28.0, 8/3]

tspan_init = (0.0,30.0)
tspan_attractor = (30.0,50.0)
u0 = rand(3)
prob_init = ODEProblem(lorenz!,u0,tspan_init,p)
sol_init = solve(prob_init,Tsit5())
prob_attractor = ODEProblem(lorenz!,sol_init[end],tspan_attractor,p)

g(u,p,t) = u[end]

function G(p)
  _prob = remake(prob_attractor,p=p)
  _sol = solve(_prob,Vern9(),abstol=1e-14,reltol=1e-14,saveat=0.01,sensealg=ForwardLSS(g=g))
  sum(getindex.(_sol.u,3))
end
dp1 = Zygote.gradient(p->G(p),p)
```

Alternatively, we can define the `ForwardLSSProblem` and solve it
via `shadow_forward` as follows:

```@example chaosode
sol_attractor = solve(prob_attractor, Vern9(), abstol=1e-14, reltol=1e-14)
lss_problem = ForwardLSSProblem(sol_attractor, ForwardLSS(g=g))
resfw = shadow_forward(lss_problem)
```
