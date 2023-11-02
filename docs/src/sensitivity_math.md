# [Mathematics of Sensitivity Analysis](@id sensitivity_math)

## Forward Sensitivity Analysis

The local sensitivity is computed using the sensitivity ODE:

```math
\frac{d}{dt}\frac{\partial u}{\partial p_{j}}=\frac{\partial f}{\partial u}\frac{\partial u}{\partial p_{j}}+\frac{\partial f}{\partial p_{j}}=J\cdot S_{j}+F_{j}
```

where

```math
J=\left(\begin{array}{cccc}
\frac{\partial f_{1}}{\partial u_{1}} & \frac{\partial f_{1}}{\partial u_{2}} & \cdots & \frac{\partial f_{1}}{\partial u_{k}}\\
\frac{\partial f_{2}}{\partial u_{1}} & \frac{\partial f_{2}}{\partial u_{2}} & \cdots & \frac{\partial f_{2}}{\partial u_{k}}\\
\cdots & \cdots & \cdots & \cdots\\
\frac{\partial f_{k}}{\partial u_{1}} & \frac{\partial f_{k}}{\partial u_{2}} & \cdots & \frac{\partial f_{k}}{\partial u_{k}}
\end{array}\right)
```

is the Jacobian of the system,

```math
F_{j}=\left(\begin{array}{c}
\frac{\partial f_{1}}{\partial p_{j}}\\
\frac{\partial f_{2}}{\partial p_{j}}\\
\vdots\\
\frac{\partial f_{k}}{\partial p_{j}}
\end{array}\right)
```

are the parameter derivatives, and

```math
S_{j}=\left(\begin{array}{c}
\frac{\partial u_{1}}{\partial p_{j}}\\
\frac{\partial u_{2}}{\partial p_{j}}\\
\vdots\\
\frac{\partial u_{k}}{\partial p_{j}}
\end{array}\right)
```

is the vector of sensitivities. Since this ODE is dependent on the values of the
independent variables themselves, this ODE is computed simultaneously with the
actual ODE system.

Note that the Jacobian-vector product

```math
\frac{\partial f}{\partial u}\frac{\partial u}{\partial p_{j}}
```

can be computed without forming the Jacobian. With finite differences, this can be done using the following
formula for the directional derivative

```math
Jv \approx \frac{f(x+v \epsilon) - f(x)}{\epsilon},
```

alternatively and without truncation error,
by using a dual number with a single partial dimension, ``d = x + v \epsilon`` we get that

```math
f(d) = f(x) + Jv \epsilon
```

as a fast way to calculate ``Jv``. Thus, except when a sufficiently good function for `J` is given
by the user, the Jacobian is never formed. For more details, consult the
[MIT 18.337 lecture notes on forward mode AD](https://book.sciml.ai/notes/08-Forward-Mode_Automatic_Differentiation_(AD)_via_High_Dimensional_Algebras/).

## Adjoint Sensitivity Analysis

This adjoint requires the definition of some scalar functional ``g(u,p)``
where ``u(t,p)`` is the (numerical) solution to the differential equation
``\frac{du(t,p)}{dt}=f(t,u,p)`` with ``t\in [0,T]`` and ``u(t_0,p)=u_0``.
Adjoint sensitivity analysis finds the gradient of

```math
G(u,p)=G(u(\cdot,p))=\int_{t_{0}}^{T}g(u(t,p),p)dt
```

some integral of the solution. It does so by solving the adjoint problem

```math
\frac{d\lambda^{\star}}{dt}=g_{u}(u(t,p),p)-\lambda^{\star}(t)f_{u}(t,u(t,p),p),\thinspace\thinspace\thinspace\lambda^{\star}(T)=0
```

where ``f_u`` is the Jacobian of the system with respect to the state ``u`` while
``f_p`` is the Jacobian with respect to the parameters. The adjoint problem's
solution gives the sensitivities through the integral:

```math
\frac{dG}{dp}=\int_{t_{0}}^{T}\lambda^{\star}(t)f_{p}(t)+g_{p}(t)dt+\lambda^{\star}(t_{0})u_{p}(t_{0})
```

Notice that since the adjoints require the Jacobian of the system at the state,
it requires the ability to evaluate the state at any point in time. Thus it
requires the continuous forward solution in order to solve the adjoint solution,
and the adjoint solution is required to be continuous in order to calculate the
resulting integral.

There is one extra detail to consider. In many cases, we would like to calculate
the adjoint sensitivity of some discontinuous functional of the solution. One
canonical function is the L2 loss against some data points, that is:

```math
L(u,p)=\sum_{i=1}^{n}\Vert\tilde{u}(t_{i})-u(t_{i},p)\Vert^{2}
```

In this case, we can reinterpret our summation as the distribution integral:

```math
G(u,p)=\int_{0}^{T}\sum_{i=1}^{n}\Vert\tilde{u}(t_{i})-u(t_{i},p)\Vert^{2}\delta(t_{i}-t)dt
```

where ``δ`` is the Dirac distribution. In this case, the integral is continuous
except at finitely many points. Thus it can be calculated between each ``t_i``.
At a given ``t_i``, given that the ``t_i`` are unique, we have that

```math
g_{u}(t_{i})=2\left(\tilde{u}(t_{i})-u(t_{i},p)\right)
```

Thus the adjoint solution ``\lambda^{\star}(t)`` is given by integrating between the integrals and
applying the jump function ``g_u`` at every data point ``t_i``.

We note that

```math
\lambda^{\star}(t)f_{u}(t)
```

is a vector-transpose Jacobian product, also known as a `vjp`, which can be efficiently computed
using the pullback of backpropagation on the user function `f` with a forward pass at `u` with a
pullback vector ``\lambda^{\star}``. For more information, consult the
[MIT 18.337 lecture notes on reverse mode AD](https://book.sciml.ai/notes/10-Basic_Parameter_Estimation-Reverse-Mode_AD-and_Inverse_Problems/).
