# Frequently Asked Questions (FAQ)

## How do I isolate potential gradient issues and improve performance?

If you see the warnings:

```
┌ Warning: Reverse-Mode AD VJP choices all failed. Falling back to numerical VJPs
└ @ SciMLSensitivity C:\Users\accou\.julia\dev\SciMLSensitivity\src\concrete_solve.jl:145
┌ Warning: Potential performance improvement omitted. EnzymeVJP tried and failed in the automated AD choice algorithm. To show the stack trace, set SciMLSensitivity.STACKTRACE_WITH_VJPWARN[] = true. To turn off this printing, add `verbose = false` to the `solve` call.
└ @ SciMLSensitivity C:\Users\accou\.julia\dev\SciMLSensitivity\src\concrete_solve.jl:100
```

then you're in luck! Well, not really. But there are things you can do. You can isolate the
issue to automatic differentiation of your `f` function in order to either fix your `f`
function, or open an issue with the AD library directly without the ODE solver involved.

If you have an in-place function, then you will want to isolate it to Enzyme. This is done
as follows for an arbitrary problem:

```julia
using Enzyme
u0 = prob.u0
p = prob.p
tmp2 = Enzyme.make_zero(p)
t = prob.tspan[1]
du = zero(u0)

if DiffEqBase.isinplace(prob)
    _f = prob.f
else
    _f = (du, u, p, t) -> (du .= prob.f(u, p, t); nothing)
end

_tmp6 = Enzyme.make_zero(_f)
tmp3 = zero(u0)
tmp4 = zero(u0)
ytmp = zero(u0)
tmp1 = zero(u0)

Enzyme.autodiff(Enzyme.Reverse, Enzyme.Duplicated(_f, _tmp6),
    Enzyme.Const, Enzyme.Duplicated(tmp3, tmp4),
    Enzyme.Duplicated(ytmp, tmp1),
    Enzyme.Duplicated(p, tmp2),
    Enzyme.Const(t))
```

This is exactly the inner core Enzyme call and if this fails, that is the issue that
needs to be fixed.

And similarly, for out-of-place functions the Zygote isolation is as follows:

```julia
p = prob.p
y = prob.u0
f = prob.f
λ = zero(prob.u0)
_dy, back = Zygote.pullback(y, p) do u, p
    vec(f(u, p, t))
end
tmp1, tmp2 = back(λ)
```

## When fitting a differential equation how do I visualize the fit during the optimization iterations?

The `Optimization.jl` package has a callback function that can be used to visualize the
progress of the optimization. This is done as follows (pseudo-code):

```julia
callback = function (state, l)
    println(l)
    pl = visualize(state.u)
    display(pl)
    return false
end
```

Earlier we used to allow extra returns from the objective function in addition to the loss value and you could use that in the callback, but this is no longer supported.
This was done to allow support for combined evaluation of the primal (loss value) and the backward pass (gradient) thus making it more efficient by a factor. So now, to
create a plot in the callback, you need to solve the differential equation again (forward pass) inside the callback, this is less expensive than allowing the extra
returns, but it is more expensive than a simple callback that just prints the loss value, and can result in slower optimization.
