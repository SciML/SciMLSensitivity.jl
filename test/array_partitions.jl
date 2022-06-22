import OrdinaryDiffEq
import DiffEqBase: DynamicalODEProblem
import DiffEqSensitivity:
    solve,
    ODEProblem,
    ODEAdjointProblem,
    InterpolatingAdjoint,
    ZygoteVJP,
    ReverseDiffVJP
import RecursiveArrayTools: ArrayPartition

sol = solve(
    DynamicalODEProblem(
        (v, x, p, t) -> [0.0, 0.0],

        # ERROR: LoadError: type Nothing has no field x
        # (v, x, p, t) -> [0.0, 0.0],

        # ERROR: LoadError: MethodError: no method matching ndims(::Type{Nothing})
        (v, x, p, t) -> v,

        [0.0, 0.0],
        [0.0, 0.0],
        (0.0, 1.0),
    ),OrdinaryDiffEq.Tsit5()
)

solve(
    ODEAdjointProblem(
        sol,
        InterpolatingAdjoint(autojacvec=ZygoteVJP(allow_nothing=true)),
        (out, x, p, t, i) -> (out .= 0),
        [sol.t[end]],
    ),OrdinaryDiffEq.Tsit5()
)

dyn_v(v_ap, x_ap, p, t) = ArrayPartition(zeros(), [0.0])
# Originally, I imagined that this may be a bug in Zygote, and it still may be, but I tried doing a pullback on this
# function on its own and didn't have any trouble with that. So I'm led to believe that it has something to do with
# how DiffEqSensitivity is invoking Zygote. At least this was as far as I was able to simplify the reproduction.
dyn_x(v_ap, x_ap, p, t) = begin
    # ERROR: LoadError: MethodError: no method matching ndims(::Type{NamedTuple{(:x,),Tuple{Tuple{Nothing,Array{Float64,1}}}}})
    v = v_ap.x[2]

    # ERROR: LoadError: type Nothing has no field x
    # v = [0.0]
    ArrayPartition(zeros(), v)
end

v0 = [-1.0]
x0 = [0.75]

sol = solve(
    DynamicalODEProblem(
        dyn_v,
        dyn_x,
        ArrayPartition(zeros(), v0),
        ArrayPartition(zeros(), x0),
        (0.0, 1.0),
        zeros()
    ),OrdinaryDiffEq.Tsit5(),
    # Without setting parameters, we end up with https://github.com/SciML/DifferentialEquations.jl/issues/679 again.
    p = zeros()
)

g = ArrayPartition(ArrayPartition(zeros(), zero(v0)), ArrayPartition(zeros(), zero(x0)))
bwd_sol = solve(
    ODEAdjointProblem(
        sol,
        InterpolatingAdjoint(autojacvec=ZygoteVJP(allow_nothing=true)),
        # Also fails, but due to a different bug:
        # InterpolatingAdjoint(autojacvec=ReverseDiffVJP()),
        (out, x, p, t, i) -> (out[:] = g),
        [sol.t[end]],
    ),OrdinaryDiffEq.Tsit5()
)