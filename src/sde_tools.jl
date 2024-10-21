# for Ito / Stratonovich conversion
struct StochasticTransformedFunction{pType, fType <: AbstractDiffEqFunction,
    gType, noiseType, cfType} <: TransformedFunction
    prob::pType
    f::fType
    g::gType
    gtmp::noiseType
    inplace::Bool
    corfunc_analytical::cfType
end

function StochasticTransformedFunction(sol, f, g, corfunc_analytical = nothing)
    (; prob) = sol

    if SciMLBase.is_diagonal_noise(prob)
        gtmp = copy(sol.u[end])
    else
        gtmp = similar(prob.p, size(prob.noise_rate_prototype))
    end

    return StochasticTransformedFunction(prob, f, g, gtmp, DiffEqBase.isinplace(prob),
        corfunc_analytical)
end

function (Tfunc::StochasticTransformedFunction)(du, u, p, t)
    (; gtmp, f, g, corfunc_analytical) = Tfunc

    ducor = similar(u, size(u))

    if corfunc_analytical !== nothing
        corfunc_analytical(ducor, u, p, t)
    else
        tape = ReverseDiff.GradientTape((u, p, [t])) do uloc, ploc, tloc
            du1 = similar(uloc, size(gtmp))
            g(du1, uloc, ploc, first(tloc))
            return vec(du1)
        end
        tu, tp, tt = ReverseDiff.input_hook(tape)
        output = ReverseDiff.output_hook(tape)

        ReverseDiff.unseed!(tu) # clear any "leftover" derivatives from previous calls
        ReverseDiff.unseed!(tp)
        ReverseDiff.unseed!(tt)

        ReverseDiff.value!(tu, u)
        ReverseDiff.value!(tp, p)
        ReverseDiff.value!(tt, [t])

        ReverseDiff.forward_pass!(tape)
        ReverseDiff.increment_deriv!(output, vec(ReverseDiff.value(output)))
        ReverseDiff.reverse_pass!(tape)

        ReverseDiff.deriv(tu)
        ReverseDiff.pull_value!(output)
        copyto!(vec(ducor), ReverseDiff.deriv(tu))
    end

    f(du, u, p, t)

    @. du = du - ducor
    return nothing
end

function (Tfunc::StochasticTransformedFunction)(u, p, t)
    (; f, g, corfunc_analytical) = Tfunc
    #ducor = vecjacobian(u, p, t, Tfunc)

    if corfunc_analytical !== nothing
        ducor = corfunc_analytical(u, p, t)
    else
        _dy, back = Zygote.pullback(u, p) do uloc, ploc
            vec(g(uloc, ploc, t))
        end
        ducor, _ = back(_dy)
    end
    du = f(u, p, t)

    ducor !== nothing && (du = @. du - ducor)
    return du
end
