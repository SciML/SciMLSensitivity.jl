# taken from https://github.com/SciML/SciMLStructures.jl/pull/28
using OrdinaryDiffEq, SciMLSensitivity, Zygote
using LinearAlgebra
import SciMLStructures as SS

mutable struct SubproblemParameters{P, Q, R}
    p::P # tunable
    q::Q
    r::R
end
mutable struct Parameters{P, C}
    subparams::P
    coeffs::C # tunable matrix
end
# the rhs is `du[i] = p[i] * u[i]^2 + q[i] * u[i] + r[i] * t` for i in 1:length(subparams)
# and `du[length(subparams)+1:end] .= coeffs * u`
function rhs!(du, u, p::Parameters, t)
    for (i, subpars) in enumerate(p.subparams)
        du[i] = subpars.p * u[i]^2 + subpars.q * u[i] + subpars.r * t
    end
    N = length(p.subparams)
    mul!(view(du, (N + 1):(length(du))), p.coeffs, u)
    return nothing
end
u = sin.(0.1:0.1:1.0)
subparams = [SubproblemParameters(0.1i, 0.2i, 0.3i) for i in 1:5]
p = Parameters(subparams, cos.([0.1i + 0.33j for i in 1:5, j in 1:10]))
tspan = (0.0, 1.0)
prob = ODEProblem(rhs!, u, tspan, p)
solve(prob, Tsit5())

# Mark the struct as a SciMLStructure
SS.isscimlstructure(::Parameters) = true
# It is mutable
SS.ismutablescimlstructure(::Parameters) = true
# Only contains `Tunable` portion
# We could also add a `Constants` portion to contain the values that are
# not tunable. The implementation would be similar to this one.
SS.hasportion(::SS.Tunable, ::Parameters) = true
function SS.canonicalize(::SS.Tunable, p::Parameters)
    # concatenate all tunable values into a single vector
    buffer = vcat([subpar.p for subpar in p.subparams], vec(p.coeffs))
    # repack takes a new vector of the same length as `buffer`, and constructs
    # a new `Parameters` object using the values from the new vector for tunables
    # and retaining old values for other parameters. This is exactly what replace does,
    # so we can use that instead.
    repack = let p = p
        function repack(newbuffer)
            SS.replace(SS.Tunable(), p, newbuffer)
        end
    end
    # the canonicalized vector, the repack function, and a boolean indicating
    # whether the buffer aliases values in the parameter object (here, it doesn't)
    return buffer, repack, false
end
function SS.replace(::SS.Tunable, p::Parameters, newbuffer)
    N = length(p.subparams) + length(p.coeffs)
    @assert length(newbuffer) == N
    subparams = [SubproblemParameters(newbuffer[i], subpar.q, subpar.r)
                 for (i, subpar) in enumerate(p.subparams)]
    coeffs = reshape(
        view(newbuffer, (length(p.subparams) + 1):length(newbuffer)), size(p.coeffs))
    return Parameters(subparams, coeffs)
end
function SS.replace!(::SS.Tunable, p::Parameters, newbuffer)
    N = length(p.subparams) + length(p.coeffs)
    @assert length(newbuffer) == N
    for (subpar, val) in zip(p.subparams, newbuffer)
        subpar.p = val
    end
    copyto!(coeffs, view(newbuffer, (length(p.subparams) + 1):length(newbuffer)))
    return p
end

Zygote.gradient(0.1ones(length(SS.canonicalize(SS.Tunable(), p)[1]))) do tunables
    newp = SS.replace(SS.Tunable(), p, tunables)
    newprob = remake(prob; p = newp)
    sol = solve(newprob, Tsit5())
    return sum(sol.u[end])
end
