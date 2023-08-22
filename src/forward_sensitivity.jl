"""
ODEForwardSensitivityFunction{iip,F,A,Tt,OJ,J,JP,S,PJ,TW,TWt,UF,PF,JC,PJC,Alg,fc,JM,pJM,MM,CV} <: DiffEqBase.AbstractODEFunction{iip}

ODEForwardSensitivityFunction is an internal to the ODEForwardSensitivityProblem which extends the AbstractODEFunction
to be used in an ODEProblem, but defines the tools requires for calculating the extra differential equations associated
with the derivative terms.

ODEForwardSensitivityFunction is not intended to be part of the public API.
"""
struct ODEForwardSensitivityFunction{iip, F, A, Tt, OJ, J, JP, S, PJ, TW, TWt, UF, PF, JC,
    PJC, Alg, fc, JM, pJM, MM, CV} <:
       DiffEqBase.AbstractODEFunction{iip}
    f::F
    analytic::A
    tgrad::Tt
    original_jac::OJ
    jac::J
    jac_prototype::JP
    sparsity::S
    paramjac::PJ
    Wfact::TW
    Wfact_t::TWt
    uf::UF
    pf::PF
    J::JM
    pJ::pJM
    jac_config::JC
    paramjac_config::PJC
    alg::Alg
    numparams::Int
    numindvar::Int
    f_cache::fc
    mass_matrix::MM
    isautojacvec::Bool
    isautojacmat::Bool
    colorvec::CV
end

TruncatedStacktraces.@truncate_stacktrace ODEForwardSensitivityFunction

has_original_jac(S) = isdefined(S, :original_jac) && S.original_jac !== nothing

struct NILSSForwardSensitivityFunction{iip, sensefunType, senseType, MM} <:
       DiffEqBase.AbstractODEFunction{iip}
    S::sensefunType
    sensealg::senseType
    nus::Int
    mass_matrix::MM
end

function ODEForwardSensitivityFunction(f, analytic, tgrad, original_jac, jac, jac_prototype,
    sparsity, paramjac, Wfact, Wfact_t, uf, pf, u0,
    jac_config, paramjac_config, alg, p, f_cache, mm,
    isautojacvec, isautojacmat, colorvec, nus)
    numparams = length(p)
    numindvar = length(u0)
    J = isautojacvec ? nothing : Matrix{eltype(u0)}(undef, numindvar, numindvar)
    pJ = Matrix{eltype(u0)}(undef, numindvar, numparams) # number of funcs size

    sensefun = ODEForwardSensitivityFunction{isinplace(f), typeof(f), typeof(analytic),
        typeof(tgrad), typeof(original_jac),
        typeof(jac), typeof(jac_prototype),
        typeof(sparsity),
        typeof(paramjac),
        typeof(Wfact), typeof(Wfact_t), typeof(uf),
        typeof(pf), typeof(jac_config),
        typeof(paramjac_config), typeof(alg),
        typeof(f_cache),
        typeof(J), typeof(pJ), typeof(mm),
        typeof(f.colorvec)}(f, analytic, tgrad,
        original_jac, jac,
        jac_prototype,
        sparsity, paramjac, Wfact,
        Wfact_t, uf, pf, J, pJ,
        jac_config,
        paramjac_config, alg,
        numparams, numindvar,
        f_cache, mm, isautojacvec,
        isautojacmat, colorvec)
    if nus !== nothing
        sensefun = NILSSForwardSensitivityFunction{isinplace(f), typeof(sensefun),
            typeof(alg), typeof(mm)}(sensefun, alg,
            nus, mm)
    end

    return sensefun
end

function (S::ODEForwardSensitivityFunction)(du, u, p, t)
    y = @view u[1:(S.numindvar)] # These are the independent variables
    dy = @view du[1:(S.numindvar)]
    S.f(dy, y, p, t) # Make the first part be the ODE

    # Now do sensitivities
    # Compute the Jacobian

    if !S.isautojacvec && !S.isautojacmat
        if has_original_jac(S)
            S.original_jac(S.J, y, p, t) # Calculate the Jacobian into J
        else
            S.uf.t = t
            jacobian!(S.J, S.uf, y, S.f_cache, S.alg, S.jac_config)
        end
    end

    if DiffEqBase.has_paramjac(S.f)
        S.paramjac(S.pJ, y, p, t) # Calculate the parameter Jacobian into pJ
    else
        S.pf.t = t
        copyto!(S.pf.u, y)
        jacobian!(S.pJ, S.pf, p, S.f_cache, S.alg, S.paramjac_config)
    end

    # Compute the parameter derivatives
    if !S.isautojacvec && !S.isautojacmat
        dp = @view du[reshape((S.numindvar + 1):((length(p) + 1) * S.numindvar),
            S.numindvar, length(p))]
        Sj = @view u[reshape((S.numindvar + 1):((length(p) + 1) * S.numindvar), S.numindvar,
            length(p))]
        mul!(dp, S.J, Sj)
        DiffEqBase.@.. dp += S.pJ
    elseif S.isautojacmat
        S.uf.t = t
        Sj = @view u[reshape((S.numindvar + 1):end, S.numindvar, S.numparams)]
        dp = @view du[reshape((S.numindvar + 1):end, S.numindvar, S.numparams)]
        jacobianmat!(dp, S.uf, y, Sj, S.alg, S.jac_config)
        DiffEqBase.@.. dp += S.pJ
    else
        S.uf.t = t
        for i in eachindex(p)
            Sj = @view u[(i * S.numindvar + 1):((i + 1) * S.numindvar)]
            dp = @view du[(i * S.numindvar + 1):((i + 1) * S.numindvar)]
            jacobianvec!(dp, S.uf, y, Sj, S.alg, S.jac_config)
            dp .+= @view S.pJ[:, i]
        end
    end
    return nothing
end
@deprecate ODELocalSensitivityProblem(args...; kwargs...) ODEForwardSensitivityProblem(args...;
    kwargs...)

struct ODEForwardSensitivityProblem{iip, A}
    sensealg::A
end

function ODEForwardSensitivityProblem(f::F, args...; kwargs...) where {F}
    ODEForwardSensitivityProblem(ODEFunction(f), args...; kwargs...)
end

function ODEForwardSensitivityProblem(prob::ODEProblem, alg; kwargs...)
    ODEForwardSensitivityProblem(prob.f, prob.u0, prob.tspan, prob.p, alg; kwargs...)
end

const FORWARD_SENSITIVITY_PARAMETER_COMPATABILITY_MESSAGE = """
                                                            ODEForwardSensitivityProblem requires being able to solve
                                                            a differential equation defined by the parameter struct `p`. Even though
                                                            DifferentialEquations.jl can support any parameter struct type, usage
                                                            with ODEForwardSensitivityProblem requires that `p` could be a valid
                                                            type for being the initial condition `u0` of an array. This means that
                                                            many simple types, such as `Tuple`s and `NamedTuple`s, will work as
                                                            parameters in normal contexts but will fail during ODEForwardSensitivityProblem
                                                            construction. To work around this issue for complicated cases like nested structs,
                                                            look into defining `p` using `AbstractArray` libraries such as RecursiveArrayTools.jl
                                                            or ComponentArrays.jl.
                                                            """

struct ForwardSensitivityParameterCompatibilityError <: Exception end

function Base.showerror(io::IO, e::ForwardSensitivityParameterCompatibilityError)
    print(io, FORWARD_SENSITIVITY_PARAMETER_COMPATABILITY_MESSAGE)
end

const FORWARD_SENSITIVITY_OUT_OF_PLACE_MESSAGE = """
                                                 ODEForwardSensitivityProblem is not compatible with out of place ODE definitions,
                                                 i.e. `du=f(u,p,t)` definitions. It requires an in-place mutating function
                                                 `f(du,u,p,t)`. For more information on in-place vs out-of-place ODE definitions,
                                                 see the ODEProblem or ODEFunction documentation.
                                                 """

struct ForwardSensitivityOutOfPlaceError <: Exception end

function Base.showerror(io::IO, e::ForwardSensitivityOutOfPlaceError)
    print(io, FORWARD_SENSITIVITY_OUT_OF_PLACE_MESSAGE)
end

@doc doc"""
function ODEForwardSensitivityProblem(f::Union{Function,DiffEqBase.AbstractODEFunction},
                                      u0,tspan,p=nothing,
                                      alg::AbstractForwardSensitivityAlgorithm = ForwardSensitivity();
                                      kwargs...)

Local forward sensitivity analysis gives a solution along with a timeseries of
the sensitivities. Thus, if one wishes to have a derivative at every possible
time point, directly using the `ODEForwardSensitivityProblem` can be the most
efficient method.

!!! warning

      ODEForwardSensitivityProblem requires being able to solve
      a differential equation defined by the parameter struct `p`. Even though
      DifferentialEquations.jl can support any parameter struct type, usage
      with ODEForwardSensitivityProblem requires that `p` could be a valid
      type for being the initial condition `u0` of an array. This means that
      many simple types, such as `Tuple`s and `NamedTuple`s, will work as
      parameters in normal contexts but will fail during ODEForwardSensitivityProblem
      construction. To work around this issue for complicated cases like nested structs,
      look into defining `p` using `AbstractArray` libraries such as RecursiveArrayTools.jl
      or ComponentArrays.jl.

### ODEForwardSensitivityProblem Syntax

`ODEForwardSensitivityProblem` is similar to an `ODEProblem`, but takes an
`AbstractForwardSensitivityAlgorithm` that describes how to append the forward sensitivity
equation calculation to the time evolution to simultaneously compute the derivative
of the solution with respect to parameters.

```julia
ODEForwardSensitivityProblem(f::SciMLBase.AbstractODEFunction,u0,
                             tspan,p=nothing,
                             sensealg::AbstractForwardSensitivityAlgorithm = ForwardSensitivity();
                             kwargs...)
```

Once constructed, this problem can be used in `solve` just like any other ODEProblem.
The solution can be deconstructed into the ODE solution and sensitivities parts using the
`extract_local_sensitivities` function, with the following dispatches:

```julia
extract_local_sensitivities(sol, asmatrix::Val=Val(false)) # Decompose the entire time series
extract_local_sensitivities(sol, i::Integer, asmatrix::Val=Val(false)) # Decompose sol[i]
extract_local_sensitivities(sol, t::Union{Number,AbstractVector}, asmatrix::Val=Val(false)) # Decompose sol(t)
```

For information on the mathematics behind these calculations, consult
[the sensitivity math page](@ref sensitivity_math)

### Example using an ODEForwardSensitivityProblem

To define a sensitivity problem, simply use the `ODEForwardSensitivityProblem` type
instead of an ODE type. For example, we generate an ODE with the sensitivity
equations attached to the Lotka-Volterra equations by:

```julia
function f(du,u,p,t)
  du[1] = dx = p[1]*u[1] - p[2]*u[1]*u[2]
  du[2] = dy = -p[3]*u[2] + u[1]*u[2]
end

p = [1.5,1.0,3.0]
prob = ODEForwardSensitivityProblem(f,[1.0;1.0],(0.0,10.0),p)
```

This generates a problem which the ODE solvers can solve:

```julia
sol = solve(prob,DP8())
```

Note that the solution is the standard ODE system and the sensitivity system combined.
We can use the following helper functions to extract the sensitivity information:

```julia
x,dp = extract_local_sensitivities(sol)
x,dp = extract_local_sensitivities(sol,i)
x,dp = extract_local_sensitivities(sol,t)
```

In each case, `x` is the ODE values and `dp` is the matrix of sensitivities
The first gives the full timeseries of values and `dp[i]` contains the time series of the
sensitivities of all components of the ODE with respect to `i`th parameter.
The second returns the `i`th time step, while the third
interpolates to calculate the sensitivities at time `t`. For example, if we do:

```julia
x,dp = extract_local_sensitivities(sol)
da = dp[1]
```

then `da` is the timeseries for ``\frac{\partial u(t)}{\partial p}``. We can
plot this

```julia
plot(sol.t,da',lw=3)
```

transposing so that the rows (the timeseries) is plotted.

![Local Sensitivity Solution](https://user-images.githubusercontent.com/1814174/170916167-11d1b5c6-3c3c-439a-92af-d3899e24d2ad.png)

Here we see that there is a periodicity to the sensitivity which matches
the periodicity of the Lotka-Volterra solutions. However, as time goes on, the
sensitivity increases. This matches the analysis of Wilkins in Sensitivity
Analysis for Oscillating Dynamical Systems.

We can also quickly see that these values are equivalent to those given by
automatic differentiation and numerical differentiation through the ODE solver:

```julia
using ForwardDiff, Calculus
function test_f(p)
  prob = ODEProblem(f,eltype(p).([1.0,1.0]),eltype(p).((0.0,10.0)),p)
  solve(prob,Vern9(),abstol=1e-14,reltol=1e-14,save_everystep=false)[end]
end

p = [1.5,1.0,3.0]
fd_res = ForwardDiff.jacobian(test_f,p)
calc_res = Calculus.finite_difference_jacobian(test_f,p)
```

Here we just checked the derivative at the end point.

### Internal representation of the Solution

For completeness, we detail the internal representation. When using
ForwardDiffSensitivity, the representation is with `Dual` numbers under the
standard interpretation. The values for the ODE's solution at time `i` are the
`ForwardDiff.value.(sol[i])` portions, and the derivative with respect to
parameter `j` is given by `ForwardDiff.partials.(sol[i])[j]`.

When using ForwardSensitivity, the solution to the ODE are the first `n`
components of the solution. This means we can grab the matrix of solution
values like:

```julia
x = sol[1:sol.prob.indvars,:]
```

Since each sensitivity is a vector of derivatives for each function, the sensitivities
are each of size `sol.prob.indvars`. We can pull out the parameter sensitivities from
the solution as follows:

```julia
da = sol[sol.prob.indvars+1:sol.prob.indvars*2,:]
db = sol[sol.prob.indvars*2+1:sol.prob.indvars*3,:]
dc = sol[sol.prob.indvars*3+1:sol.prob.indvars*4,:]
```

This means that `da[1,i]` is the derivative of the `x(t)` by the parameter `a`
at time `sol.t[i]`. Note that all the functionality available to ODE solutions
is available in this case, including interpolations and plot recipes (the recipes
will plot the expanded system).
"""
function ODEForwardSensitivityProblem(f::F, u0,
    tspan, p = nothing,
    alg::ForwardSensitivity = ForwardSensitivity();
    nus = nothing, # determine if Nilss is used
    w0 = nothing,
    v0 = nothing,
    kwargs...) where {F <: DiffEqBase.AbstractODEFunction}
    isinplace = SciMLBase.isinplace(f)
    # if there is an analytical Jacobian provided, we are not going to do automatic `jac*vec`
    isautojacmat = get_jacmat(alg)
    isautojacvec = get_jacvec(alg)
    p === nothing &&
        error("You must have parameters to use parameter sensitivity calculations!")

    if !(typeof(p) <: Union{Nothing, SciMLBase.NullParameters, AbstractArray}) ||
       (p isa AbstractArray && !Base.isconcretetype(eltype(p)))
        throw(ForwardSensitivityParameterCompatibilityError())
    end

    uf = DiffEqBase.UJacobianWrapper(unwrapped_f(f), tspan[1], p)
    pf = DiffEqBase.ParamJacobianWrapper(unwrapped_f(f), tspan[1], copy(u0))
    if isautojacmat
        if alg_autodiff(alg)
            jac_config_seed = ForwardDiff.Dual{
                typeof(uf),
            }.(u0,
                [ntuple(x -> zero(eltype(u0)), length(p))
                 for i in eachindex(u0)])
            jac_config_buffer = similar(jac_config_seed)
            jac_config = jac_config_seed, jac_config_buffer
        else
            error("Jacobian matrix products only work with automatic differentiation.")
        end
    elseif isautojacvec
        if alg_autodiff(alg)
            # if we are using automatic `jac*vec`, then we need to use a `jac_config`
            # that is a tuple in the form of `(seed, buffer)`
            jac_config_seed = ForwardDiff.Dual{typeof(jacobianvec!)}.(u0, u0)
            jac_config_buffer = similar(jac_config_seed)
            jac_config = jac_config_seed, jac_config_buffer
        else
            jac_config = (similar(u0), similar(u0))
        end
    elseif DiffEqBase.has_jac(f)
        jac_config = nothing
    else
        jac_config = build_jac_config(alg, uf, u0)
    end

    if DiffEqBase.has_paramjac(f)
        paramjac_config = nothing
    else
        paramjac_config = build_param_jac_config(alg, pf, u0, p)
    end

    # TODO: make it better
    if f.mass_matrix isa UniformScaling
        mm = f.mass_matrix
    else
        nn = size(f.mass_matrix, 1)
        mm = zeros(eltype(f.mass_matrix), (length(p) + 1) * nn, (length(p) + 1) * nn)
        mm[1:nn, 1:nn] = f.mass_matrix
        for i in 1:length(p)
            mm[(i * nn + 1):((i + 1)nn), (i * nn + 1):((i + 1)nn)] = f.mass_matrix
        end
    end

    # TODO: Use user tgrad. iW can be safely ignored here.
    sense = ODEForwardSensitivityFunction(f, f.analytic, nothing, f.jac, nothing,
        nothing, nothing, f.paramjac,
        nothing, nothing,
        uf, pf, u0, jac_config,
        paramjac_config, alg,
        p, similar(u0), mm,
        isautojacvec, isautojacmat, f.colorvec, nus)

    if !SciMLBase.isinplace(sense)
        throw(ForwardSensitivityOutOfPlaceError())
    end

    if nus === nothing
        sense_u0 = [u0; zeros(eltype(u0), sense.numindvar * sense.numparams)]
    else
        if w0 === nothing && v0 === nothing
            sense_u0 = [u0;
                zeros(eltype(u0),
                (nus + 1) * sense.S.numindvar * sense.S.numparams)]
        else
            sense_u0 = [u0; w0; v0]
        end
    end
    ODEProblem(sense, sense_u0, tspan, p,
        ODEForwardSensitivityProblem{DiffEqBase.isinplace(f),
            typeof(alg)}(alg);
        kwargs...)
end

function seed_duals(x::AbstractArray{V}, f,
    ::ForwardDiff.Chunk{N} = ForwardDiff.Chunk(x, typemax(Int64))) where {V,
    N}
    seeds = ForwardDiff.construct_seeds(ForwardDiff.Partials{N, V})
    duals = ForwardDiff.Dual{typeof(ForwardDiff.Tag(f, eltype(vec(x))))}.(vec(x), seeds)
end

function seed_duals(x::Number, f,
    ::ForwardDiff.Chunk{N} = ForwardDiff.Chunk(x, typemax(Int64))) where {N}
    seeds = ForwardDiff.construct_seeds(ForwardDiff.Partials{N, typeof(x)})
    duals = ForwardDiff.Dual{typeof(ForwardDiff.Tag(f, typeof(x)))}(x, seeds[1])
end

has_continuous_callback(cb::DiscreteCallback) = false
has_continuous_callback(cb::ContinuousCallback) = true
has_continuous_callback(cb::CallbackSet) = !isempty(cb.continuous_callbacks)

function ODEForwardSensitivityProblem(f::DiffEqBase.AbstractODEFunction, u0,
    tspan, p, alg::ForwardDiffSensitivity;
    du0 = zeros(eltype(u0), length(u0), length(p)), # perturbations of initial condition
    dp = I(length(p)), # perturbations of parameters
    kwargs...)
    num_sen_par = size(du0, 2)
    if num_sen_par != size(dp, 2)
        error("Same number of perturbations of initial conditions and parameters required")
    end
    if size(du0, 1) != length(u0)
        error("Perturbations for all initial conditions required")
    end
    if size(dp, 1) != length(p)
        error("Perturbations for all parameters required")
    end

    pdual = ForwardDiff.Dual{
        typeof(ForwardDiff.Tag(f, eltype(vec(p)))),
    }.(p,
        [ntuple(j -> dp[i, j], num_sen_par) for i in eachindex(p)])
    u0dual = ForwardDiff.Dual{
        typeof(ForwardDiff.Tag(f, eltype(vec(u0)))),
    }.(u0,
        [ntuple(j -> du0[i, j], num_sen_par)
         for i in eachindex(u0)])

    if (convert_tspan(alg) === nothing &&
        haskey(kwargs, :callback) && has_continuous_callback(kwargs.callback)) ||
       (convert_tspan(alg) !== nothing && convert_tspan(alg))
        tspandual = convert.(eltype(pdual), tspan)
    else
        tspandual = tspan
    end

    prob_dual = ODEProblem(f, u0dual, tspan, pdual,
        ODEForwardSensitivityProblem{DiffEqBase.isinplace(f),
            typeof(alg)}(alg);
        kwargs...)
end

"""
extract_local_sensitivities

Extracts the time series for the local sensitivities from the ODE solution. This requires
that the ODE was defined via `ODEForwardSensitivityProblem`.

```julia
extract_local_sensitivities(sol, asmatrix::Val = Val(false)) # Decompose the entire time series
extract_local_sensitivities(sol, i::Integer, asmatrix::Val = Val(false)) # Decompose sol[i]
extract_local_sensitivities(sol, t::Union{Number, AbstractVector},
                            asmatrix::Val = Val(false)) # Decompose sol(t)
```
"""
function extract_local_sensitivities(sol, asmatrix::Val = Val(false))
    extract_local_sensitivities(sol, sol.prob.problem_type.sensealg, asmatrix)
end
function extract_local_sensitivities(sol, asmatrix::Bool)
    extract_local_sensitivities(sol, Val{asmatrix}())
end
function extract_local_sensitivities(sol, i::Integer, asmatrix::Val = Val(false))
    _extract(sol, sol.prob.problem_type.sensealg, sol[i], asmatrix)
end
function extract_local_sensitivities(sol, i::Integer, asmatrix::Bool)
    extract_local_sensitivities(sol, i, Val{asmatrix}())
end
function extract_local_sensitivities(sol, t::Union{Number, AbstractVector},
    asmatrix::Val = Val(false))
    _extract(sol, sol.prob.problem_type.sensealg, sol(t), asmatrix)
end
function extract_local_sensitivities(sol, t, asmatrix::Bool)
    extract_local_sensitivities(sol, t, Val{asmatrix}())
end
function extract_local_sensitivities(tmp, sol, t::Union{Number, AbstractVector},
    asmatrix::Val = Val(false))
    _extract(sol, sol.prob.problem_type.sensealg, sol(tmp, t), asmatrix)
end
function extract_local_sensitivities(tmp, sol, t, asmatrix::Bool)
    extract_local_sensitivities(tmp, sol, t, Val{asmatrix}())
end

# Get ODE u vector and sensitivity values from all time points
function extract_local_sensitivities(sol, ::ForwardSensitivity, ::Val{false})
    ni = sol.prob.f.numindvar
    u = sol[1:ni, :]
    du = [sol[(ni * j + 1):(ni * (j + 1)), :] for j in 1:(sol.prob.f.numparams)]
    return u, du
end

function extract_local_sensitivities(sol, ::ForwardDiffSensitivity, ::Val{false})
    u = ForwardDiff.value.(sol)
    du_full = ForwardDiff.partials.(sol)
    firststate = first(du_full)
    firstparam = first(firststate)
    Js = map(1:length(firstparam)) do j
        map(CartesianIndices(du_full)) do II
            du_full[II][j]
        end
    end
    return u, Js
end

function extract_local_sensitivities(sol, ::ForwardSensitivity, ::Val{true})
    prob = sol.prob
    ni = prob.f.numindvar
    pn = prob.f.numparams
    jsize = (ni, pn)
    sol[1:ni, :], map(sol.u) do u
        collect(reshape((@view u[(ni + 1):end]), jsize))
    end
end

function extract_local_sensitivities(sol, ::ForwardDiffSensitivity, ::Val{true})
    retu = ForwardDiff.value.(sol)
    jsize = length(sol.u[1]), ForwardDiff.npartials(sol.u[1][1])
    du = map(sol.u) do u
        du_i = similar(retu, jsize)
        for i in eachindex(u)
            du_i[i, :] = ForwardDiff.partials(u[i])
        end
        du_i
    end
    retu, du
end

# Get ODE u vector and sensitivity values from sensitivity problem u vector
function _extract(sol, sensealg::ForwardSensitivity, su::AbstractVector,
    asmatrix::Val = Val(false))
    u = view(su, 1:(sol.prob.f.numindvar))
    du = _extract_du(sol, sensealg, su, asmatrix)
    return u, du
end

function _extract(sol, sensealg::ForwardDiffSensitivity, su::AbstractVector,
    asmatrix::Val = Val(false))
    u = ForwardDiff.value.(su)
    du = _extract_du(sol, sensealg, su, asmatrix)
    return u, du
end

# Get sensitivity values from sensitivity problem u vector (nested form)
function _extract_du(sol, ::ForwardSensitivity, su::Vector, ::Val{false})
    ni = sol.prob.f.numindvar
    return [view(su, (ni * j + 1):(ni * (j + 1))) for j in 1:(sol.prob.f.numparams)]
end

function _extract_du(sol, ::ForwardDiffSensitivity, su::Vector, ::Val{false})
    du_full = ForwardDiff.partials.(su)
    return [[du_full[i][j] for i in 1:size(du_full, 1)] for j in 1:length(du_full[1])]
end

# Get sensitivity values from sensitivity problem u vector (matrix form)
function _extract_du(sol, ::ForwardSensitivity, su::Vector, ::Val{true})
    ni = sol.prob.f.numindvar
    np = sol.prob.f.numparams
    return view(reshape(su, ni, np + 1), :, 2:(np + 1))
end

function _extract_du(sol, ::ForwardDiffSensitivity, su::Vector, ::Val{true})
    du_full = ForwardDiff.partials.(su)
    return [du_full[i][j] for i in 1:size(du_full, 1), j in 1:length(du_full[1])]
end

### Bonus Pieces

function SciMLBase.remake(prob::ODEProblem{uType, tType, isinplace, P, F, K,
        <:ODEForwardSensitivityProblem};
    f = nothing, tspan = nothing, u0 = nothing, p = nothing,
    kwargs...) where
    {uType, tType, isinplace, P, F, K}
    _p = p === nothing ? prob.p : p
    _f = f === nothing ? prob.f.f : f
    _u0 = u0 === nothing ? prob.u0[1:(prob.f.numindvar)] : u0[1:(prob.f.numindvar)]
    _tspan = tspan === nothing ? prob.tspan : tspan
    ODEForwardSensitivityProblem(_f, _u0,
        _tspan, _p, prob.problem_type.sensealg;
        prob.kwargs..., kwargs...)
end
SciMLBase.ODEFunction(f::ODEForwardSensitivityFunction; kwargs...) = f
