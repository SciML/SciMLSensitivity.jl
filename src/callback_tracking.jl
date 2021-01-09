"""
Appends a tracking process to determine the time of the callback to be used in
the reverse pass. The rationale is explain in:

https://github.com/SciML/DiffEqSensitivity.jl/issues/4
"""
track_callbacks(cb,t) = track_callbacks(CallbackSet(cb),t)
track_callbacks(cb::CallbackSet,t) = CallbackSet(_track_callback.(cb.continuous_callbacks,t),
                                                 _track_callback.(cb.discrete_callbacks,t))

struct TrackedAffect{T,T2}
    event_times::Vector{T}
    affect!::T2
end

TrackedAffect(t::Number,affect!::Nothing) = nothing
TrackedAffect(t::Number,affect!) = TrackedAffect(Vector{typeof(t)}(undef,0),affect!)

function (f::TrackedAffect)(integrator)
    f.affect!(integrator)
    if integrator.u_modified
        push!(f.event_times,integrator.t)
    end
end

function _track_callback(cb::DiscreteCallback,t)
    DiscreteCallback(cb.condition,
                     TrackedAffect(t,cb.affect!),
                     cb.initialize,
                     cb.finalize,
                     cb.save_positions)
end

function _track_callback(cb::ContinuousCallback,t)
    ContinuousCallback(
        cb.condition,
        TrackedAffect(t,cb.affect!),
        TrackedAffect(t,cb.affect_neg!),
        cb.initialize,
        cb.finalize,
        cb.idxs,
        cb.rootfind,cb.interp_points,
        cb.save_positions,
        cb.dtrelax,cb.abstol,cb.reltol)
end

function _track_callback(cb::VectorContinuousCallback,t)
    VectorContinuousCallback(
               cb.condition,
               TrackedAffect(t,cb.affect!),
               TrackedAffect(t,cb.affect_neg!),
               cb.len,cb.initialize,cb.finalize,cb.idxs,
               cb.rootfind,cb.interp_points,
               collect(cb.save_positions),
               cb.dtrelax,cb.abstol,cb.reltol)
end

struct FakeIntegrator{uType,P,tType}
    u::uType
    p::P
    t::tType
end

struct CallbackSensitivityFunction{fType,Alg<:DiffEqBase.AbstractSensitivityAlgorithm,C<:AdjointDiffCache,pType} <: SensitivityFunction
    f::fType
    sensealg::Alg
    diffcache::C
    prob::pType
end

getprob(S::CallbackSensitivityFunction) = S.prob
inplace_sensitivity(S::CallbackSensitivityFunction) = true

"""
Sets up callbacks for the adjoint pass. This is a version that has an effect
at each event of the forward pass and defines the reverse pass values via the
vjps as described in https://arxiv.org/pdf/1905.10403.pdf Equation 13.

For more information, see https://github.com/SciML/DiffEqSensitivity.jl/issues/4
"""
setup_reverse_callbacks(cb,sensealg) = setup_reverse_callbacks(CallbackSet(cb),sensealg)
function setup_reverse_callbacks(cb::CallbackSet,sensealg)
    cb = CallbackSet(_setup_reverse_callbacks.(cb.continuous_callbacks,(sensealg,))...,
                     reverse(_setup_reverse_callbacks.(cb.discrete_callbacks,(sensealg,)))...)
    cb
end

function _setup_reverse_callbacks(cb::Union{ContinuousCallback,DiscreteCallback,VectorContinuousCallback},sensealg)

    if typeof(cb) <: Union{ContinuousCallback,VectorContinuousCallback}
        error("Continuous callbacks are currently not supported with continuous adjoint methods. Please use a discrete adjoint method like ReverseDiffAdjoint.")
    end

    function affect!(integrator)

        local _p

        function w(du,u,p,t)
          fakeinteg = FakeIntegrator([x for x in u],p,t)
          cb.affect!.affect!(fakeinteg)
          _p = fakeinteg.p
          du .= u - fakeinteg.u
        end

        S = integrator.f.f # get the sensitivity function

        # Create a fake sensitivity function to do the vjps
        fakeS = CallbackSensitivityFunction(w,sensealg,S.diffcache,integrator.sol.prob)

        du = first(get_tmp_cache(integrator))
        λ,grad,y,dλ,dgrad,dy = split_states(du,integrator.u,integrator.t,S)

        vecjacobian!(dλ, y, λ, integrator.p, integrator.t, fakeS;
                              dgrad=dgrad, dy=dy)

        integrator.u .+= du
        _p != integrator.p && (integrator.p = _p)
    end

    times = if typeof(cb) <: DiscreteCallback
        cb.affect!.event_times
    else
        [cb.affect!.event_times;cb.affect_neg!.event_times]
    end

    PresetTimeCallback(times,
                       affect!,
                       save_positions = (false,false))
end
