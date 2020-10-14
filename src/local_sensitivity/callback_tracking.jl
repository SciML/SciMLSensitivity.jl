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
        push!(event_times,integrator.t)
    end
end

function _track_callbacks(cb::DiscreteCallback,t)
    DiscreteCallback(cb.condition,
                     TrackedAffect(t,cb.affect!),
                     cb.initialize,
                     cb.save_positions)
end

function _track_callbacks(cb::ContinuousCallback,t)
    ContinuousCallback(
        cb.condition,
        TrackedAffect(t,cb.affect!),
        TrackedAffect(t,cb.affect_neg!),
        cb.initialize,
        cb.idxs,
        cb.rootfind,cb.interp_points,
        cb.save_positions,
        cb.dtrelax,cb.abstol,cb.reltol)
end

function _track_callbacks(cb::VectorContinuousCallback,t)
    VectorContinuousCallback(
               cb.condition,
               TrackedAffect(t,cb.affect!),
               TrackedAffect(t,cb.affect_neg!),
               cb.len,cb.initialize,cb.idxs,
               cb.rootfind,cb.interp_points,
               collect(cb.save_positions),
               cb.dtrelax,cb.abstol,cb.reltol)
end

struct FakeIntegrator{uType,P,tType}
    u::uType
    p::P
    t::tType
end

struct CallbackSensitivityFunction{F,S,D,P} <: SensitivityFunction
    f::F
    sensealg::S
    diffcache::D
    prob::P
end
getprob(S::CallbackSensitivityFunction) = S.prob

"""
Sets up callbacks for the adjoint pass. This is a version that has an effect
at each event of the forward pass and defines the reverse pass values via the
vjps as described in https://arxiv.org/pdf/1905.10403.pdf Equation 13.

For more information, see https://github.com/SciML/DiffEqSensitivity.jl/issues/4
"""
setup_reverse_callbacks(cb,sensealg) = setup_reverse_callbacks(CallbackSet(cb),sensealg)
function setup_reverse_callbacks(cb::CallbackSet,sensealg)
    cb = CallbackSet(_setup_reverse_callbacks.(cb.continuous_callbacks,sensealg)
                     reverse(_setup_reverse_callbacks.(cb.discrete_callbacks,sensealg)))
    cb
end

function _setup_reverse_callbacks(cb::DiscreteCallback,sensealg)
    condition(u,t,integrator) = t ∈ cb.affect!.event_times

    function affect!(integrator)
        function w(u,p,t)
          integrator = FakeIntegrator(u,p,t)
          cb.affect!(integrator)
          u - integrator.u
        end

        S = integrator.f # get the sensitivity function

        # Create a fake sensitivity function to do the vjps
        fakeS = CallbackSensitivityFunction(w,sensealg,integrator.f.diffcache,integrator.sol.prob)

        idx = length(S.y)

        # Hardcoding to match BacksolveAdjoint
        λ     = @view integrator.u[1:idx]
        grad  = @view integrator.u[idx+1:end-idx]
        y     = @view integrator.u[end-idx+1:end]

        du = get_tmp_cache(integrator)
        dλ    = @view du[1:idx]
        dgrad = @view du[idx+1:end-idx]
        dy    = @view du[end-idx+1:end]

        vecjacobian!(dλ, y, λ, p, t, S::SensitivityFunction;
                              dgrad=dgrad, dy=dy)
        integrator.u .+= du
    end

    PresetTimeCallback(cb.affect!.event_times,
                       condition,affect!,
                       save_positions = (false,false))
end
