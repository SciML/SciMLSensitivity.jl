"""
Appends a tracking process to determine the time of the callback to be used in
the reverse pass. The rationale is explain in:

https://github.com/SciML/DiffEqSensitivity.jl/issues/4
"""
track_callbacks(cb,t,u;save_values = true, always_save_t = false) = track_callbacks(CallbackSet(cb),t,u,p;save_values,always_save_t)
track_callbacks(cb::CallbackSet,t,u,p; save_values = true, always_save_t = false) = CallbackSet(
   map(cb->_track_callback(cb,t,u,p,save_values,always_save_t), cb.continuous_callbacks),
   map(cb->_track_callback(cb,t,u,p,save_values,always_save_t), cb.discrete_callbacks))

struct TrackedAffect{T,T2,T3,T4}
    event_times::Vector{T}
    uleft::Vector{T2}
    pleft::Vector{T3}
    affect!::T4
    save_values::Bool
    always_save_t::Bool
end

TrackedAffect(t::Number,u,p,affect!::Nothing,save_values,always_save_t) = nothing
TrackedAffect(t::Number,u,p,affect!,save_values,always_save_t) = TrackedAffect(Vector{typeof(t)}(undef,0),Vector{typeof(p)}(undef,0),Vector{typeof(u)}(undef,0),affect!,save_values,always_save_t)

function (f::TrackedAffect)(integrator)
    f.save_values && (uleft = deepcopy(integrator.u))
    f.save_values && (pleft = deepcopy(integrator.p))
    f.affect!(integrator)
    (f.always_save_t || integrator.u_modified) && push!(f.event_times,integrator.t)
    if integrator.u_modified
        f.save_values && push!(f.uleft,uleft)
        f.save_values && push!(f.pleft,pleft)
    end
end

function _track_callback(cb::DiscreteCallback,t,u,p,save_values,always_save_t)
    DiscreteCallback(cb.condition,
                     TrackedAffect(t,u,p,cb.affect!,save_values,always_save_t),
                     cb.initialize,
                     cb.finalize,
                     cb.save_positions)
end

function _track_callback(cb::ContinuousCallback,t,u,p,save_values,always_save_t)
    ContinuousCallback(
        cb.condition,
        TrackedAffect(t,u,p,cb.affect!,save_values,always_save_t),
        TrackedAffect(t,u,p,cb.affect_neg!,save_values,always_save_t),
        cb.initialize,
        cb.finalize,
        cb.idxs,
        cb.rootfind,cb.interp_points,
        cb.save_positions,
        cb.dtrelax,cb.abstol,cb.reltol,cb.repeat_nudge)
end

function _track_callback(cb::VectorContinuousCallback,t,u,p,save_values,always_save_t)
    VectorContinuousCallback(
               cb.condition,
               TrackedAffect(t,u,p,cb.affect!,save_values,always_save_t),
               TrackedAffect(t,u,p,cb.affect_neg!,save_values,always_save_t),
               cb.len,cb.initialize,cb.finalize,cb.idxs,
               cb.rootfind,cb.interp_points,
               collect(cb.save_positions),
               cb.dtrelax,cb.abstol,cb.reltol,cb.repeat_nudge)
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

    function affect!(integrator)

        function w(du,u,p,t)
          fakeinteg = FakeIntegrator([x for x in u],[x for x in p],t)
          cb.affect!.affect!(fakeinteg)
          du .= fakeinteg.u
        end

        S = integrator.f.f # get the sensitivity function

        # Create a fake sensitivity function to do the vjps
        fakeS = CallbackSensitivityFunction(w,sensealg,S.diffcache,integrator.sol.prob)

        du = first(get_tmp_cache(integrator))
        λ,grad,y,dλ,dgrad,dy = split_states(du,integrator.u,integrator.t,S)

        update_p = copy_to_integrator!(cb,y,integrator.p,integrator.t)

        if update_p
          # changes in parameters
          if !(sensealg isa QuadratureAdjoint)
            function wp(dp,p,u,t)
              fakeinteg = FakeIntegrator([x for x in u],[x for x in p],t)
              cb.affect!.affect!(fakeinteg)
              dp .= fakeinteg.p
            end

            fakeSp = CallbackSensitivityFunction(wp,sensealg,S.diffcache,integrator.sol.prob)
            #vjp with Jacobin given by dw/dp before event and vector given by grad
            vecjacobian!(dgrad, integrator.p, grad, y, integrator.t, fakeSp;
                                  dgrad=nothing, dy=nothing)
            grad .= dgrad
          end
        end

        vecjacobian!(dλ, y, λ, integrator.p, integrator.t, fakeS;
                              dgrad=dgrad, dy=dy)

        λ .= dλ

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


function copy_to_integrator!(cb::DiscreteCallback, y, p, t)
  indx = searchsortedfirst(cb.affect!.event_times,t)
  copyto!(y, cb.affect!.uleft[indx])
  update_p = (p != cb.affect!.pleft[indx])
  update_p && copyto!(p, cb.affect!.pleft[indx])
  update_p
end

function copy_to_integrator!(cb::Union{ContinuousCallback,VectorContinuousCallback}, y, p, t)
  if !isempty(cb.affect!.event_times)
    indx = searchsortedfirst(cb.affect!.event_times,t)
    if cb.affect!.event_times[indx]!=t
      indx = searchsortedfirst(cb.affect_neg!.event_times,t)
      copyto!(y, cb.affect_neg!.uleft[indx])
      update_p = (p != cb.affect!.pleft[indx])
      update_p && copyto!(p, cb.affect_neg!.pleft[indx])
    else
      copyto!(y, cb.affect!.uleft[indx])
      update_p = (p != cb.affect!.pleft[indx])
      update_p && copyto!(p, cb.affect!.pleft[indx])
    end
  else
    indx = searchsortedfirst(cb.affect_neg!.event_times,t)
    copyto!(y, cb.affect_neg!.uleft[indx])
    update_p = (p != cb.affect_neg!.pleft[indx])
    update_p && copyto!(p, cb.affect_neg!.pleft[indx])
  end
  update_p
end
