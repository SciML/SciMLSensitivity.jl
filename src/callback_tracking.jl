"""
Appends a tracking process to determine the time of the callback to be used in
the reverse pass. The rationale is explain in:

https://github.com/SciML/DiffEqSensitivity.jl/issues/4
"""
track_callbacks(cb,t,u,p,sensealg) = track_callbacks(CallbackSet(cb),t,u,p,sensealg)
track_callbacks(cb::CallbackSet,t,u,p,sensealg) = CallbackSet(
   map(cb->_track_callback(cb,t,u,p,sensealg), cb.continuous_callbacks),
   map(cb->_track_callback(cb,t,u,p,sensealg), cb.discrete_callbacks))

mutable struct ImplicitCorrection{T1,T2,T3,T4,T5,T6,T7,T8,T9,T10}
  gt_val::T1
  gu_val::T2
  gt::T3
  gu::T4
  gt_conf::T5
  gu_conf::T6
  condition::T7
  Lu_left::T8
  Lu_right::T9
  dy::T10
end

ImplicitCorrection(cb::DiscreteCallback,t,u,p,sensealg) = nothing
function ImplicitCorrection(cb,t,u,p,sensealg)
  condition = cb.condition

  gt_val = similar(u,1)
  gu_val = similar(u)

  fakeinteg = FakeIntegrator(u,p,t)
  gt = ConditionTimeWrapper(condition,u,fakeinteg)
  gu = ConditionUWrapper(condition,t,fakeinteg)

  gt_conf = build_deriv_config(sensealg,gt,gt_val,t)
  gu_conf = build_grad_config(sensealg,gu,u,p)

  dy = similar(u)

  Lu_left = similar(u)
  Lu_right = similar(u)

  ImplicitCorrection(gt_val,gu_val,gt,gu,gt_conf,gu_conf,condition,Lu_left,Lu_right,dy)
end

struct TrackedAffect{T,T2,T3,T4,T5}
    event_times::Vector{T}
    uleft::Vector{T2}
    pleft::Vector{T3}
    affect!::T4
    correction::T5
end

TrackedAffect(t::Number,u,p,affect!::Nothing,correction) = nothing
TrackedAffect(t::Number,u,p,affect!,correction) = TrackedAffect(Vector{typeof(t)}(undef,0),Vector{typeof(p)}(undef,0),Vector{typeof(u)}(undef,0),affect!,correction)

function (f::TrackedAffect)(integrator)
    uleft = deepcopy(integrator.u)
    pleft = deepcopy(integrator.p)
    f.affect!(integrator)
    if integrator.u_modified
        if isempty(f.event_times)
          push!(f.event_times,integrator.t)
          push!(f.uleft,uleft)
          push!(f.pleft,pleft)
        else
          if !maximum(.≈(integrator.t, f.event_times, rtol=0.0, atol=1e-14))
            push!(f.event_times,integrator.t)
            push!(f.uleft,uleft)
            push!(f.pleft,pleft)
          end
        end
    end
end

function _track_callback(cb::DiscreteCallback,t,u,p,sensealg)
    correction = ImplicitCorrection(cb,t,u,p,sensealg)
    DiscreteCallback(cb.condition,
                     TrackedAffect(t,u,p,cb.affect!,correction),
                     cb.initialize,
                     cb.finalize,
                     cb.save_positions)
end

function _track_callback(cb::ContinuousCallback,t,u,p,sensealg)
    correction = ImplicitCorrection(cb,t,u,p,sensealg)
    ContinuousCallback(
        cb.condition,
        TrackedAffect(t,u,p,cb.affect!,correction),
        TrackedAffect(t,u,p,cb.affect_neg!,correction),
        cb.initialize,
        cb.finalize,
        cb.idxs,
        cb.rootfind,cb.interp_points,
        cb.save_positions,
        cb.dtrelax,cb.abstol,cb.reltol,cb.repeat_nudge)
end

function _track_callback(cb::VectorContinuousCallback,t,u,p,sensealg)
    correction = ImplicitCorrection(cb,t,u,p,sensealg)
    VectorContinuousCallback(
               cb.condition,
               TrackedAffect(t,u,p,cb.affect!,correction),
               TrackedAffect(t,u,p,cb.affect_neg!,correction),
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

        if cb isa Union{ContinuousCallback,VectorContinuousCallback}
          # correction of the loss function sensitivity for continuous callbacks
          # wrt dependence of event time t on parameters and initial state.
          # Must be handled here because otherwise it is unclear if continuous or
          # discrete callback was triggered.
          @unpack correction = cb.affect!
          @unpack Lu_right = correction
          implicit_correction!(Lu_right,λ,correction,sensealg,S,y,integrator)
        end

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
        if !(sensealg isa QuadratureAdjoint)
          grad .-= dgrad
        end

        if cb isa Union{ContinuousCallback,VectorContinuousCallback}
          # second correction to correct for other value
          @unpack Lu_left = correction
          implicit_correction!(Lu_left,λ,correction,sensealg,S,y,integrator)
          λ .-= Lu_left - Lu_right
        end
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

function implicit_correction!(Lu,λ,correction,sensealg,S,y,integrator)
  @unpack gt_val, gu_val, gt, gu, gt_conf, gu_conf, condition, dy = correction

  p, t = integrator.p, integrator.t

  fakeinteg = FakeIntegrator([x for x in y],p,t)

  # derivative and gradient of condition with respect to time and state, respectively
  gt.u = y
  gt.integrator = fakeinteg

  gu.t = t
  gu.integrator = fakeinteg

  derivative!(gt_val, gt, t, sensealg, gt_conf)
  gradient!(gu_val, gu, y, sensealg, gu_conf)

  if inplace_sensitivity(S)
    S.f(dy,y,p,t)
  else
    dy[:] .= S.f(y,p,t)
  end

  gt_val .+= dot(gu_val,dy)
  @. gt_val = inv(gt_val) # allocates?

  @. gu_val *= -gt_val

  # loss function gradient (not condition!)
  # ∂L∂t correction should be added
  # correct adjoint
  Lu .= dot(λ,dy)*gu_val
  return nothing
end

mutable struct ConditionTimeWrapper{F,uType,Integrator} <: Function
  f::F
  u::uType
  integrator::Integrator
end
(ff::ConditionTimeWrapper)(t) = [ff.f(ff.u,t,ff.integrator)]

mutable struct ConditionUWrapper{F,tType,Integrator} <: Function
  f::F
  t::tType
  integrator::Integrator
end
(ff::ConditionUWrapper)(u) = ff.f(u,ff.t,ff.integrator)
