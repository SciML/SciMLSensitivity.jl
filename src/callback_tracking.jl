"""
Appends a tracking process to determine the time of the callback to be used in
the reverse pass. The rationale is explain in:

https://github.com/SciML/DiffEqSensitivity.jl/issues/4
"""
track_callbacks(cb,t,u,p,sensealg) = track_callbacks(CallbackSet(cb),t,u,p,sensealg)
track_callbacks(cb::CallbackSet,t,u,p,sensealg) = CallbackSet(
   map(cb->_track_callback(cb,t,u,p,sensealg), cb.continuous_callbacks),
   map(cb->_track_callback(cb,t,u,p,sensealg), cb.discrete_callbacks))

mutable struct ImplicitCorrection{T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,RefType}
  gt_val::T1
  gu_val::T2
  gt::T3
  gu::T4
  gt_conf::T5
  gu_conf::T6
  condition::T7
  Lu_left::T8
  Lu_right::T9
  dy_left::T10
  dy_right::T11
  cur_time::RefType # initialized as "dummy" Ref that gets overwritten by Ref of loss
  terminated::Bool
end

ImplicitCorrection(cb::DiscreteCallback,t,u,p,sensealg) = nothing
function ImplicitCorrection(cb,t,u,p,sensealg)
  condition = cb.condition

  gt_val = similar(u,1)
  gu_val = similar(u)

  fakeinteg = FakeIntegrator(u,p,t,t)
  gt, gu = build_condition_wrappers(cb,condition,u,t,fakeinteg)

  gt_conf = build_deriv_config(sensealg,gt,gt_val,t)
  gu_conf = build_grad_config(sensealg,gu,u,p)


  dy_left = similar(u)
  dy_right = similar(u)

  Lu_left = similar(u)
  Lu_right = similar(u)

  cur_time = Ref(1) # initialize the Ref, set to Ref of loss below
  terminated = false

  ImplicitCorrection(gt_val,gu_val,gt,gu,gt_conf,gu_conf,condition,Lu_left,Lu_right,dy_left,dy_right,cur_time,terminated)
end

struct TrackedAffect{T,T2,T3,T4,T5,T6}
    event_times::Vector{T}
    tprev::Vector{T}
    uleft::Vector{T2}
    pleft::Vector{T3}
    affect!::T4
    correction::T5
    event_idx::Vector{T6}
end

TrackedAffect(t::Number,u,p,affect!::Nothing,correction) = nothing
TrackedAffect(t::Number,u,p,affect!,correction) = TrackedAffect(Vector{typeof(t)}(undef,0),Vector{typeof(t)}(undef,0),
                                                                Vector{typeof(u)}(undef,0),Vector{typeof(p)}(undef,0),affect!,correction,
                                                                Vector{Int}(undef,0))

function (f::TrackedAffect)(integrator,event_idx=nothing)
    uleft = deepcopy(integrator.u)
    pleft = deepcopy(integrator.p)
    if event_idx===nothing
      f.affect!(integrator)
    else
      f.affect!(integrator,event_idx)
    end
    if integrator.u_modified
        if isempty(f.event_times)
          push!(f.event_times,integrator.t)
          push!(f.tprev,integrator.tprev)
          push!(f.uleft,uleft)
          push!(f.pleft,pleft)
          if event_idx !== nothing
            push!(f.event_idx,event_idx)
          end
        else
          if !maximum(.≈(integrator.t, f.event_times, rtol=0.0, atol=1e-14))
            push!(f.event_times,integrator.t)
            push!(f.tprev,integrator.tprev)
            push!(f.uleft,uleft)
            push!(f.pleft,pleft)
            if event_idx !== nothing
              push!(f.event_idx, event_idx)
            end
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

struct FakeIntegrator{uType,P,tType,tprevType}
    u::uType
    p::P
    t::tType
    tprev::tprevType
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
setup_reverse_callbacks(cb,sensealg,g,cur_time,terminated) = setup_reverse_callbacks(CallbackSet(cb),sensealg,g,cur_time,terminated)
function setup_reverse_callbacks(cb::CallbackSet,sensealg,g,cur_time,terminated)
    cb = CallbackSet(_setup_reverse_callbacks.(cb.continuous_callbacks,(sensealg,),(g,),(cur_time,),(terminated,))...,
                     reverse(_setup_reverse_callbacks.(cb.discrete_callbacks,(sensealg,),(g,),(cur_time,),(terminated,)))...)
    return cb
end

function _setup_reverse_callbacks(cb::Union{ContinuousCallback,DiscreteCallback,VectorContinuousCallback},sensealg,g,loss_ref,terminated)

    if cb isa Union{ContinuousCallback,VectorContinuousCallback} && cb.affect! !== nothing
      cb.affect!.correction.cur_time = loss_ref # set cur_time
      cb.affect!.correction.terminated = terminated # flag if time evolution was terminated by callback
    end

    # ReverseLossCallback adds gradients before and after the callback if save_positions is (true, true).
    # This, however, means that we must check the save_positions setting within the callback.
    # if save_positions = [1,1] is true the loss gradient is accumulated correctly before and after callback.
    # if save_positions = [0,0] no extra gradient is added.
    # if save_positions = [0,1] the gradient contribution is added before the callback but no additional gradient is added afterwards.
    # if save_positions = [1,0] the gradient contribution is added before, and in principle we would need to correct the adjoint state again. Thefore,

    cb.save_positions == [1,0] && error("save_positions=[1,0] is currently not supported.")

    function affect!(integrator)

        indx, pos_neg = get_indx(cb,integrator.t)
        tprev = get_tprev(cb,indx,pos_neg)
        event_idx = cb isa VectorContinuousCallback ? get_event_idx(cb,indx,pos_neg) : nothing

        w = let tprev=tprev, pos_neg=pos_neg, event_idx=event_idx
              function (du,u,p,t)
                _affect! = get_affect!(cb,pos_neg)
                fakeinteg = FakeIntegrator([x for x in u],[x for x in p],t,tprev)
                if cb isa VectorContinuousCallback
                  _affect!(fakeinteg,event_idx)
                else
                  _affect!(fakeinteg)
                end
                du .= fakeinteg.u
              end
        end
        S = integrator.f.f # get the sensitivity function

        # Create a fake sensitivity function to do the vjps
        fakeS = CallbackSensitivityFunction(w,sensealg,S.diffcache,integrator.sol.prob)

        du = first(get_tmp_cache(integrator))
        λ,grad,y,dλ,dgrad,dy = split_states(du,integrator.u,integrator.t,S)

        # if save_positions[2] = false, then the right limit is not saved. Thus, for
        # the QuadratureAdjoint we would need to lift y from the left to the right limit.
        # However, one also needs to update dgrad later on.
        if (sensealg isa QuadratureAdjoint && !cb.save_positions[2]) # || (sensealg isa InterpolatingAdjoint && ischeckpointing(sensealg))
          # lifting for InterpolatingAdjoint is not needed anymore. Callback is already applied.
          w(y,y,integrator.p,integrator.t)
        end

        if cb isa Union{ContinuousCallback,VectorContinuousCallback}
          # correction of the loss function sensitivity for continuous callbacks
          # wrt dependence of event time t on parameters and initial state.
          # Must be handled here because otherwise it is unclear if continuous or
          # discrete callback was triggered.
          @unpack correction = cb.affect!
          @unpack dy_right, Lu_right = correction
          # compute #f(xτ_right,p_right,τ(x₀,p))
          compute_f!(dy_right,S,y,integrator)
          # if callback did not terminate the time evolution, we have to compute one more correction term.
          if cb.save_positions[2] && !correction.terminated
            loss_indx = correction.cur_time[] + 1
            loss_correction!(Lu_right,y,integrator,g,loss_indx)
          else
            Lu_right .*= false
          end
        end

        update_p = copy_to_integrator!(cb,y,integrator.p,integrator.t,indx,pos_neg)
        # reshape u and du (y and dy) to match forward pass (e.g., for matrices as initial conditions). Only needed for BacksolveAdjoint
        if sensealg isa BacksolveAdjoint
          _size = pos_neg ? size(cb.affect!.uleft[indx]) : size(cb.affect_neg!.uleft[indx])
          y = reshape(y, _size)
          dy = reshape(dy, _size)
        end

        if cb isa Union{ContinuousCallback,VectorContinuousCallback}
          # compute the correction of the right limit (with left state limit inserted into dgdt)
          @unpack dy_left, cur_time = correction
          compute_f!(dy_left,S,y,integrator)
          dgdt(dy_left,correction,sensealg,y,integrator,tprev,event_idx)
          if !correction.terminated
            implicit_correction!(Lu_right,dλ,λ,dy_right,correction)
            correction.terminated = false # additional callbacks might have happened which didn't terminate the time evolution
          end
        end

        if update_p
          # changes in parameters
          if !(sensealg isa QuadratureAdjoint)

            wp = let tprev=tprev, pos_neg=pos_neg, event_idx=event_idx
              function (dp,p,u,t)
                _affect! = get_affect!(cb,pos_neg)
                fakeinteg = FakeIntegrator([x for x in u],[x for x in p],t,tprev)
                if cb isa VectorContinuousCallback
                  _affect!(fakeinteg, event_idx)
                else
                  _affect!(fakeinteg)
                end
                dp .= fakeinteg.p
              end
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

        dgrad!==nothing && (dgrad .*= -1)
        if cb isa Union{ContinuousCallback,VectorContinuousCallback}
          # second correction to correct for left limit
          @unpack Lu_left = correction
          implicit_correction!(Lu_left,dλ,dy_left,correction)
          dλ .+= Lu_left - Lu_right

          if cb.save_positions[1] == true
            # if the callback saved the first position, we need to implicitly correct this value as well
            loss_indx = correction.cur_time[]
            implicit_correction!(Lu_left,dy_left,correction,y,integrator,g,loss_indx)
            dλ .+= Lu_left
          end
        end

        λ .= dλ

        if !(sensealg isa QuadratureAdjoint)
          grad .-= dgrad
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

get_indx(cb::DiscreteCallback,t) = (searchsortedfirst(cb.affect!.event_times,t), true)
function get_indx(cb::Union{ContinuousCallback,VectorContinuousCallback}, t)
  if !isempty(cb.affect!.event_times) || !isempty(cb.affect_neg!.event_times)
    indx = searchsortedfirst(cb.affect!.event_times,t)
    indx_neg = searchsortedfirst(cb.affect_neg!.event_times,t)
    if !isempty(cb.affect!.event_times) && cb.affect!.event_times[min(indx,length(cb.affect!.event_times))]==t
      return indx, true
    elseif !isempty(cb.affect_neg!.event_times) && cb.affect_neg!.event_times[min(indx_neg,length(cb.affect_neg!.event_times))]==t
      return indx_neg, false
    else
      error("Event was triggered but no corresponding event in ContinuousCallback was found. Please report this error.")
    end
  else
    error("No event was recorded. Please report this error.")
  end
end

get_tprev(cb::DiscreteCallback,indx,bool) = cb.affect!.tprev[indx]
function get_tprev(cb::Union{ContinuousCallback,VectorContinuousCallback}, indx, bool)
  if bool
    return cb.affect!.tprev[indx]
  else
    return cb.affect_neg!.tprev[indx]
  end
end

function get_event_idx(cb::VectorContinuousCallback, indx, bool)
  if bool
    return cb.affect!.event_idx[indx]
  else
    return cb.affect_neg!.event_idx[indx]
  end
end

function copy_to_integrator!(cb::DiscreteCallback, y, p, t, indx, bool)
  copyto!(y, cb.affect!.uleft[indx])
  update_p = (p != cb.affect!.pleft[indx])
  update_p && copyto!(p, cb.affect!.pleft[indx])
  update_p
end

function copy_to_integrator!(cb::Union{ContinuousCallback,VectorContinuousCallback}, y, p, t, indx, bool)
  if bool
    copyto!(y, cb.affect!.uleft[indx])
    update_p = (p != cb.affect!.pleft[indx])
    update_p && copyto!(p, cb.affect!.pleft[indx])
  else
    copyto!(y, cb.affect_neg!.uleft[indx])
    update_p = (p != cb.affect_neg!.pleft[indx])
    update_p && copyto!(p, cb.affect_neg!.pleft[indx])
  end
  update_p
end

function compute_f!(dy,S,y,integrator)
  p, t = integrator.p, integrator.t

  if inplace_sensitivity(S)
    S.f(dy,y,p,t)
  else
    dy[:] .= S.f(y,p,t)
  end
  return nothing
end

function dgdt(dy,correction,sensealg,y,integrator,tprev,event_idx)
  # dy refers to f evaluated on left limit
  @unpack gt_val, gu_val, gt, gu, gt_conf, gu_conf, condition = correction

  p, t = integrator.p, integrator.t

  fakeinteg = FakeIntegrator([x for x in y],p,t,tprev)

  # derivative and gradient of condition with respect to time and state, respectively
  gt.u = y
  gt.integrator = fakeinteg

  gu.t = t
  gu.integrator = fakeinteg

  # for VectorContinuousCallback we also need to set the event_idx.
  if gt isa VectorConditionTimeWrapper
    gt.event_idx = event_idx
    gu.event_idx = event_idx

    # safety check: evaluate condition to check if several conditions were true.
    # This is currently not supported
    condition(gt.out_cache,y,t,integrator)
    gt.out_cache .= abs.(gt.out_cache) .< 1000*eps(eltype(gt.out_cache))
    (sum(gt.out_cache)!=1 || gt.out_cache[event_idx]!=1) && error("Either several events were triggered or `event_idx` was falsely identified. Output of conditions $(gt.out_cache)")
  end

  derivative!(gt_val, gt, t, sensealg, gt_conf)
  gradient!(gu_val, gu, y, sensealg, gu_conf)

  gt_val .+= dot(gu_val,dy)
  @. gt_val = inv(gt_val) # allocates?

  @. gu_val *= -gt_val
  return nothing
end

function loss_correction!(Lu,y,integrator,g,indx)
  # ∂L∂t correction should be added if L depends explicitly on time.
  p, t = integrator.p, integrator.t
  g(Lu,y,p,t,indx)
  return nothing
end

function implicit_correction!(Lu,dλ,λ,dy,correction)
  @unpack gu_val = correction

  # remove gradients from adjoint state to compute correction factor
  @. dλ = λ - Lu
  Lu .= dot(dλ,dy)*gu_val

  return nothing
end

function implicit_correction!(Lu,λ,dy,correction)
  @unpack gu_val = correction

  Lu .= dot(λ,dy)*gu_val

  return nothing
end

function implicit_correction!(Lu,dy,correction,y,integrator,g,indx)
  @unpack gu_val = correction

  p, t = integrator.p, integrator.t

  # loss function gradient (not condition!)
  # ∂L∂t correction should be added, also ∂L∂p is missing.
  # correct adjoint
  g(Lu,y,p,t,indx)

  Lu .= dot(Lu,dy)*gu_val

  # note that we don't add the gradient Lu here again to the correction because it will be added by the ReverseLossCallback.
  return nothing
end

# ConditionTimeWrapper: Wrapper for implicit correction for ContinuousCallback
# VectorConditionTimeWrapper: Wrapper for implicit correction for VectorContinuousCallback
function build_condition_wrappers(cb::ContinuousCallback,condition,u,t,fakeinteg)
  gt = ConditionTimeWrapper(condition,u,fakeinteg)
  gu = ConditionUWrapper(condition,t,fakeinteg)
  return gt, gu
end
function build_condition_wrappers(cb::VectorContinuousCallback,condition,u,t,fakeinteg)
  out = similar(u, cb.len) # create a cache for condition function (out,u,t,integrator)
  gt = VectorConditionTimeWrapper(condition,u,fakeinteg,1,out)
  gu = VectorConditionUWrapper(condition,t,fakeinteg,1,out)
  return gt, gu
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
mutable struct VectorConditionTimeWrapper{F,uType,Integrator,outType} <: Function
  f::F
  u::uType
  integrator::Integrator
  event_idx::Int
  out_cache::outType
end
(ff::VectorConditionTimeWrapper)(t) = (ff.f(ff.out_cache,ff.u,t,ff.integrator); [ff.out_cache[ff.event_idx]])

mutable struct VectorConditionUWrapper{F,tType,Integrator,outType} <: Function
  f::F
  t::tType
  integrator::Integrator
  event_idx::Int
  out_cache::outType
end
(ff::VectorConditionUWrapper)(u) = (out = similar(u,length(ff.out_cache)); ff.f(out,u,ff.t,ff.integrator); out[ff.event_idx])

DiffEqBase.terminate!(i::FakeIntegrator) = nothing

# get the affect function of the callback. For example, allows us to get the `f` in PeriodicCallback without the integrator.tstops handling.
get_affect!(cb::DiscreteCallback,bool) = get_affect!(cb.affect!)
get_affect!(cb::Union{ContinuousCallback,VectorContinuousCallback},bool) = bool ? get_affect!(cb.affect!) : get_affect!(cb.affect_neg!)
get_affect!(affect!::TrackedAffect) = get_affect!(affect!.affect!)
get_affect!(affect!) = affect!
get_affect!(affect!::DiffEqCallbacks.PeriodicCallbackAffect) = affect!.affect!
