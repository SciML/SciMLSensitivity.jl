# Precompilation setup for SciMLSensitivity
# This file precompiles commonly-used code paths to reduce TTFX

using PrecompileTools: @compile_workload, @setup_workload

@setup_workload begin
    # Simple test function for precompilation (not exported)
    function _precompile_lorenz!(du, u, p, t)
        du[1] = p[1] * (u[2] - u[1])
        du[2] = u[1] * (p[2] - u[3]) - u[2]
        du[3] = u[1] * u[2] - p[3] * u[3]
        nothing
    end

    function _precompile_lorenz(u, p, t)
        dx = p[1] * (u[2] - u[1])
        dy = u[1] * (p[2] - u[3]) - u[2]
        dz = u[1] * u[2] - p[3] * u[3]
        [dx, dy, dz]
    end

    @compile_workload begin
        # Precompile sensitivity algorithm constructors
        # These are commonly used and their type parameters need compilation
        ForwardSensitivity()
        ForwardSensitivity(; autodiff = true)
        ForwardSensitivity(; autodiff = false)

        ForwardDiffSensitivity()
        ForwardDiffSensitivity(; chunk_size = 0)

        InterpolatingAdjoint()
        InterpolatingAdjoint(; autojacvec = nothing)
        InterpolatingAdjoint(; checkpointing = true)
        InterpolatingAdjoint(; checkpointing = false)

        BacksolveAdjoint()
        BacksolveAdjoint(; checkpointing = true)
        BacksolveAdjoint(; checkpointing = false)

        QuadratureAdjoint()
        QuadratureAdjoint(; abstol = 1e-6, reltol = 1e-3)

        GaussAdjoint()
        GaussAdjoint(; checkpointing = true)

        # VJP choices
        ZygoteVJP()
        ZygoteVJP(; allow_nothing = true)
        ReverseDiffVJP()
        ReverseDiffVJP(true)
        TrackerVJP()
        TrackerVJP(; allow_nothing = true)
        EnzymeVJP()
        EnzymeVJP(; chunksize = 0)

        # Second order algorithms
        ForwardDiffOverAdjoint(InterpolatingAdjoint())
        ForwardDiffOverAdjoint(BacksolveAdjoint())

        # SteadyState
        SteadyStateAdjoint()
        SteadyStateAdjoint(; autojacvec = nothing)

        # Shadowing algorithms
        ForwardLSS()
        AdjointLSS()
        TimeDilation(10.0)
        TimeDilation(10.0, 0.0, 0.0)

        # Utility functions - precompile common type combinations
        alg_autodiff(InterpolatingAdjoint())
        alg_autodiff(BacksolveAdjoint())
        alg_autodiff(QuadratureAdjoint())
        alg_autodiff(GaussAdjoint())

        get_chunksize(InterpolatingAdjoint())
        get_chunksize(BacksolveAdjoint())
        get_chunksize(QuadratureAdjoint())

        ischeckpointing(InterpolatingAdjoint())
        ischeckpointing(BacksolveAdjoint())
        ischeckpointing(GaussAdjoint())

        isnoisemixing(InterpolatingAdjoint())
        isnoisemixing(BacksolveAdjoint())

        convert_tspan(ForwardDiffSensitivity())
        convert_tspan(InterpolatingAdjoint())

        compile_tape(ReverseDiffVJP())
        compile_tape(ReverseDiffVJP(true))
        compile_tape(true)
        compile_tape(false)

        # Precompile get_autodiff_from_vjp for common VJP types
        get_autodiff_from_vjp(ReverseDiffVJP())
        get_autodiff_from_vjp(ReverseDiffVJP(true))
        get_autodiff_from_vjp(ZygoteVJP())
        get_autodiff_from_vjp(TrackerVJP())
        get_autodiff_from_vjp(nothing)
        get_autodiff_from_vjp(true)
        get_autodiff_from_vjp(false)
    end
end
