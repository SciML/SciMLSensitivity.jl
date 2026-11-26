using Test, SciMLSensitivity, Enzyme, Reactant

# Test that VJP choice types are treated as inactive by Enzyme
# This addresses issue #1225 where sensealg in ODEProblem constructor would fail
# because Enzyme tried to differentiate through VJP choice objects

@testset "Enzyme VJP Choice Inactive Types" begin

    # Test 1: Basic test that VJP objects can be stored in data structures during Enzyme differentiation
    @testset "VJP types in data structures" begin
        vjp = EnzymeVJP()

        function test_func(x)
            # Store the VJP in a data structure (this would fail without inactive rules)
            data = (value = x[1] + x[2], vjp = vjp)
            return data.value * 2.0
        end

        x = [1.0, 2.0]
        dx = Enzyme.make_zero(x)

        # This should not throw an error
        @test_nowarn Enzyme.autodiff(
            Enzyme.Reverse, test_func, Enzyme.Active, Enzyme.Duplicated(x, dx)
        )
        @test dx ≈ [2.0, 2.0]
    end

    # Test 2: Test different VJP choice types are inactive
    @testset "Different VJP types inactive" begin
        vjp_types = [EnzymeVJP(), ZygoteVJP(), ReverseDiffVJP(), TrackerVJP(), ReactantVJP()]

        for vjp in vjp_types
            function test_func(x)
                data = (value = x[1] * x[2], vjp = vjp)
                return data.value + 1.0
            end

            x = [2.0, 3.0]
            dx = Enzyme.make_zero(x)

            @test_nowarn Enzyme.autodiff(
                Enzyme.Reverse, test_func, Enzyme.Active, Enzyme.Duplicated(x, dx)
            )
        end
    end

    # Test 3: Sensitivity algorithms with VJP choices can be stored in tuples
    @testset "Sensitivity algorithm with VJP in data structure" begin
        sensealg = BacksolveAdjoint(autojacvec = EnzymeVJP())

        function test_func(x)
            data = (value = x[1] * x[2], alg = sensealg)
            return data.value
        end

        x = [3.0, 4.0]
        dx = Enzyme.make_zero(x)

        @test_nowarn Enzyme.autodiff(
            Enzyme.Reverse, test_func, Enzyme.Active, Enzyme.Duplicated(x, dx)
        )
        @test dx ≈ [4.0, 3.0]
    end

    # Test 4: `automatic_sensealg_choice` is inactive to Enzyme.
    # The sensealg selection probes candidate VJP backends inside try/catch and
    # returns a discrete algorithm choice, so it must not be differentiated. This
    # is required for forward-over-reverse (nested Enzyme), where the outer
    # forward pass would otherwise type-analyze the Union-typed probing code and
    # fail with an IllegalTypeAnalysisException (SciMLSensitivity issue #1427).
    @testset "automatic_sensealg_choice inactive" begin
        @test Enzyme.EnzymeRules.inactive(SciMLSensitivity.automatic_sensealg_choice) ===
            nothing
        @test Enzyme.EnzymeRules.inactive(SciMLSensitivity._enzyme_originator_vjp) ===
            nothing
        @test Base.isconst(SciMLSensitivity, :AbstractGAdjoint)
    end

    # Test 5: Forward-over-reverse (HVP and Hessian) with AutoEnzyme on out-of-place
    # ComponentArray ODEs (issue #1644: second_order_adjoints.md).
    @static if VERSION >= v"1.12-"
        @testset "Forward-over-Reverse HVP and Hessian with ComponentArray (issue #1644)" begin
            using ComponentArrays, FiniteDiff, LinearAlgebra, Optimization, OrdinaryDiffEq

            u0_oop = [1.0, 0.5]
            tspan_oop = (0.0, 0.5)
            tsteps_oop = range(tspan_oop[1], tspan_oop[2], length = 6)
            p_oop = ComponentArray(a = [0.2, -0.3], b = [0.4, -0.1])
            ax_oop = getaxes(p_oop)
            v_oop = ComponentArray(a = [0.5, -0.25], b = [0.1, 0.75])

            function f_oop_1644(u, p, t)
                return [
                    p.a[1] * u[1] + p.b[1] * tanh(u[2]),
                    p.a[2] * u[2] + p.b[2] * u[1]^2,
                ]
            end

            prob_probe = ODEProblem(f_oop_1644, u0_oop, tspan_oop, p_oop)
            tun_probe, repack_probe, _ = SciMLSensitivity.SciMLStructures.canonicalize(
                SciMLSensitivity.SciMLStructures.Tunable(), p_oop
            )
            @test SciMLSensitivity._enzyme_originator_vjp(
                prob_probe, u0_oop, tun_probe, repack_probe, p_oop
            ) isa EnzymeVJP
            @test SciMLSensitivity._repack_enzyme_cotangent(p_oop, [1.0, 2.0, 3.0, 4.0]) ==
                [1.0, 2.0, 3.0, 4.0]

            adtype = Optimization.AutoEnzyme(;
                mode = Enzyme.set_runtime_activity(Enzyme.Reverse)
            )

            loss_default = function (x, _)
                prob = ODEProblem(f_oop_1644, u0_oop, tspan_oop, x)
                sol = solve(
                    prob, Tsit5();
                    saveat = tsteps_oop, abstol = 1.0e-9, reltol = 1.0e-9
                )
                return sum(abs2, Array(sol))
            end

            optf = Optimization.instantiate_function(
                Optimization.OptimizationFunction(loss_default, adtype),
                p_oop, adtype, Optimization.SciMLBase.NullParameters();
                g = true, h = true, hv = true
            )

            g = zero(p_oop)
            optf.grad(g, p_oop)
            g_fd = FiniteDiff.finite_difference_gradient(
                x -> loss_default(ComponentVector(x, ax_oop), nothing), collect(p_oop)
            )
            @test norm(collect(g) - g_fd) / norm(g_fd) < 1.0e-5

            hv = zero(p_oop)
            optf.hv(hv, p_oop, v_oop)
            H_fd = FiniteDiff.finite_difference_hessian(
                x -> loss_default(ComponentVector(x, ax_oop), nothing), collect(p_oop)
            )
            hv_fd = H_fd * collect(v_oop)
            @test norm(collect(hv) - hv_fd) / norm(hv_fd) < 1.0e-4

            H = zeros(length(p_oop), length(p_oop))
            optf.hess(H, p_oop)
            @test norm(H - H_fd) / norm(H_fd) < 1.0e-4

            # Verify GaussAdjoint(EnzymeVJP()) with non-const module-level globals
            # (as in second_order_adjoints.md).
            mod_glob = @eval module $(gensym(:Issue1644NonConstGlobals))
            using ComponentArrays, Enzyme, Optimization, OrdinaryDiffEq, SciMLSensitivity
            u0 = [1.0, 0.5]
            tspan = (0.0, 0.5)
            tsteps = range(tspan[1], tspan[2], length = 6)
            function f_ode(u, p, t)
                return [
                    p.a[1] * u[1] + p.b[1] * tanh(u[2]),
                    p.a[2] * u[2] + p.b[2] * u[1]^2,
                ]
            end
            function loss(x, _)
                prob = ODEProblem(f_ode, u0, tspan, x)
                sol = solve(
                    prob, Tsit5();
                    saveat = tsteps, abstol = 1.0e-9, reltol = 1.0e-9,
                    sensealg = GaussAdjoint(autojacvec = EnzymeVJP()),
                )
                return sum(abs2, Array(sol))
            end
            end

            optf_glob = Optimization.instantiate_function(
                Optimization.OptimizationFunction(mod_glob.loss, adtype),
                p_oop, adtype, Optimization.SciMLBase.NullParameters();
                g = true, hv = true
            )
            g_glob = zero(p_oop)
            optf_glob.grad(g_glob, p_oop)
            g_fd_glob = FiniteDiff.finite_difference_gradient(
                x -> mod_glob.loss(ComponentVector(x, ax_oop), nothing), collect(p_oop)
            )
            @test norm(collect(g_glob) - g_fd_glob) / norm(g_fd_glob) < 1.0e-4

            H_fd_glob = FiniteDiff.finite_difference_hessian(
                x -> mod_glob.loss(ComponentVector(x, ax_oop), nothing), collect(p_oop)
            )
            hv_glob = zero(p_oop)
            optf_glob.hv(hv_glob, p_oop, v_oop)
            @test norm(collect(hv_glob) - H_fd_glob * collect(v_oop)) /
                norm(H_fd_glob * collect(v_oop)) < 1.0e-4
        end
    end
end
