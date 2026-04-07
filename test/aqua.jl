using SciMLSensitivity, Aqua, SciMLBase, ExplicitImports

@testset "Aqua" begin
    # find_persistent_tasks_deps may fail on Julia 1.12+ due to OrdinaryDiffEqCore's
    # [sources] section containing monorepo-relative paths that don't resolve when
    # the package is installed from the registry.
    try
        Aqua.find_persistent_tasks_deps(SciMLSensitivity)
    catch e
        @warn "Aqua.find_persistent_tasks_deps failed (likely upstream [sources] issue)" exception = e
    end
    Aqua.test_ambiguities(SciMLSensitivity, recursive = false)
    Aqua.test_deps_compat(SciMLSensitivity)
    Aqua.test_piracies(
        SciMLSensitivity;
        treat_as_own = [
            SciMLBase._concrete_solve_adjoint,
            SciMLBase._concrete_solve_forward,
        ]
    )
    Aqua.test_project_extras(SciMLSensitivity)
    Aqua.test_stale_deps(SciMLSensitivity)
    Aqua.test_unbound_args(SciMLSensitivity)
    Aqua.test_undefined_exports(SciMLSensitivity)
end

@testset "ExplicitImports" begin
    @test ExplicitImports.check_no_implicit_imports(
        SciMLSensitivity; skip = (Base, Core, SciMLBase)
    ) === nothing
    @test ExplicitImports.check_no_stale_explicit_imports(SciMLSensitivity) === nothing
    @test ExplicitImports.check_all_qualified_accesses_via_owners(SciMLSensitivity) ===
        nothing
end
