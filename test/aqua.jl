using SciMLSensitivity, Aqua, SciMLBase, ExplicitImports, Zygote

@testset "Aqua" begin
    Aqua.find_persistent_tasks_deps(SciMLSensitivity)
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
    Aqua.test_stale_deps(
        SciMLSensitivity;
        ignore = [:PreallocationTools]  # Used but detection fails
    )
    Aqua.test_unbound_args(SciMLSensitivity)
    Aqua.test_undefined_exports(SciMLSensitivity)
end

@testset "ExplicitImports" begin
    @test ExplicitImports.check_no_implicit_imports(
        SciMLSensitivity; skip = (Base, Core, SciMLBase)
    ) === nothing
    @test ExplicitImports.check_no_stale_explicit_imports(SciMLSensitivity) === nothing
    @test ExplicitImports.check_all_qualified_accesses_via_owners(
        SciMLSensitivity;
        skip = (Zygote => Zygote.ZygoteRules,)
    ) === nothing
end
