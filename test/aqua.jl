using SciMLSensitivity, Aqua, DiffEqBase

@testset "Aqua" begin
    Aqua.find_persistent_tasks_deps(SciMLSensitivity)
    Aqua.test_ambiguities(SciMLSensitivity, recursive = false)
    Aqua.test_deps_compat(SciMLSensitivity)
    Aqua.test_piracies(SciMLSensitivity;
        treat_as_own = [DiffEqBase._concrete_solve_adjoint,
            DiffEqBase._concrete_solve_forward])
    Aqua.test_project_extras(SciMLSensitivity)
    Aqua.test_stale_deps(SciMLSensitivity)
    Aqua.test_unbound_args(SciMLSensitivity)
    Aqua.test_undefined_exports(SciMLSensitivity)
end
