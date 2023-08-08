using OrdinaryDiffEq, SciMLSensitivity, ForwardDiff, Zygote, Test
A = [0.0 1.0; 1.0 0.0; 0 0; 0 0];

B = [1.0 0.0; 0.0 1.0; 0 0; 0 0];

utarget = A;
const T = 10.0;

function f(u, p, t)
    return -p[1] * u # just a silly example to demonstrate the issue
end

u0 = [1.0 0.0; 0.0 1.0; 0.0 0.0; 0.0 0.0];

tspan = (0.0, T)
tsteps = 0.0:(T / 100.0):T

p = [1.7, 1.0, 3.0, 1.0]

prob_ode = ODEProblem(f, u0, tspan, p);

fd_ode = ForwardDiff.gradient(p) do p
    sum(last(solve(prob_ode, Tsit5(), p = p, abstol = 1e-12, reltol = 1e-12)))
end

grad_ode = Zygote.gradient(p) do p
    sum(last(solve(prob_ode, Tsit5(), p = p, abstol = 1e-12, reltol = 1e-12)))
end[1]

@test fd_ode≈grad_ode rtol=1e-6

grad_ode = Zygote.gradient(p) do p
    sum(last(solve(prob_ode, Tsit5(), p = p, abstol = 1e-12, reltol = 1e-12,
        sensealg = InterpolatingAdjoint())))
end[1]

@test fd_ode≈grad_ode rtol=1e-6
