using OrdinaryDiffEq, DiffEqSensitivity, Zygote
A = [0. 1.; 1. 0.; 0 0; 0 0];
B = [1. 0.; 0. 1.; 0 0; 0 0];

utarget = A;
const T = 10.0;

function f(u, p, t)
    return -p[1]*u # just a silly example to demonstrate the issue
end


u0 = [1.0 0.0; 0.0 1.0; 0.0 0.0; 0.0 0.0];

tspan = (0.0, T)
tsteps = 0.0:T/100.0:T

p = [1.7, 1.0, 3.0, 1.0]

prob_ode = ODEProblem(f, u0, tspan, p);
sol_ode = Zygote.gradient(p->sum(last(solve(prob_ode, Tsit5(),p=p))),p)
