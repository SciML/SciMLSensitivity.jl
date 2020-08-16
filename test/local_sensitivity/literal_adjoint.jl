using DiffEqSensitivity, OrdinaryDiffEq, Zygote, Test
function lv!(du, u, p, t)
    x,y = u
    a, b, c, d = p
    du[1] = a*x - b*x*y
    du[2] = -c*y + d*x*y
end
function test(u0,p)
    tspan = [0.,1.]
    prob = ODEProblem(lv!, u0, tspan, p)
    sol = solve(prob,Tsit5())
    return sol.u[end][1]
end
function test2(u0,p)
    tspan = [0.,1.]
    prob = ODEProblem(lv!, u0, tspan, p)
    sol = solve(prob,Tsit5())
    return Array(sol)[1,end]
end

u0 = [1.,1.]
p = [1.,1.,1.,1.]
@test Zygote.gradient(test,u0,p) == Zygote.gradient(test2,u0,p)
