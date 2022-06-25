using OrdinaryDiffEq, SciMLSensitivity, Zygote, Test

function loss1(p;sensealg=nothing)
  f(x,p,t) = [p[1]]
  prob = DiscreteProblem(f, [0.0], (1,10), p)
  sol = solve(prob, FunctionMap(scale_by_time = true), saveat=[1,2,3])
  return sum(sol)
end
dp1 = Zygote.gradient(loss1,[1.0])[1]
dp2 = Zygote.gradient(x->loss1(x,sensealg=TrackerAdjoint()),[1.0])[1]
dp3 = Zygote.gradient(x->loss1(x,sensealg=ReverseDiffAdjoint()),[1.0])[1]
dp4 = Zygote.gradient(x->loss1(x,sensealg=ForwardDiffSensitivity()),[1.0])[1]
@test dp1 == dp2
@test dp1 == dp3
@test dp1 == dp4
