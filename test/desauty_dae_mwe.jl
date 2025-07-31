using ModelingToolkit, OrdinaryDiffEq
using ModelingToolkitStandardLibrary.Electrical
using ModelingToolkitStandardLibrary.Blocks: Sine
using NonlinearSolve
import SciMLStructures as SS
import SciMLSensitivity
using SymbolicIndexingInterface
import ModelingToolkit as MTK
using Zygote

function create_model(; C₁ = 3e-5, C₂ = 1e-6)
    @variables t
    @named resistor1 = Resistor(R = 5.0)
    @named resistor2 = Resistor(R = 2.0)
    @named capacitor1 = Capacitor(C = C₁)
    @named capacitor2 = Capacitor(C = C₂)
    @named source = Voltage()
    @named input_signal = Sine(frequency = 100.0)
    @named ground = Ground()
    @named ampermeter = CurrentSensor()

    eqs = [connect(input_signal.output, source.V)
           connect(source.p, capacitor1.n, capacitor2.n)
           connect(source.n, resistor1.p, resistor2.p, ground.g)
           connect(resistor1.n, capacitor1.p, ampermeter.n)
           connect(resistor2.n, capacitor2.p, ampermeter.p)]

    @named circuit_model = ODESystem(eqs, t,
        systems = [
            resistor1, resistor2, capacitor1, capacitor2,
            source, input_signal, ground, ampermeter
        ])
end

desauty_model = create_model()
sys = structural_simplify(desauty_model)

prob = ODEProblem(sys, [sys.resistor1.v => 1.0], (0.0, 0.1))
iprob = prob.f.initialization_data.initializeprob
isys = iprob.f.sys

tunables, repack, aliases = SS.canonicalize(SS.Tunable(), parameter_values(iprob))

linsolve = LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.QRFactorization)
sensealg = SciMLSensitivity.SteadyStateAdjoint(autojacvec = SciMLSensitivity.ZygoteVJP())
# sensealg = SciMLSensitivity.SteadyStateAdjoint(autojacvec = SciMLSensitivity.ZygoteVJP(), linsolve = linsolve)
igs, = Zygote.gradient(tunables) do p
    iprob2 = remake(iprob, p = repack(p))
    sol = solve(iprob2,
        sensealg = sensealg
    )
    sum(Array(sol))
end

@test !iszero(sum(igs))
