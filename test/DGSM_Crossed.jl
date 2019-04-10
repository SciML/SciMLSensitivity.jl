using Test, DiffEqSensitivity, Distributions

#Test on classical Ishigami function
#Reference: Roustant, Olivier, Jana Fruth, Bertrand Iooss and Sonja Kuhnt. “Derivative-Based Sensitivity Measures for Interactions.” (2013).
#As provided in paper theoretical values of asq for the variables are
#crossedsq for x1:x2 = 0.0
#crossedsq for x2:x3 = 0.0
#crossedsq for x1:x3 = 12.686

samples = 1000000

f2(x) = sin(x[1]) + 7*sin(x[2])^2 + 0.1* (x[3]^4) *sin(x[1]) 
dist2 = [Uniform(-pi,pi),Uniform(-pi,pi),Uniform(-pi,pi)]
c = DGSM_Crossed(f2,samples,dist2)
@test [c["X1:X2"].crossedsq] ≈ [0.0] atol= 20e-1
@test [c["X2:X3"].crossedsq] ≈ [0.0] atol= 20e-1
@test [c["X1:X3"].crossedsq] ≈ [12.686] atol= 20e-1
