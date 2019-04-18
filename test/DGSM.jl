using Test, DiffEqSensitivity, Distributions

samples = 2000000

f1(x) = x[1] + 2*x[2] + 6.00*x[3]
dist1 = [Uniform(4,10),Normal(4,23),Beta(2,3)]
b = DGSM(f1,samples,dist1)
@test [b.a[1],b.absa[1],b.asq[1]] ≈ [1.0,1.0,1.0] atol=10e-1
@test [b.a[2],b.absa[2],b.asq[2]] ≈ [2.0,2.0,4.0] atol=10e-1
@test [b.a[3],b.absa[3],b.asq[3]] ≈ [6.0,6.0,36.0] atol=10e-1

#Test on classical Ishigami function
#Reference: https://hal.archives-ouvertes.fr/hal-01164215/document
#As provided in paper theoretical values of asq for the variables are
#asq for x[1] = 7.7
#asq for x[2] = 24.5
#asq for x[3] = 11.0

f2(x) = sin(x[1]) + 7sin(x[2])^2 + 0.1* (x[3]^4) *sin(x[1]) 
dist2 = [Uniform(-pi,pi),Uniform(-pi,pi),Uniform(-pi,pi)]
c = DGSM(f2,samples,dist2)
@test [c.asq[1]] ≈ [7.7] atol= 10e-1
@test [c.asq[2]] ≈ [24.5] atol= 10e-1
@test [c.asq[3]] ≈ [11.0] atol= 10e-1
@test [c.sigma[1]] ≈ [-14.7] atol= 10e-1
@test [c.sigma[2]] ≈ [-38.7] atol= 10e-1
@test [c.sigma[3]] ≈ [-42.0] atol= 10e-1
@test [c.tao[1]] ≈ [6.13] atol= 10e-1
@test [c.tao[2]] ≈ [16.94] atol= 10e-1
@test [c.tao[3]] ≈ [15.91] atol= 10e-1

#Some test functions from a paper
#Reference:New methods for the sensitivity analysis of
#black-box functions with an application to
#sheet metal forming A doctoral Thesis by Jana Fruth

f3(x) = 0.5*sin(50*x[1])+0.5
dist3 = [Uniform(0,1)]
d = DGSM(f3,samples,dist3)
@test [d.asq[1]] ≈ [310.9] atol= 10e-1

#Test on classical Ishigami function
#Reference: Roustant, Olivier, Jana Fruth, Bertrand Iooss and Sonja Kuhnt. “Derivative-Based Sensitivity Measures for Interactions.” (2013).
#As provided in paper theoretical values of asq for the variables are
#crossedsq for x1:x2 = 0.0
#crossedsq for x2:x3 = 0.0
#crossedsq for x1:x3 = 12.686


f2(x) = sin(x[1]) + 7*sin(x[2])^2 + 0.1* (x[3]^4) *sin(x[1]) 
dist2 = [Uniform(-pi,pi),Uniform(-pi,pi),Uniform(-pi,pi)]
c = DGSM(f2,samples,dist2,true)
@test [c.crossedsq[2]] ≈ [0.0] atol= 20e-1
@test [c.crossedsq[6]] ≈ [0.0] atol= 20e-1
@test [c.crossedsq[3]] ≈ [12.686] atol= 20e-1


