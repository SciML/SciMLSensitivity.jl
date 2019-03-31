using Test, DiffEqSensitivity, Distributions

samples = 2000000

f1(x) = x[1] + 2*x[2] + 6.00*x[3]
dist1 = [Uniform(4,10),Normal(4,23),Beta(2,3)]
b = DGSM(f1,samples,dist1)
@test [b[1].a,b[1].absa,b[1].asq] ≈ [1.0,1.0,1.0] atol=10e-1
@test [b[2].a,b[2].absa,b[2].asq] ≈ [2.0,2.0,4.0] atol=10e-1
@test [b[3].a,b[3].absa,b[3].asq] ≈ [6.0,6.0,36.0] atol=10e-1

#Test on classical Ishigami function
#Reference: https://hal.archives-ouvertes.fr/hal-01164215/document
#As provided in paper theoretical values of asq for the variables are
#asq for x[1] = 7.7
#asq for x[2] = 24.5
#asq for x[3] = 11.0

f2(x) = sin(x[1]) + 7sin(x[2])^2 + 0.1* (x[3]^4) *sin(x[1]) 
dist2 = [Uniform(-pi,pi),Uniform(-pi,pi),Uniform(-pi,pi)]
c = DGSM(f2,samples,dist2)
@test [c[1].asq] ≈ [7.7] atol= 10e-1
@test [c[2].asq] ≈ [24.5] atol= 10e-1
@test [c[3].asq] ≈ [11.0] atol= 10e-1

#Some test functions from a paper
#Reference:New methods for the sensitivity analysis of
#black-box functions with an application to
#sheet metal forming A doctoral Thesis by Jana Fruth

f3(x) = 0.5*sin(50*x[1])+0.5
dist3 = [Uniform(0,1)]
d = DGSM(f3,samples,dist3)
@test [d[1].asq] ≈ [310.9] atol= 10e-1


