
mutable struct DGSM
    a::Float64
    absa::Float64
    asq::Float64
end

"""
The inputs to the function DGSM are as follows:
1.f: 
    This is the input function based on which the values of DGSM are to be evaluated
    Eg- f(x) = x[1]+x[2]^2
        This is function in 2 variables
2.samples:
    Depicts the number of sampling set of points to be used for evaluation of E(a), E(|a|) and E(a^2)
    a = partial derivative of f wrt x_i
3.distri:
    Array of distribution of respective variables
    Eg- dist = [Normal(5,6),Uniform(2,3)]
	for two variables
"""
function DGSM(f,samples::Int,distr::AbstractArray)
    
    k = length(distr)
    
    #XX is the matrix consisting of 'samples' number of sampling based on respective 
    #distributions of variables
    
    XX = [rand.(distr) for x = 1:samples]
    
    #function to evaluate gradient of f wrt x
    grad(x)= ForwardDiff.gradient(f,x)
    
    #Evaluating the derivatives with AD
    
    dfdx = [grad(XX[i]) for i = 1:samples]
    dfdx = reduce(hcat,dfdx)
    dfdx = dfdx'
    
    
    #Evaluating E(a) E(|a|) and E(a^2)
    
    DGSM_Vi = [DGSM(mean(dfdx[:,i]),mean(abs.(dfdx[:,i])),mean(dfdx[:,i].^2)) for i in 1:k]
    
    #This function finally returns an array of structures, consisting a, absa and asq
    #respectively for the k independent parameters
    
    return DGSM_Vi
end
