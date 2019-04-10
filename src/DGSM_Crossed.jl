
mutable struct DGSM_Crossed
    crossedsq::Float64
    crossed::Float64
    abscrossed::Float64
end

"""
The inputs to the function DGSM_Crossed are as follows:
1.f: 
    This is the input function based on which the values of Crossed DGSM are to be evaluated
    Eg- f(x) = x[1]+x[2]^2
        This is function in 2 variables
2.samples:
    Depicts the number of sampling set of points to be used for evaluation of E(c), E(|c|) and E(c^2)
    c = partial derivative of 2nd order of f wrt x_i,x_j. 
3.distri:
    Array of distribution of respective variables
    Eg- dist = [Normal(5,6),Uniform(2,3)]
	for two variables
"""
function DGSM_Crossed(f,samples::Int64,distr::AbstractArray)
    
    k = length(distr)
    
    #XX is the matrix consisting of 'samples' number of sampling based on respective 
    #distributions of variables
    
    XX = [rand.(distr) for x = 1 : samples]
    
    #function to evaluate hessian of f wrt x
    hess(x) = ForwardDiff.hessian(f,x)
    
    #Evaluating the derivatives with AD
    dfdxdy = [hess(XX[i]) for i in 1 : samples]
    
    #creating a dicitionay which is returned
    crossed = Dict{String, DGSM_Crossed}()
    
    #assigning elements of dictionary, the key would be of the form "XiXj" and the data would be of type struct DGSM_Crossed
    #which consists of E(c), E(|c|) and E(c^2) where c is partial derivative of 2nd order of f wrt x_i,x_j. 
    for a in 1:k
        for b in a+1:k
            crossed["X$a:X$b"] = DGSM_Crossed(mean(dfdxdy[i][a+(b-1)*k]^2 for i in 1:samples),mean(dfdxdy[i][a+(b-1)*k] for i in 1:samples),mean(abs(dfdxdy[i][a+(b-1)*k]) for i in 1:samples))
        end
    end
    
    return crossed
end
