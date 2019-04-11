mutable struct DGSM
    a::Array
    absa::Array
    asq::Array
    crossed::Array
    abscrossed::Array
    crossedsq::Array
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
4.crossed:
    A string(True/False) which act as indicator for computation of DGSM crossed indices
    Eg- a True value over there will lead to evauation of crossed indices
"""
function DGSM(f,samples::Int,distr::AbstractArray, crossed::Bool = false)
    
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

    a = [mean(dfdx[:,i]) for i in 1:k]
    asq = [mean(dfdx[:,i].^2) for i in 1:k]
    absa = [mean(abs.(dfdx[:,i])) for i in 1:k]
    
    cross1 = zeros(Float64,0,0)
    cross2 = zeros(Float64,0,0)
    cross3 = zeros(Float64,0,0)
    
    if crossed == true
        cross = DGSM_Crossed(f,samples,distr)
        cross1 = cross[1]
        cross2 = cross[2]
        cross3 = cross[3]
    end
    
    DGSM_Vi = DGSM(a, absa, asq, cross1, cross2, cross3)
    return DGSM_Vi
    #returns a struct of 6 elements i.e. a,absa,asq(all 3 arrays) and cross1(crossed), cross2(abscrossed),cross3(crossedsq) (all 3 matrices)
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
3.distr:
    Array of distribution of respective variables
    Eg- dist = [Normal(5,6),Uniform(2,3)]
	for two variables
"""
function DGSM_Crossed(f,samples::Int,distr::AbstractArray)
    
    k = length(distr)
    
    #XX is the matrix consisting of 'samples' number of sampling based on respective 
    #distributions of variables
    
    XX = [rand.(distr) for x = 1 : samples]
    
    #function to evaluate hessian of f wrt x
    hess(x) = ForwardDiff.hessian(f,x)
    
    #Evaluating the derivatives with AD
    dfdxdy = [hess(XX[i]) for i in 1 : samples]
    
    crossed = zeros(Float64,k,k)
    crossedsq = zeros(Float64,k,k)
    abscrossed = zeros(Float64,k,k)
    
    #computing the elements of crossed, crossedsq, abscrossed
    
    for a in 1:k
        for b in a+1:k
            crossed[b + (a-1)*k] = mean(dfdxdy[i][b + (a-1)*k] for i in 1:samples)
            crossed[a + (b-1)*k] = crossed[b + (a-1)*k]
            crossedsq[b + (a-1)*k] = mean(dfdxdy[i][b + (a-1)*k]^2 for i in 1:samples)
            crossedsq[a + (b-1)*k] = crossedsq[b + (a-1)*k]
            abscrossed[b + (a-1)*k] = mean(abs(dfdxdy[i][b + (a-1)*k]) for i in 1:samples)
            abscrossed[a + (b-1)*k] = crossed[b + (a-1)*k]
        end
    end
    
    #returns a tuple of 3 matrices consisting of crossed, abscrossed, crossedsq
    return crossed, abscrossed, crossedsq
end

