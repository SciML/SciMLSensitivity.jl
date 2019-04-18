mutable struct DGSM{T}
    a::Array{T,1}
    absa::Array{T,1}
    asq::Array{T,1}
    sigma::Array{T,1}
    tao::Array{T,1}
    crossed::Union{Nothing,Array{T,2}}
    abscrossed::Union{Nothing,Array{T,2}}
    crossedsq::Union{Nothing,Array{T,2}}
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
    
    #function to evaluate hessian of f wrt x
    hess(x) = ForwardDiff.hessian(f,x)
    
    #Evaluating the derivatives with AD
    
    dfdx = [grad(XX[i]) for i = 1:samples]
    dfdx = reduce(hcat,dfdx)
    dfdx = dfdx'
    
    
    #Evaluating E(a) E(|a|) and E(a^2)

    a = [mean(dfdx[:,i]) for i in 1:k]
    asq = [mean(dfdx[:,i].^2) for i in 1:k]
    absa = [mean(abs.(dfdx[:,i])) for i in 1:k]
    
    sigma = zeros(Float64,k)
    tao = zeros(Float64,k)

    #Evaluating tao_i for all input parameters
    
    for i in 1:k
        for j in 1:samples
            tao[i] += (dfdx[j + (i-1)*samples]^2)*(1 - 3*XX[j][i] + XX[j][i]^2)/6
        end
        tao[i] = tao[i]/samples
    end

    #Evaluating sigma_i for all input parameters

    for i in 1:k
        for j in 1:samples
            sigma[i] += 0.5*(XX[j][i])*(1-XX[j][i])*dfdx[j + (i-1)*samples]^2
        end
        sigma[i] = sigma[i]/samples
    end
    
    if crossed == true
        
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
                abscrossed[a + (b-1)*k] = abscrossed[b + (a-1)*k]
            end
        end
        
    else
    	return DGSM(a, absa, asq, sigma, tao, nothing, nothing, nothing)
    end
    
    return DGSM(a, absa, asq, sigma, tao, crossed, abscrossed, crossedsq)
    #returns a struct of 7 elements i.e. a, absa, asq, sigma(all 4 arrays) and crossed, abscrossed, crossedsq (all 3 matrices)
end


