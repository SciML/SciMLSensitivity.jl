
mutable struct DGSM_Crossed
    crossedsq::Float64
    crossed::Float64
    abscrossed::Float64
end

function DGSM_Crossed(f,samples::Int64,distr::AbstractArray)
    
    k = length(distr)
    
    XX = [rand.(distr) for x = 1 : samples]
    
    hess(x) = ForwardDiff.hessian(f,x)
    
    dfdxdy = [hess(XX[i]) for i in 1 : samples]
    
    crossed = Dict{String, DGSM_Crossed}()
    
    
    for a in 1:k
        for b in a+1:k
            crossed["X$a:X$b"] = DGSM_Crossed(mean(dfdxdy[i][a+(b-1)*k]^2 for i in 1:samples),mean(dfdxdy[i][a+(b-1)*k] for i in 1:samples),mean(abs(dfdxdy[i][a+(b-1)*k]) for i in 1:samples))
        end
    end
    
    return crossed
end
