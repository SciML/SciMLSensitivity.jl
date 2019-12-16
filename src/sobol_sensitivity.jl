@with_kw mutable struct Sobol <: GSAMethod
    order::Array{Int}=[0,1]
    nboot::Int=0
    conf_int::Float64=0.95
end

mutable struct SobolResult{T1,T2,T3,T4}
    S1::T1
    S1_Conf_Int::T2
    S2::T3
    S2_Conf_Int::T4
    ST::T1
    ST_Conf_Int::T2
end

function fuse_designs(A,B)
    d = size(A,1)
    Aᵦ = [copy(A) for i in 1:d]
    for i in 1:d
        Aᵦ[i][i,:] = B[i,:]
    end
    Aᵦ
end


# function fuse_designs_second(A,B)
#     d = size(A,1)
#     Aᵦ = [copy(A) for i in 1:(d*(d-1))/2]
#     k = 1
#     for i in 1:d
#         for j in i+1:d
#             Aᵦ[k][i,:] = B[i,:]
#             Aᵦ[k][j,:] = B[j,:]
#             k += 1
#         end
#     end
#     Aᵦ
# end

function gsa(f,method::Sobol,A::AbstractMatrix,B::AbstractMatrix;
             batch=false,Ei_estimator = :Jansen1999)
    Aᵦ = fuse_designs(A,B)
    d,n = size(A)
    multioutput = false
    all_points = hcat(A,B,reduce(hcat,Aᵦ))

    if batch
        all_y = f(all_points)
        multioutput = all_y isa AbstractMatrix
    else
        _y = [f(all_points[:,i]) for i in 1:size(all_points,2)]
        multioutput = !(eltype(_y) <: Number)
        all_y = multioutput ? reduce(hcat,_y) : _y
    end

    if !multioutput

        Ey = mean(all_y[1:2n])
        Vary = var(all_y[1:2n])

        fA = all_y[1:n]
        fB = all_y[(n+1):(2n)]
        fAⁱ= [all_y[(j*n+1):((j+1)*n)] for j in 2:(d+1)]
        if 2 in method.order 
            fAⁱʲ= all_y[Int(end - (d*(d-1))/2):end]
        end
        Vᵢ = [sum(fB.*(fAⁱ[i].-fA)) for i in 1:d]./n
        if 2 in method.order 
            Vᵢⱼ = [sum(abs2, fAⁱ[i] - fAⁱ[j]) for i in 1:d for j in i+1:d]./(2n)
        end
        if Ei_estimator == :Homma1996
            Eᵢ = [Vary .- sum(fA .* fAⁱ[i])./(n) + Ey.^2 for i in 1:d]
        elseif Ei_estimator == :Sobol2007
            Eᵢ = [sum(abs2,fA-fAⁱ[i]) for i in 1:d]./(2n)
        elseif Ei_estimator == :Jansen1999
            Eᵢ = [sum(fA.*(fA.-fAⁱ[i])) for i in 1:d]./(n)
        end

    else

        Ey = mean(all_y[:,1:2n],dims=2)
        Vary = var(all_y[:,1:2n],dims=2)

        fA = all_y[:,1:n]
        fB = all_y[:,(n+1):(2n)]
        fAⁱ= [all_y[:,(j*n+1):((j+1)*n)] for j in 2:(d+1)]

        Vᵢ = reduce(hcat,[sum(fB.*(fAⁱ[i].-fA),dims=2) for i in 1:d]./n)

        if 2 in method.order 
            Vᵢⱼ = reduce(hcat, [sum(abs2, fAⁱ[i] - fAⁱ[j]) for i in 1:d for j in i+1:d]./(2n))
        end
        if Ei_estimator == :Homma1996
            Eᵢ = reduce(hcat,[Vary .- sum(fA .* fAⁱ[i],dims=2)./(n) + Ey.^2 for i in 1:d])
        elseif Ei_estimator == :Sobol2007
            Eᵢ = reduce(hcat,[sum(abs2,fA-fAⁱ[i],dims=2) for i in 1:d]./(2n))
        elseif Ei_estimator == :Jansen1999
            Eᵢ = reduce(hcat,[sum(fA.*(fA.-fAⁱ[i]),dims=2) for i in 1:d]./(n))
        end

    end
    if 2 in method.order
        Sᵢⱼ= (Vᵢⱼ)./Vary
    end

    Sᵢ = Vᵢ ./Vary
    Tᵢ = Eᵢ ./Vary
    SobolResult(Sᵢ,nothing,2 in method.order ? Sᵢⱼ : nothing ,nothing,Tᵢ,nothing)
end
