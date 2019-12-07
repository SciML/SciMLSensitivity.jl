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
        _y = [f(all_points[:,i])' for i in 1:size(all_points,2)]
        multioutput = !(eltype(_y) <: Number)
        all_y = multioutput ? reduce(vcat,_y) : _y
    end

    if !multioutput

        Ey = mean(all_y[1:2n])
        Vary = var(all_y[1:2n])

        fA = all_y[1:n]
        fB = all_y[(n+1):(2n)]
        fAⁱ= [all_y[(j*n+1):((j+1)*n)] for j in 2:(d+1)]

        Vᵢ = [sum(fB.*(fAⁱ[i].-fA)) for i in 1:d]./n

        if Ei_estimator == :Homma1996
            Eᵢ = [Vary .- sum(fA .* fAⁱ[i])./(n) + Ey.^2 for i in 1:d]
        elseif Ei_estimator == :Sobol2007
            Eᵢ = [sum(abs2,fA-fAⁱ[i]) for i in 1:d]./(2n)
        elseif Ei_estimator == :Jansen1999
            Eᵢ = [sum(fA.*(fA.-fAⁱ[i])) for i in 1:d]./(n)
        end

    else

        Ey = mean(all_y[1:2n,:],dims=1)
        Vary = var(all_y[1:2n,:],dims=1)

        fA = all_y[1:n,:]
        fB = all_y[(n+1):(2n),:]
        fAⁱ= [all_y[(j*n+1):((j+1)*n),:] for j in 2:(d+1)]

        Vᵢ = reduce(vcat,[sum(fB.*(fAⁱ[i].-fA),dims=1) for i in 1:d]./n)

        if Ei_estimator == :Homma1996
            Eᵢ = reduce(vcat,[Vary .- sum(fA .* fAⁱ[i],dims=1)./(n) + Ey.^2 for i in 1:d])
        elseif Ei_estimator == :Sobol2007
            Eᵢ = reduce(vcat,[sum(abs2,fA-fAⁱ[i],dims=1) for i in 1:d]./(2n))
        elseif Ei_estimator == :Jansen1999
            Eᵢ = reduce(vcat,[sum(fA.*(fA.-fAⁱ[i]),dims=1) for i in 1:d]./(n))
        end

    end
    #Eᵢⱼ = [sum(abs2,fAⁱ[i] - fAⁱ[j]) for i in 1:d, j in 1:d]./(2n)
    #Vᵢⱼ = Vary .- Eᵢⱼ
    #Sᵢⱼ= Vᵢⱼ./Vary

    Sᵢ = Vᵢ ./Vary
    Tᵢ = Eᵢ ./Vary
    SobolResult(Sᵢ,nothing,nothing,nothing,Tᵢ,nothing)
end
