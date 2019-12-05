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
             batch=false)
    Aᵦ = fuse_designs(A,B)
    d,n = size(A)
    all_points = hcat(A,B,reduce(hcat,Aᵦ))

    if batch
        all_y = f(all_points)
    else
        all_y = [f(all_points[:,i]) for i in 1:size(all_points,2)]
    end

    Ey = mean(all_y[1:2n])
    Vary = var(all_y[1:2n])

    fA = all_y[1:n]
    fB = all_y[(n+1):(2n)]
    fAⁱ= [all_y[(j*n+1):((j+1)*n)] for j in 2:(d+1)]

    Vᵢ = [sum(fB.*(fAⁱ[i].-fA)) for i in 1:d]./n

    Eᵢ = [Vary - sum(fA .* fAⁱ[i])./(n) + Ey^2 for i in 1:d]
    Eᵢ = [sum(abs2,fA-fAⁱ[i]) for i in 1:d]./(2n)
    Eᵢ = [sum(fA.*(fA.-fAⁱ[i])) for i in 1:d]./(n)

    #Eᵢⱼ = [sum(abs2,fAⁱ[i] - fAⁱ[j]) for i in 1:d, j in 1:d]./(2n)
    #Vᵢⱼ = Vary .- Eᵢⱼ
    #Sᵢⱼ= Vᵢⱼ./Vary

    Sᵢ = Vᵢ ./Vary
    Tᵢ = Eᵢ ./Vary
    SobolResult(Sᵢ,nothing,nothing,nothing,Tᵢ,nothing)
end
