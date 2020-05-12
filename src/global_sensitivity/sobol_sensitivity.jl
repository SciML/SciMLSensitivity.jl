@with_kw mutable struct Sobol <: GSAMethod
    order::Array{Int}=[0, 1]
    nboot::Int=1
    conf_int::Float64=0.95
end

mutable struct SobolResult{T1, T2, T3, T4}
    S1::T1
    S1_Conf_Int::T2
    S2::T3
    S2_Conf_Int::T4
    ST::T1
    ST_Conf_Int::T2
end

function fuse_designs(A, B)
    d = size(A,1)
    Aᵦ = [copy(A) for i in 1:d]
    for i in 1:d
        Aᵦ[i][i,:] = B[i,:]
    end
    Aᵦ
end

function gsa(f, method::Sobol, A::AbstractMatrix, B::AbstractMatrix;
             batch=false, Ei_estimator = :Jansen1999, kwargs...)
    Aᵦ = fuse_designs(A, B)
    d,n = size(A)
    multioutput = false
    desol = false
    all_points = hcat(A, B, reduce(hcat,Aᵦ))

    if batch
        all_y = f(all_points)
        multioutput = all_y isa AbstractMatrix
    else
        _y = [f(all_points[:, i]) for i in 1:size(all_points, 2)]
        multioutput = !(eltype(_y) <: Number)
        if eltype(_y) <: RecursiveArrayTools.AbstractVectorOfArray
            y_size = size(_y[1])
            _y = vec.(_y)
            desol = true
        end
        all_y = multioutput ? reduce(hcat, _y) : _y
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

        Ey = mean(all_y[:, 1:2n],dims=2)
        Vary = var(all_y[:, 1:2n],dims=2)

        fA = all_y[:, 1:n]
        fB = all_y[:, (n+1):(2n)]
        fAⁱ= [all_y[:, (j*n+1):((j+1)*n)] for j in 2:(d+1)]

        Vᵢ = reduce(hcat, [sum(fB.*(fAⁱ[i].-fA), dims=2) for i in 1:d]./n)

        if 2 in method.order 
            Vᵢⱼ = reduce(hcat, [sum(abs2, fAⁱ[i] - fAⁱ[j], dims=2) for i in 1:d for j in i+1:d]./(2n))
        end
        if Ei_estimator == :Homma1996
            Eᵢ = reduce(hcat, [Vary .- sum(fA .* fAⁱ[i], dims=2)./(n) + Ey.^2 for i in 1:d])
        elseif Ei_estimator == :Sobol2007
            Eᵢ = reduce(hcat, [sum(abs2,fA-fAⁱ[i],dims=2) for i in 1:d]./(2n))
        elseif Ei_estimator == :Jansen1999
            Eᵢ = reduce(hcat, [sum(fA.*(fA.-fAⁱ[i]), dims=2) for i in 1:d]./(n))
        end

    end
    if 2 in method.order
        Sᵢⱼ= (Vᵢⱼ)./Vary
    end

    Sᵢ = Vᵢ ./Vary
    Tᵢ = Eᵢ ./Vary
    if desol 
        f_shape = x -> [reshape(x[:,i],y_size) for i in 1:size(x,2)]  
        Sᵢ = f_shape(Sᵢ)
        if 2 in method.order
            Sᵢⱼ = f_shape(Sᵢⱼ)
        end
        Tᵢ = f_shape(Tᵢ)
    end
    SobolResult(Sᵢ, nothing, 2 in method.order ? Sᵢⱼ : nothing, nothing, Tᵢ, nothing)
end

function gsa(f,method::Sobol,A::AbstractVector{<:AbstractMatrix},B::AbstractVector{<:AbstractMatrix};kwargs...)
    d,n = size(A[1])
    res = map(A,B) do A_,B_
        Threads.@spawn gsa(f, method, A_, B_; kwargs...)
    end
    sobolres = fetch.(res)
    if method.nboot > 1
        size_ = size(sobolres[1].S1)
        S1 = [[sobol.S1[i] for sobol in sobolres] for i in 1:length(sobolres[1].S1)]
        ST = [[sobol.ST[i] for sobol in sobolres] for i in 1:length(sobolres[1].ST)]
        
        function calc_ci(x)
            alpha = (1 - method.conf_int)
            tstar = quantile(TDist(length(x)-1), 1 - alpha/2)
            std(x)/sqrt(length(x))
        end
        S1_CI = map(calc_ci,S1)
        ST_CI = map(calc_ci,ST)

        if 2 in method.order
           size__= size(sobolres[1].S2)
            S2 = [[sobol.S2[i] for sobol in sobolres] for i in 1:length(sobolres[1].S2)]
            S2_CI = reshape(map(calc_ci,S2),size__...)
            Sᵢⱼ = reshape(mean.(S2),size__...)
        end
        return SobolResult(reshape(mean.(S1),size_...), reshape(S1_CI,size_...),  2 in method.order ? Sᵢⱼ : nothing, 2 in method.order ? S2_CI : nothing, reshape(mean.(ST),size_...), reshape(ST_CI,size_...))
    else
        return sobolres[1]
    end
end

function gsa(f,method::Sobol,p_range::AbstractVector; N, kwargs...)
    A = QuasiMonteCarlo.generate_design_matrices(N, [i[1] for i in p_range], [i[2] for i in p_range], QuasiMonteCarlo.SobolSample(),2*method.nboot)
    gsa(f, method, A[1:method.nboot], A[method.nboot+1:end]; kwargs...)
end
