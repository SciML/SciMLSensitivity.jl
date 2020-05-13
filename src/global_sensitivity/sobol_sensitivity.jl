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
    hcat(A,B,reduce(hcat,Aᵦ))
end

function gsa(f, method::Sobol, A::AbstractMatrix, B::AbstractMatrix;
             batch=false, Ei_estimator = :Jansen1999, kwargs...)
    
    d,n = size(A)
    n = Int(n/method.nboot)
    multioutput = false
    desol = false
    all_points = mapreduce(fuse_designs, hcat, [A[:,n*(i-1)+1:n*(i)] for i in 1:method.nboot], [B[:,n*(i-1)+1:n*(i)] for i in 1:method.nboot])

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

    Eys = multioutput ? Array{eltype(all_y)}[] : eltype(all_y)[]
    Varys = multioutput ? Array{eltype(all_y)}[] : eltype(all_y)[]
    Vᵢs = multioutput ? Array{eltype(all_y)}[] : Array{eltype(all_y)}[]
    Vᵢⱼs = multioutput ? Array{eltype(all_y)}[] : Array{eltype(all_y)}[]
    Eᵢs = multioutput ? Array{eltype(all_y)}[] : Array{eltype(all_y)}[]
    if !multioutput
        for i in 1:d+2:(d+2)*method.nboot
            push!(Eys,mean(all_y[(i-1)*n+1:(i+1)*n]))
            push!(Varys,var(all_y[(i-1)*n+1:(i+1)*n]))

            fA = all_y[(i-1)*n+1:i*n]
            fB = all_y[(i*n+1):(i+1)*n]
            fAⁱ= [all_y[(j*n+1):((j+1)*n)] for j in i+1:(i+d)]

            push!(Vᵢs,[sum(fB.*(fAⁱ[k].-fA)) for k in 1:d]./n)
            if 2 in method.order 
                push!(Vᵢⱼs,[sum(abs2, fAⁱ[k] - fAⁱ[j]) for k in 1:d for j in k+1:d]./(2n))
            end
            if Ei_estimator == :Homma1996
                push!(Eᵢs,[Varys[i] .- sum(fA .* fAⁱ[k])./(n) + Eys[i].^2 for k in 1:d])
            elseif Ei_estimator == :Sobol2007
                push!(Eᵢs,[sum(abs2,fA-fAⁱ[k]) for k in 1:d]./(2n))
            elseif Ei_estimator == :Jansen1999
                push!(Eᵢs,[sum(fA.*(fA.-fAⁱ[k])) for k in 1:d]./(n))
            end
        end
    else
        for i in 1:d+2:(d+2)*method.nboot
            push!(Eys,mean(all_y[:, (i-1)*n+1:(i+1)*n],dims=2))
            push!(Varys,var(all_y[:, (i-1)*n+1:(i+1)*n],dims=2))

            fA = all_y[:, (i-1)*n+1:i*n]
            fB = all_y[:, (i*n+1):(i+1)*n]
            fAⁱ= [all_y[:, (j*n+1):((j+1)*n)] for j in i+1:(i+d)]

            push!(Vᵢs,reduce(hcat, [sum(fB.*(fAⁱ[k].-fA), dims=2) for k in 1:d]./n))

            if 2 in method.order 
                push!(Vᵢⱼs,reduce(hcat, [sum(abs2, fAⁱ[k] - fAⁱ[j], dims=2) for k in 1:d for j in k+1:d]./(2n)))
            end
            if Ei_estimator == :Homma1996
                push!(Eᵢs,reduce(hcat, [Varys[i] .- sum(fA .* fAⁱ[k], dims=2)./(n) + Eys[i].^2 for k in 1:d]))
            elseif Ei_estimator == :Sobol2007
                push!(Eᵢs,reduce(hcat, [sum(abs2,fA-fAⁱ[k],dims=2) for k in 1:d]./(2n)))
            elseif Ei_estimator == :Jansen1999
                push!(Eᵢs,reduce(hcat, [sum(fA.*(fA.-fAⁱ[k]), dims=2) for k in 1:d]./(n)))
            end
        end
    end
    if 2 in method.order
        Sᵢⱼs = [(Vᵢⱼs[i])./Varys[i] for i in 1:method.nboot]
    end

    Sᵢs = [Vᵢs[i] ./Varys[i] for i in 1:method.nboot]
    Tᵢs = [Eᵢs[i] ./Varys[i] for i in 1:method.nboot]
    if method.nboot > 1
        size_ = size(Sᵢs[1])
        S1 = [[Sᵢ[i] for Sᵢ in Sᵢs] for i in 1:length(Sᵢs[1])]
        ST = [[Tᵢ[i] for Tᵢ in Tᵢs] for i in 1:length(Tᵢs[1])]

        function calc_ci(x)
            alpha = (1 - method.conf_int)
            tstar = quantile(TDist(length(x)-1), 1 - alpha/2)
            std(x)/sqrt(length(x))
        end
        S1_CI = map(calc_ci,S1)
        ST_CI = map(calc_ci,ST)

        if 2 in method.order
            size__= size(Sᵢⱼs[1])
            S2 = [[Sᵢⱼ[i] for Sᵢⱼ in Sᵢⱼs] for i in 1:length(Sᵢⱼs[1])]
            S2_CI = reshape(map(calc_ci,S2),size__...)
            Sᵢⱼ = reshape(mean.(S2),size__...)
        end

        Sᵢ = reshape(mean.(S1),size_...)
        Tᵢ = reshape(mean.(ST),size_...)
    else
        Sᵢ = Sᵢs[1]
        Tᵢ = Tᵢs[1]
        if 2 in method.order
            Sᵢⱼ = Sᵢⱼs[1]
        end
    end
    if desol 
        f_shape = x -> [reshape(x[:,i],y_size) for i in 1:size(x,2)]  
        Sᵢ = f_shape(Sᵢ)
        if 2 in method.order
            Sᵢⱼ = f_shape(Sᵢⱼ)
        end
        Tᵢ = f_shape(Tᵢ)
    end
    return SobolResult(Sᵢ,
                     method.nboot > 1 ? reshape(S1_CI,size_...) : nothing,  
                     2 in method.order ? Sᵢⱼ : nothing,  
                     method.nboot > 1 && 2 in method.order ? S2_CI : nothing, 
                     Tᵢ , 
                     method.nboot > 1 ? reshape(ST_CI,size_...) : nothing)
end

function gsa(f,method::Sobol,p_range::AbstractVector; N, kwargs...)
    A = QuasiMonteCarlo.generate_design_matrices(N, [i[1] for i in p_range], [i[2] for i in p_range], QuasiMonteCarlo.SobolSample(),2*method.nboot)
    gsa(f, method, hcat(A[1:method.nboot]...), hcat(A[method.nboot+1:end]...); kwargs...)
end
