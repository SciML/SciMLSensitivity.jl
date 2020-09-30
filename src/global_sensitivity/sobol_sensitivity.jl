# @with_kw
mutable struct Sobol <: GSAMethod
    order::Vector{Int}
    nboot::Int
    conf_int::Float64
end
Sobol(; order = [0, 1], nboot = 1, conf_int = 0.95) = Sobol(order, nboot, conf_int)
      
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

function gsa(f, method::Sobol, A::AbstractMatrix{TA}, B::AbstractMatrix;
             batch=false, Ei_estimator = :Jansen1999, kwargs...) where {TA}
    d,n = size(A)
    n = Int(n/method.nboot)
    multioutput = false
    nboot = method.nboot # load to help alias analysis
    Anb = Vector{Matrix{TA}}(undef, nboot)
    for i ∈ 1:nboot
        Anb[i] = A[:,n*(i-1)+1:n*(i)]
    end
    Bnb = Vector{Matrix{TA}}(undef, nboot)
    for i ∈ 1:nboot
        Bnb[i] = B[:,n*(i-1)+1:n*(i)]
    end
    all_points = mapreduce(fuse_designs, hcat, Anb, Bnb)

    if batch
        all_y = f(all_points)
        multioutput = all_y isa AbstractMatrix
        y_size = nothing
        gsa_sobol_all_y_analysis(method, all_y, d, n, Ei_estimator, y_size, Val(multioutput))
    else
        _y = [f(all_points[:, i]) for i in 1:size(all_points, 2)]
        multioutput = !(eltype(_y) <: Number)
        if eltype(_y) <: RecursiveArrayTools.AbstractVectorOfArray
            y_size = size(_y[1])
            _y = vec.(_y)
        else
            y_size = nothing
        end
        if multioutput
            gsa_sobol_all_y_analysis(method, reduce(hcat, _y), d, n, Ei_estimator, y_size, Val(true))
        else
            gsa_sobol_all_y_analysis(method, _y, d, n, Ei_estimator, y_size, Val(false))
        end
    end
end
function gsa_sobol_all_y_analysis(method, all_y::AbstractArray{T}, d, n, Ei_estimator, y_size, ::Val{multioutput}) where {T, multioutput}
    nboot = method.nboot 
    Eys = multioutput ? Matrix{T}[] : T[]
    Varys = multioutput ? Matrix{T}[] : T[]
    Vᵢs = multioutput ? Matrix{T}[] : Vector{T}[]
    Vᵢⱼs = multioutput ? Matrix{T}[] : Vector{T}[]
    Eᵢs = multioutput ? Matrix{T}[] : Vector{T}[]
    if !multioutput
        for i in 1:d+2:(d+2)*nboot
            push!(Eys,mean(all_y[(i-1)*n+1:(i+1)*n]))
            push!(Varys,var(all_y[(i-1)*n+1:(i+1)*n]))

            fA = all_y[(i-1)*n+1:i*n]
            fB = all_y[(i*n+1):(i+1)*n]
            fAⁱ= [all_y[(j*n+1):((j+1)*n)] for j in i+1:(i+d)]

            push!(Vᵢs, [sum(fB.*(fAⁱ[k].-fA)) for k in 1:d]./n)
            if 2 in method.order 
                push!(Vᵢⱼs, [sum(abs2, fAⁱ[k] - fAⁱ[j]) for k in 1:d for j in k+1:d] ./= (2n))
            end
            if Ei_estimator === :Homma1996
                push!(Eᵢs,[Varys[i] .- sum(fA .* fAⁱ[k])./(n) + Eys[i].^2 for k in 1:d])
            elseif Ei_estimator === :Sobol2007
                push!(Eᵢs,[sum(abs2,fA-fAⁱ[k]) for k in 1:d]./(2n))
            elseif Ei_estimator === :Jansen1999
                push!(Eᵢs,[sum(fA.*(fA.-fAⁱ[k])) for k in 1:d]./(n))
            end
        end
    else
        for i in 1:d+2:(d+2)*nboot
            push!(Eys,mean(all_y[:, (i-1)*n+1:(i+1)*n],dims=2))
            push!(Varys,var(all_y[:, (i-1)*n+1:(i+1)*n],dims=2))

            fA = all_y[:, (i-1)*n+1:i*n]
            fB = all_y[:, (i*n+1):(i+1)*n]
            fAⁱ= [all_y[:, (j*n+1):((j+1)*n)] for j in i+1:(i+d)]

            push!(Vᵢs,reduce(hcat, [sum(fB.*(fAⁱ[k].-fA), dims=2) for k in 1:d]./n))

            if 2 in method.order 
                push!(Vᵢⱼs,reduce(hcat, [sum(abs2, fAⁱ[k] - fAⁱ[j], dims=2) for k in 1:d for j in k+1:d]./(2n)))
            end
            if Ei_estimator === :Homma1996
                push!(Eᵢs,reduce(hcat, [Varys[i] .- sum(fA .* fAⁱ[k], dims=2)./(n) + Eys[i].^2 for k in 1:d]))
            elseif Ei_estimator === :Sobol2007
                push!(Eᵢs,reduce(hcat, [sum(abs2,fA-fAⁱ[k],dims=2) for k in 1:d]./(2n)))
            elseif Ei_estimator === :Jansen1999
                push!(Eᵢs,reduce(hcat, [sum(fA.*(fA.-fAⁱ[k]), dims=2) for k in 1:d]./(n)))
            end
        end
    end
    if 2 in method.order
        Sᵢⱼs = similar(Vᵢⱼs)
        for i ∈ 1:nboot
            Sᵢⱼs[i] = Vᵢⱼs[i] ./ Varys[i]
        end
    end

    Sᵢs = [Vᵢs[i] ./Varys[i] for i in 1:nboot]
    Tᵢs = [Eᵢs[i] ./Varys[i] for i in 1:nboot]
    if nboot > 1
        size_ = size(Sᵢs[1])
        S1 = [[Sᵢ[i] for Sᵢ in Sᵢs] for i in 1:length(Sᵢs[1])]
        ST = [[Tᵢ[i] for Tᵢ in Tᵢs] for i in 1:length(Tᵢs[1])]

        function calc_ci(x,mean=nothing)
            alpha = (1 - method.conf_int)
            std(x,mean=mean)/sqrt(length(x))
        end
        S1_CI = map(calc_ci,S1)
        ST_CI = map(calc_ci,ST)

        if 2 in method.order
            size__= size(Sᵢⱼs[1])
            S2_CI = Array{T}(undef, size__)
            Sᵢⱼ = Array{T}(undef, size__)
            b = getindex.(Sᵢⱼs, 1)
            Sᵢⱼ[1] = b̄ = mean(b)
            S2_CI[1] = calc_ci(b, b̄)
            for i ∈ 2:length(Sᵢⱼs[1])
                b .= getindex.(Sᵢⱼs, i)
                Sᵢⱼ[i] = b̄ = mean(b)
                S2_CI[i] = calc_ci(b, b̄)
            end
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
    if isnothing(y_size)
        _Sᵢ = Sᵢ
        _Tᵢ = Tᵢ
        _Sᵢⱼ = 2 in method.order ? Sᵢⱼ : nothing
    else
        f_shape = let y_size = y_size
            x -> [reshape(x[:,i],y_size) for i in 1:size(x,2)]
        end
        _Sᵢ = f_shape(Sᵢ)
        _Sᵢⱼ = 2 in method.order ? f_shape(Sᵢⱼ) : nothing
        _Tᵢ = f_shape(Tᵢ)
    end
    return SobolResult(_Sᵢ,
                     nboot > 1 ? reshape(S1_CI,size_...) : nothing,  
                     2 in method.order ? Sᵢⱼ : nothing,  
                     nboot > 1 && 2 in method.order ? S2_CI : nothing, 
                     _Tᵢ , 
                     nboot > 1 ? reshape(ST_CI,size_...) : nothing)
end

function gsa(f,method::Sobol,p_range::AbstractVector; N, kwargs...)
    AB = QuasiMonteCarlo.generate_design_matrices(N, [i[1] for i in p_range], [i[2] for i in p_range], QuasiMonteCarlo.SobolSample(),2*method.nboot)
    A = reduce(hcat, @view(AB[1:method.nboot]))
    B = reduce(hcat, @view(AB[method.nboot+1:end]))
    gsa(f, method, A, B; kwargs...)
end
