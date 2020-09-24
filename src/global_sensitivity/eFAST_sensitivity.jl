struct eFAST <: GSAMethod
    num_harmonics::Int
end
eFAST(; num_harmonics::Int=4) = eFAST(num_harmonics)

struct eFASTResult{T1}
    first_order::T1
    total_order::T1
end

function gsa(f,method::eFAST,p_range::AbstractVector;n::Int=1000,batch=false, kwargs...)
    @unpack num_harmonics = method
    num_params = length(p_range)
    omega = [ (n-1) ÷ (2*num_harmonics) ]
    m = omega[1] ÷ (2*num_harmonics)
    
    if m >= num_params-1
        append!(omega, floor.(Int, collect(range(1,stop=m,length=num_params-1))))
    else
        append!(omega, collect(range(0,stop=num_params-2)) .% m .+1 )
    end

    omega_temp = similar(omega)
    s = collect((2*pi/n) * (0:n-1))
    ps = zeros(num_params,n*num_params)

    for i in 1:num_params
        omega_temp[i] = omega[1]
        omega_temp[[k for k in 1:num_params if k != i]] = omega[2:end]
        l = collect((i-1)*n+1:i*n)
        phi = 2*pi*rand()
        for j in 1:num_params
            x =  0.5 .+ (1/pi) .*(asin.(sin.(omega_temp[j]*s .+ phi)))
            ps[j,l] .= quantile.(Uniform(p_range[j][1], p_range[j][2]),x)
        end
    end

    if batch
        all_y = f(ps)
        multioutput = all_y isa AbstractMatrix
        y_size = nothing
        gsa_efast_all_y_analysis(method, all_y, num_params, y_size, n, omega, Val(multioutput))
    else
        _y = [f(ps[:,j]) for j in 1:size(ps,2)]
        multioutput = !(eltype(_y) <: Number)
        if eltype(_y) <: RecursiveArrayTools.AbstractVectorOfArray
            y_size = size(_y[1])
            _y = vec.(_y)
        else
            y_size = nothing
        end
        all_y = multioutput ? reduce(hcat,_y) : _y
        gsa_efast_all_y_analysis(method, all_y, num_params, y_size, n, omega, Val(multioutput))
    end
end
function gsa_efast_all_y_analysis(method, all_y, num_params, y_size, n, omega, ::Val{multioutput}) where {multioutput}
    @unpack num_harmonics = method
    if multioutput
        size_ = size(all_y)
        first_order = Vector{Vector{eltype(all_y)}}(undef, num_params)
        total_order = Vector{Vector{eltype(all_y)}}(undef, num_params)
    else
        first_order = Vector{eltype(all_y)}(undef, num_params)
        total_order = Vector{eltype(all_y)}(undef, num_params)
    end
    for i in 1:num_params
        if !multioutput
            ft = (fft(all_y[(i-1)*n+1:i*n]))[2:(n ÷ 2)]
            ys = abs2.(ft .* inv(n))
            varnce = 2*sum(ys)
            first_order[i] = 2*sum(ys[(1:num_harmonics)*Int(omega[1])])/varnce
            total_order[i] = 1 .- 2*sum(ys[1:(omega[1] ÷ 2)])/varnce
        else
            ys = Vector{Vector{eltype(all_y)}}(undef, size(all_y,1))
            varnce = Vector{eltype(all_y)}(undef, size(all_y,1))
            for j ∈ eachindex(ys)
                ff = fft(all_y[j,(i-1)*n+1:i*n])[2:(n ÷ 2)]
                ys[j] = ysⱼ = abs2.(ff .* inv(n))
                varnce[j] = 2*sum(ysⱼ)
            end
            first_order[i] = map((y,var) -> 2*sum(y[(1:num_harmonics)*(omega[1])])./var, ys, varnce)
            total_order[i] = map((y,var) -> 1 .- 2*sum(y[1:(omega[1] ÷ 2)])./var, ys, varnce)
        end
    end
    if !isnothing(y_size)
        f_shape = let y_size = y_size
            x -> [reshape(x[:,i], y_size) for i in 1:size(x,2)]
        end
        first_order = map(f_shape,first_order)
        total_order = map(f_shape,total_order)
    end
    return eFASTResult(reduce(hcat,first_order), reduce(hcat,total_order))
end
