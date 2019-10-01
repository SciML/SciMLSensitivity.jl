Base.@kwdef struct eFAST
    num_harmonics::Int=4
    num_params::Int
    bounds::AbstractVector
end

struct eFASTResult
    first_order::AbstractVector
    total_order::AbstractVector
end

function gsa(f,n,method::eFAST)
    @unpack num_harmonics,p_len,bounds = method
    omega = [floor((n-1)/(2*num_harmonics))]
    m = floor(omega[1]/(2*num_harmonics))
    if m>= p_len-1
        append!(omega,floor.(collect(range(1,stop=m,length=p_len-1))))
    else
        append!(omega,collect(range(0,stop=p_len-2)).%m .+1)
    end
    omega_temp = similar(omega)
    first_order = []
    total_order = []
    s = collect((2*pi/n) * (0:n-1))
    ps = zeros(n*p_len,p_len)

    for i in 1:p_len
        omega_temp[i] = omega[1]
        omega_temp[[k for k in 1:p_len if k != i]] = omega[2:end]
        l = collect((i-1)*n+1:i*n)
        phi = 2*pi*rand()
        for j in 1:p_len
            x =  0.5 .+ (1/pi) .*(asin.(sin.(omega_temp[j]*s .+ phi)))
            ps[l,j] .= quantile.(Uniform(bounds[j][1],bounds[j][2]),x)
        end
    end

    for i in 1:p_len
        ys_p = [f(ps[j,:]) for j in (i-1)*n+1:i*n]
        ft = (fft(ys_p))[2:Int(floor((n/2)))]
        ys = [((abs.(ff))./n).^2 for ff in ft]
        varnce = 2*sum(ys)
        push!(first_order,2*sum(ys[(1:num_harmonics)*Int(omega[1])])/varnce)
        push!(total_order,1 .- 2*sum(ys[1:Int(omega[1]/2)])/varnce)
    end

    return first_order, total_order
end