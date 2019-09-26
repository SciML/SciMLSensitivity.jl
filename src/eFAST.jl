using FFTW, Distributions

function fast(f,n,M,p_len)
    omega = [floor((n-1)/(2*M))]
    m = floor(omega[1]/(2*M))
    if m>= p_len
        append!(omega,floor.(collect(range(1,stop=m,length=p_len-1))))
    else
        append!(omega,floor.(collect(range(0,stop=p_len-2))))
    end
    omega_temp = similar(omega)
    ys = []
    first_order = []
    total_order = []
    for i in 1:p_len
        omega_temp[i] = omega[1]
        omega_temp[[k for k in 1:p_len if k != i]] = omega[2:end]
        ys_p = typeof(f(rand(p_len)))[]
        for j in 1:n
            s = (2*pi/n) * (j-1)
            x = quantile.(Uniform(-pi,pi),0.5 .+ (1/pi) .*(asin.(sin.(omega_temp*s))))
            push!(ys_p,f(x))
        end
        ft = (fft.(ys_p))[2:Int(floor((n/2)))]
        push!(ys,[(abs.(ff)./n).^2 for ff in ft])
        varnce = 2*sum(ys[i])
        push!(first_order,2*sum(ys[i][(1:M)*Int(omega[1])])/varnce)
        push!(total_order,1 .- 2*sum(ys[i][1:Int(omega[1]/2)])/varnce)
    end

    return first_order, total_order
end