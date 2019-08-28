function give_rand_p(p_range,p_fixed=nothing,indices=nothing)
    if p_fixed === nothing
        p = [(p_range[j][2] -p_range[j][1])*rand() + p_range[j][1] for j in 1:length(p_range)]
    else
        p =  zeros(length(p_range))
        j = 1
        for i in 1:length(p_range)
            if i in indices
                p[i] = p_fixed[j]
                j += 1
            else
                p[i] = (p_range[i][2] -p_range[i][1])*rand() + p_range[i][1]
            end
        end
    end
    p
end

function calc_mean_var(f,p_range,N)
    y1 = Array(f(give_rand_p(p_range)))
    y0 = zero(y1)
    v = zero(y1)
    for i in 1:N
        y1 = Array(f(give_rand_p(p_range)))
        @. y0 += y1
        @. v += y1^2
    end
    y0 = @. y0/N
    y0_sq = [i.^2 for i in y0]
    v = @. v/(N-1) - (N*y0_sq)/(N-1)
    y0,v
end

function first_order_var(f,p_range,N,y0,v)
    ys = Array{typeof(y0)}(undef,length(p_range))
    for i in 1:length(p_range)
        y = zero(y0)
        for j in 1:N
            p2 = give_rand_p(p_range)
            p1 = give_rand_p(p_range,[p2[i]],[i])
            p3 = give_rand_p(p_range,p1[1:end .!= i],[k for k in 1:length(p_range) if k != i])
            yer =  Array(f(p2)) .* (Array(f(p1)) .- Array(f(p3)))
            @. y += yer
        end
        ys[i] = copy(y/N)
    end
    for i in 1:length(ys)
        ys[i] = @. ys[i] / v
    end
    ys
end

function second_order_var(f,p_range,N,y0,v)
    ys = Array{typeof(y0)}(undef,Int((length(p_range)*(length(p_range)-1))/2))
    curr = 1
    for i in 1:length(p_range)
        for j in i+1:length(p_range)
            y = zero(y0)
            for k in 1:N
                p2 = give_rand_p(p_range)
                p1 = give_rand_p(p_range,p2[1:end .!= i],[l for l in 1:length(p_range) if l != i])
                p3 = give_rand_p(p_range,p2[1:end .!= j],[l for l in 1:length(p_range) if l != j])
                yer =  (Array(f(p1)) .- Array(f(p3))).^2
                @. y += yer
            end
            ys[curr] = copy(y/(2*N))
            curr += 1
        end
    end
    # ys_frst_order = first_order_var(f,p_range,N,y0,v)
    # j = 1
    # for i in 1:length(p_range)
    #     for k in i+1:length(p_range)
    #         ys[j] = @. ys[j] - ( ys_frst_order[i] + ys_frst_order[k] )
    #         j += 1
    #     end
    # end
    for i in 1:length(ys)
        ys[i] = @. ys[i] / v
    end
    ys
end


function total_var(f,p_range,N,y0,v)
    ys = Array{typeof(y0)}(undef,length(p_range))
    for i in 1:length(p_range)
        y = zero(y0)
        for j in 1:N
            p2 = give_rand_p(p_range)
            p1 = give_rand_p(p_range,p2[1:end .!= i],[k for k in 1:length(p_range) if k != i])
            yer =  (Array(f(p2)) .- Array(f(p1))).^2
            @. y += yer
        end
        ys[i] = copy(y/(2*N))
    end
    for i in 1:length(ys)
        ys[i] = @. ys[i] / v
    end
    ys
end

mutable struct SobolResult{T1,T2}
    S1::T1
    S1_Conf_Int::T2
    S2::T1
    S2_Conf_Int::T2
    ST::T1
    ST_Conf_Int::T2
end

function calc_ci(f,p_range,N,y0,v,nboot,conf_int,sa_func)
    conf_int_samples = [sa_func(f,p_range,N,y0,v) for i in 1:nboot]
    elems_ = []
    for i in 1:length(conf_int_samples[1])
        elems = []
        for k in 1:length(conf_int_samples[1][1])
            elem = eltype(conf_int_samples[1][1])[]
            for j in 1:length(conf_int_samples)
                push!(elem,conf_int_samples[j][i][k])
            end
            push!(elems,elem)
        end
        push!(elems_,elems)
    end
    z = -quantile(Normal(), (1-conf_int)/2)
    S1_Conf_Int = [[z*std(sample) for sample in el] for el in elems_]
end

function gsa(f,p_range::AbstractVector,method::Sobol,N::Int64,order=[0],nboot=100,conf_int=0.95)
    y0,v = calc_mean_var(f,p_range,N)
    sobol_sens = SobolResult(Array{Float64}[],Array{Array{Float64},1}[],Array{Float64}[],Array{Array{Float64},1}[],Array{Float64}[],Array{Array{Float64},1}[])
    if 1 in order
        first_order = first_order_var(f,p_range,N,y0,v)
        sobol_sens.S1 = first_order
        if nboot > 0
            ci = calc_ci(f,p_range,N,y0,v,nboot,conf_int,first_order_var)
            sobol_sens.S1_Conf_Int = [vec.(first_order) - ci, vec.(first_order) + ci]
        end
    end
    if 2 in order
        second_order = second_order_var(f,p_range,N,y0,v)
        sobol_sens.S2 = second_order
        if nboot > 0
            ci = calc_ci(f,p_range,N,y0,v,nboot,conf_int,second_order_var)
            sobol_sens.S2_Conf_Int = [vec.(second_order) - ci, vec.(second_order) + ci]
        end
    end
    if 0 in order
        total_indices = total_var(f,p_range,N,y0,v)
        sobol_sens.ST = total_indices
        if nboot > 0
            ci = calc_ci(f,p_range,N,y0,v,nboot,conf_int,total_var)
            sobol_sens.ST_Conf_Int = [vec.(total_indices) - ci, vec.(total_indices) + ci]
        end
    end
    sobol_sens
end

function gsa(prob::DiffEqBase.DEProblem,alg::DiffEqBase.DEAlgorithm,t,p_range::AbstractVector,method::Sobol,N::Int64,order=[0])
    f = function (p)
        prob1 = remake(prob;p=p)
        Array(solve(prob1,alg;saveat=t))
    end
    @assert length(prob.p) == length(p_range)
    sobol_sensitivity(f,p_range,N,order)
end
