Base.@kwdef mutable struct Sobol <: GSAMethod 
    N::Int=1000
    order::Array{Int}=[0,1]
    nboot::Int=0
    conf_int::Float64=0.95
end

Base.@kwdef mutable struct SobolQuad <: GSAMethod 
    order::Array{Int}=[1]
    quadalg=HCubatureJL()
end

function give_rand_p!(p_range,p,p_fixed=nothing,indices=nothing)
    if p_fixed === nothing
        for j in 1:length(p_range)
            p[j] = (p_range[j][2] -p_range[j][1])*rand() + p_range[j][1] 
        end
    else
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
end

function calc_mean_var(f,p_range,N)
    p = Array{eltype(p_range[1])}(undef, length(p_range))
    give_rand_p!(p_range,p)
    y1 = f(p)
    y0 = zero(y1)
    v = zero(y1)
    for i in 1:N
        give_rand_p!(p_range,p)
        y1 = f(p)
        @. y0 += y1
        @. v += y1^2
    end
    @. y0 = y0/N
    @. v = v/(N-1) - (N*y0^2)/(N-1)
    y0,v
end

function calc_mean_var_quad(f,p_range,quadalg)
    prob = QuadratureProblem(f, [p_range[i][1] for i in 1:length(p_range)],[p_range[i][2] for i in 1:length(p_range)])
    E = solve(prob,quadalg)

    prob1 = QuadratureProblem((x,p) -> (f(x,p)).^2, [p_range[i][1] for i in 1:length(p_range)],[p_range[i][2] for i in 1:length(p_range)])
    V = solve(prob1,quadalg) .- E.^2
    return E, V
end

function first_order_var(f,p_range,N,y0,v,p1,p2,p3)
    ys = Array{typeof(y0)}(undef,length(p_range))
    for i in 1:length(p_range)
        y = zero(y0)
        indices = [k for k in 1:length(p_range) if k != i]
        i_arr = [i]
        for j in 1:N
            give_rand_p!(p_range,p2)
            give_rand_p!(p_range,p1,@view(p2[i:i]),i_arr)
            give_rand_p!(p_range,p3,@view(p1[indices]),indices)
            y .+= (f(p2)) .* (f(p1) .- f(p3))
        end
        ys[i] = y/N
    end
    for i in 1:length(ys)
        @. ys[i] = ys[i] / v
    end
    ys
end

function first_order_quad(f,p_range,y0,v,p1,quadalg)
    ys = Array{typeof(y0.u)}(undef,length(p_range))
    for i in 1:length(p_range)
        y = zero(y0)
        indices = [k for k in 1:length(p_range) if k != i]
         
        function f_q(x,p)
            p1[indices] = x
            f(p1,p)
        end
        prob1 = QuadratureProblem(f_q,[p_range[i][1] for i in indices],[p_range[i][2] for i in indices])
        
        function f__(x,p)
            p1[i] = x
            solve(prob1,quadalg).^2
        end
        prob2 = QuadratureProblem(f__, p_range[i][1], p_range[i][2])
        sol = solve(prob2,quadalg)
        y = @. sol - y0^2
        ys[i] = y
    end
    for i in 1:length(ys)
        @. ys[i] = ys[i] / v
    end
    ys
end

function second_order_var(f,p_range,N,y0,v,p1,p2,p3)
    ys = Array{typeof(y0)}(undef,Int((length(p_range)*(length(p_range)-1))/2))
    curr = 1
    for i in 1:length(p_range)
        for j in i+1:length(p_range)
            y = zero(y0)
            i_arr = [l for l in 1:length(p_range) if l != i]
            j_arr = [l for l in 1:length(p_range) if l != j]
            for k in 1:N
                give_rand_p!(p_range,p2)
                give_rand_p!(p_range,p1,@view(p2[i_arr]),i_arr)
                give_rand_p!(p_range,p3,@view(p2[j_arr]),j_arr)
                y .+= (f(p1) .- f(p3)).^2 
            end
            ys[curr] = y/(2*N)
            curr += 1
        end
    end
    for i in 1:length(ys)
        ys[i] = @. ys[i] / v
    end
    ys
end


function total_var(f,p_range,N,y0,v,p1,p2,p3)
    ys = Array{typeof(y0)}(undef,length(p_range))
    for i in 1:length(p_range)
        y = zero(y0)
        indices = [k for k in 1:length(p_range) if k != i]
        for j in 1:N
            give_rand_p!(p_range,p2)
            give_rand_p!(p_range,p1,@view(p2[indices]),indices)
            y .+= (f(p2) .- f(p1)).^2
        end
        ys[i] = y/(2*N)
    end
    for i in 1:length(ys)
        @. ys[i] = ys[i] / v
    end
    ys
end

function first_total_var(f,p_range,N,y0,v,p1,p2,p3)
    ys_first = Array{typeof(y0)}(undef,length(p_range))
    ys_tot = Array{typeof(y0)}(undef,length(p_range))
    for i in 1:length(p_range)
        y_first = zero(y0)
        y_tot = zero(y0)
        indices = [k for k in 1:length(p_range) if k != i]
        i_arr = [i]
        for j in 1:N
            give_rand_p!(p_range,p2)
            give_rand_p!(p_range,p1,@view(p2[i:i]),i_arr)
            give_rand_p!(p_range,p3,@view(p1[indices]),indices)
            f_p1 = f(p1)
            f_p3 = f(p3)
            y_first .+=  (f(p2)) .* (f_p1 .- f_p3)
            y_tot .+=  (f_p1 .- f_p3).^2
        end
        ys_first[i] = y_first/N
        ys_tot[i] = y_tot/(2*N)
    end
    for i in 1:length(ys_tot)
        @. ys_first[i] = ys_first[i] / v
        @. ys_tot[i] = ys_tot[i] / v
    end
    [ys_first,ys_tot]
end

mutable struct SobolResult{T1,T2}
    S1::T1
    S1_Conf_Int::T2
    S2::T1
    S2_Conf_Int::T2
    ST::T1
    ST_Conf_Int::T2
end

function calc_ci(f,p_range,N,y0,v,nboot,conf_int,sa_func,p1,p2,p3)
    conf_int_samples = [sa_func(f,p_range,N,y0,v,p1,p2,p3) for i in 1:nboot]
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

function gsa(f,p_range::AbstractVector,method::Sobol)
    @unpack N, order, nboot, conf_int = method
    y0,v = calc_mean_var(f,p_range,N)
    p2 = Array{eltype(p_range[1])}(undef, length(p_range))
    p1 = Array{eltype(p_range[1])}(undef, length(p_range))
    p3 = Array{eltype(p_range[1])}(undef, length(p_range))
    sobol_sens = SobolResult(Array{T where T}[],Array{Array{T where T},1}[],Array{T where T}[],Array{Array{T where T},1}[],Array{T where T}[],Array{Array{T where T},1}[])  
    if 0 in order && 1 in order
        first_total = first_total_var(f,p_range,N,y0,v,p1,p2,p3)
        sobol_sens.S1 = first_total[1]
        sobol_sens.ST = first_total[2]
        if nboot > 0
            ci = calc_ci(f,p_range,N,y0,v,nboot,conf_int,first_total_var,p1,p2,p3)
            sobol_sens.S1_Conf_Int = [vec.(first_total[1]) - ci[1], vec.(first_total[1]) + ci[1]]
            sobol_sens.ST_Conf_Int = [vec.(first_total[2]) - ci[2], vec.(first_total[2]) + ci[2]]        
        end
    elseif 1 in order
        first_order = first_order_var(f,p_range,N,y0,v,p1,p2,p3)
        sobol_sens.S1 = first_order
        if nboot > 0
            ci = calc_ci(f,p_range,N,y0,v,nboot,conf_int,first_order_var,p1,p2,p3)
            sobol_sens.S1_Conf_Int = [vec.(first_order) - ci, vec.(first_order) + ci]
        end
    elseif 0 in order
        total_indices = total_var(f,p_range,N,y0,v,p1,p2,p3)
        sobol_sens.ST = total_indices
        if nboot > 0
            ci = calc_ci(f,p_range,N,y0,v,nboot,conf_int,total_var)
            sobol_sens.ST_Conf_Int = [vec.(total_indices) - ci, vec.(total_indices) + ci]
        end
    end
    if 2 in order
        second_order = second_order_var(f,p_range,N,y0,v,p1,p2,p3)
        sobol_sens.S2 = second_order
        if nboot > 0
            ci = calc_ci(f,p_range,N,y0,v,nboot,conf_int,second_order_var,p1,p2,p3)
            sobol_sens.S2_Conf_Int = [vec.(second_order) - ci, vec.(second_order) + ci]
        end
    end
    sobol_sens
end

function gsa(f,p_range::AbstractVector,method::SobolQuad)
    @unpack order, quadalg = method
    p1 = Array{eltype(p_range[1])}(undef, length(p_range))
    E,V = calc_mean_var_quad(f,p_range,quadalg)
    sobol_sens = SobolResult(Array{T where T}[],Array{Array{T where T},1}[],Array{T where T}[],Array{Array{T where T},1}[],Array{T where T}[],Array{Array{T where T},1}[])
    if 1 in order
        first_order = first_order_quad(f,p_range,E,V,p1,quadalg)
        sobol_sens.S1 = first_order
    end
    sobol_sens
end

function gsa(prob::DiffEqBase.DEProblem,alg::DiffEqBase.DEAlgorithm,t,p_range::AbstractVector,method::Union{Sobol,SobolQuad})
    f = function (p)
        prob1 = remake(prob;p=p)
        Array(solve(prob1,alg;saveat=t))
    end
    @assert length(prob.p) == length(p_range)
    gsa(f,p_range,method)
end
