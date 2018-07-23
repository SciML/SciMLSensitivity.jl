function give_rand_p(p_range,p_fixed=nothing,indices=nothing)
    if p_fixed == nothing
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
    v = @. v/N - y0_sq
    y0,v
end

function first_order_var(f,p_range,N,y0)
    ys = Array{typeof(y0)}(undef,length(p_range))
    for i in 1:length(p_range)
        y = zero(y0)
        for j in 1:N
            p2 = give_rand_p(p_range)
            p1 = give_rand_p(p_range,[p2[i]],[i])
            yer =  Array(f(p1)) .* Array(f(p2))
            @. y += yer
        end
        y = @. y/N - y0^2
        ys[i] = copy(y)
    end
    ys
end

function second_order_var(f,p_range,N,y0)
    ys = Array{typeof(y0)}(undef,Int((length(p_range)*(length(p_range)-1))/2))
    curr = 1
    for i in 1:length(p_range)
        for j in i+1:length(p_range)
            y = zero(y0)
            for k in 1:N
                p2 = give_rand_p(p_range)
                p1 = give_rand_p(p_range,[p2[i],p2[j]],[i,j])
                y .+=  Array(f(p1)) .* Array(f(p2))
            end
            y = @. y/N - y0^2
            ys[curr] = copy(y)
            curr += 1
        end
    end
    ys_frst_order = first_order_var(f,p_range,N,y0)
    j = 1
    for i in 1:length(p_range)
        for k in i+1:length(p_range)
            ys[j] = @. ys[j] - ( ys_frst_order[i] + ys_frst_order[k] )
            j += 1
        end
    end
    ys
end


function total_var(f,p_range,N,y0)
    ys = Array{typeof(y0)}(undef,length(p_range))
    for i in 1:length(p_range)
        y = zero(y0)
        for j in 1:N
            p_fixed_all = []
            p_fixed_indices = []
            p2 = give_rand_p(p_range)
            for j in 1:length(p2)
                if j != i
                    push!(p_fixed_all,p2[j])
                    push!(p_fixed_indices,j)
                end
            end
            p1 = give_rand_p(p_range,p_fixed_all,p_fixed_indices)
            yer =  Array(f(p1)) .* Array(f(p2))
            @. y += yer
        end
        y = @. y/N - y0^2
        ys[i] = copy(y)
    end
    ys
end

function sobol_sensitivity(f,p_range,N,order=2)
    y0,v = calc_mean_var(f,p_range,N)
    if order == 1
        first_order = first_order_var(f,p_range,N,y0)
        for i in 1:length(first_order)
            first_order[i] = @. first_order[i] / v
        end
        first_order
    elseif order == 2
        second_order = second_order_var(f,p_range,N,y0)
        for i in 1:length(second_order)
            second_order[i] = @. second_order[i] / v
        end
        second_order
    else
        total_indices = total_var(f,p_range,N,y0)
        for i in 1:length(total_indices)
            total_indices[i] = @. 1 - (total_indices[i] / v)
        end
        total_indices
    end
end

function sobol_sensitivity(prob::DiffEqBase.DEProblem,alg,t,p_range,N,order=2)
    f = function (p)
        prob1 = remake(prob;p=p)
        Array(solve(prob1,alg;saveat=t))
    end
    @assert length(prob.p) == length(p_range)
    sobol_sensitivity(f,p_range,N,order)
end
