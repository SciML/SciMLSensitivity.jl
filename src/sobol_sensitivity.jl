function calc_mean_var(prob::DEProblem,alg,t,p_range,N)
    f = function (p)
        prob1 = remake(prob;p=p)
        Array(solve(prob1,alg;saveat=t))
    end

    calc_mean_var(f,p_range,p_steps,N)
end

function give_rand_y(f,p_range,p_fixed=nothing,indices=nothing)
    if p_fixed == nothing
        p = [rand(p_range[j][1]:p_range[j][2]) for j in 1:length(p_range)]
    else
        p = [0.0 for i in p_range]
        j = 1
        for i in 1:length(p_range)
            if i in indices
                p[i] = p_fixed[j]
                j += 1
            else
                p[i] = rand(p_range[i][1]:p_range[i][2])
            end
        end
    end
    y1 = f(p)
    Array(y1)
end

function calc_mean_var(f,p_range,N)
    y1 = give_rand_y(f,p_range)
    y0 = [0.0 for j in 1:length(y1)]
    v = [0.0 for j in 1:length(y1)]
    if length(size(y1)) != 1
        y0 = reshape(y0,size(y1)[1],size(y1)[2])
        v = reshape(v,size(y1)[1],size(y1)[2])
    end
    for i in 1:N
        y1 = give_rand_y(f,p_range)
        y0 .+= y1
        y1 = give_rand_y(f,p_range)
        v .+= y1.^2
    end
    y0 = y0./N
    y0_sq = [i.^2 for i in y0]
    v = v./N .- y0_sq
    y0,v
end

function first_order_var(f,p_range,N,y0,p_fixed)
    ys = []
    for i in 1:length(p_range)
        y = [0.0 for i in 1:length(y0)]
        for j in 1:N
            y .+= give_rand_y(f,p_range,[p_fixed[i]],[i]).^2 
        end
        y = y./N - y0.^2
        push!(ys,y)
    end
    ys
end

function sobol_sensitivity(f,p_range,N,p_fixed)
    y0,v = calc_mean_var(f,p_range,N)
    first_order = first_order_var(f,p_range,N,y0,p_fixed)
    first_order
end

