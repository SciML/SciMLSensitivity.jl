function calc_mean_var(prob::DEProblem,alg,t,p_range,p_steps,N)
    f = function (p)
        prob1 = remake(prob;p=p)
        Array(solve(prob1,alg;saveat=t))
    end

    calc_mean_var(f,p_range,p_steps,N)
end

function give_rand_y(f,p_range)
    p = [rand(p_range[j][1]:p_range[j][2]) for j in 1:length(p_range)]
    y1 = f(p)
    Array(y1)
end

function calc_mean_var(f,p_range,p_steps,N)
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

