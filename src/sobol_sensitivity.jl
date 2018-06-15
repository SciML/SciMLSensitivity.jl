function calc_mean_var(prob::DEProblem,alg,t,p_range,p_steps,N)
    f = function (p)
        prob1 = remake(prob;p=p)
        Array(solve(prob1,alg;saveat=t))
    end

    calc_mean_var(f,p_range,p_steps,N)
end

function calc_mean_var(f,p_range,p_steps,N)
    p_values = [collect(linspace(p_range[i][1],p_range[i][2],p_steps[i])) for i in 1:length(p_range)]
    p = [rand(p_values[j]) for j in 1:length(p_values)]
    y1 = f(p)
    if length(size(y1)) != 1
        y0 = [[0.0 for j in 1:size(y1)[1]] for i in 1:size(y1)[2]]
        v = [[0.0 for j in 1:size(y1)[1]] for i in 1:size(y1)[2]]
    else
        y0 = [0.0 for j in 1:size(y1)[1]]
        v = [0.0 for j in 1:size(y1)[1]]
    end
    for i in 1:N
        p = [rand(p_values[j]) for j in 1:length(p_values)]
        y1 = f(p)
        if length(size(y1)) != 1
            for j in 1:size(y1)[2]
                y0[j] .+= y1[j]
                v[j] .+= (y1[j]).^2
            end
        else
            y0 .+= y1
            v .+= y1.^2
        end
    end
    y0 = y0./N
    y0_sq = [i.^2 for i in y0]
    v = v./N .- y0_sq
    y0,v
end

