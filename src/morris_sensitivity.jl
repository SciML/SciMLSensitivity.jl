function generate_design_matrix(prob::DEProblem,p_range,p_steps,k = 10)
    ps = [linspace(p_range[i][1],p_range[i][2],p_steps[i]) for i in 1:length(p_range)]
    indices = [rand(1:i) for i in p_steps]
    all_idxs = Vector{typeof(indices)}(k)
    all_idxs[1] = copy(indices)
    i = 2
    while i <= k
        flag = 0
        indices[rand(1:length(p_range))] += (rand() < 0.5 ? -1 : 1)
        for j in indices
            if j > k || j < 1.0
                flag = 1
            end
        end
        if flag == 0
            all_idxs[i] = copy(indices)
            i = i+1
        end
    end

    B = Array{Array{Float64}}(k)
    for j in 1:k
        cur_p = [ps[u][(all_idxs[j][u])] for u in 1:length(p_range)]
        B[j] = cur_p
    end
    B
end