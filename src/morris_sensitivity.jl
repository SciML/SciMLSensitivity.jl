struct MatSpread
    mat::Vector{Vector{Float64}}
    spread::Float64
end

struct MorrisSensitivity
    means
    variances
    elementary_effects
end

function generate_design_matrix(p_range,p_steps;k = 10)
    ps = [range(p_range[i][1], stop=p_range[i][2], length=p_steps[i]) for i in 1:length(p_range)]
    indices = [rand(1:i) for i in p_steps]
    all_idxs = Vector{typeof(indices)}(undef,k)

    for i in 1:k
        flag = 0
        j = rand(1:length(p_range))
        indices[j] += (rand() < 0.5 ? -1 : 1)
        if indices[j] > p_steps[j]
            indices[j] -= 2
        elseif indices[j] < 1.0
            indices[j] += 2
        end
        all_idxs[i] = copy(indices)
    end

    B = Array{Array{Float64}}(undef,k)
    for j in 1:k
        cur_p = [ps[u][(all_idxs[j][u])] for u in 1:length(p_range)]
        B[j] = cur_p
    end
    B
end

function calculate_spread(matrix)
    spread = 0.0
    for i in 2:length(matrix)
        spread += sqrt(sum(abs2.(matrix[i] - matrix[i-1])))
    end
    spread
end

function sample_matrices(p_range,p_steps;len_trajectory=10,num_trajectory=10,total_num_trajectory=5*num_trajectory)
    matrix_array = []
    if total_num_trajectory<num_trajectory
        error("total_num_trajectory should be greater than num_trajectory preferably atleast 3-4 times higher")
    end
    for i in 1:total_num_trajectory
        mat = generate_design_matrix(p_range,p_steps;k=len_trajectory)
        spread = calculate_spread(mat)
        push!(matrix_array,MatSpread(mat,spread))
    end
    sort!(matrix_array,by = x -> x.spread,rev=true)
    matrices = [i.mat for i in matrix_array[1:num_trajectory]]
    matrices
end

function morris_sensitivity(prob::DiffEqBase.DEProblem,alg,t,p_range,p_steps;kwargs...)
    f = function (p)
      prob1 = remake(prob;p=p)
      Array(solve(prob1,alg;saveat=t))
    end
    morris_sensitivity(f,p_range,p_steps;kwargs...)
end

function morris_sensitivity(f,p_range,p_steps;relative_scale=false,kwargs...)
    design_matrices = sample_matrices(p_range,p_steps;kwargs...)
    effects = []
    for i in design_matrices
        y1 = f(i[1])
        for j in 1:length(i)-1
            y2 = y1
            del = i[j+1] - i[j]
            change_index = 0
            for k in 1:length(del)
                if abs(del[k]) > 0
                    change_index = k
                    break
                end
            end
            del = sum(del)
            y1 = f(i[j+1])
            if relative_scale == false
                elem_effect = @. abs((y1-y2)/(del))
            else
                if del > 0
                    elem_effect = @. abs((y1-y2)/(y2*del))
                else
                    elem_effect = @. abs((y1-y2)/(y1*del))
                end
            end
            if length(effects) >= change_index && change_index > 0 
                push!(effects[change_index],elem_effect)
            elseif change_index > 0
                while(length(effects) < change_index-1)
                    push!(effects,[])
                end
                push!(effects,[elem_effect])
            end
        end
    end
    means = []
    variances = []
    for k in effects
        push!(means,mean(k))
        push!(variances,var(k))
    end
    MorrisSensitivity(means,variances,effects)
end
