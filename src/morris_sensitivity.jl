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
    ps = [linspace(p_range[i][1],p_range[i][2],p_steps[i]) for i in 1:length(p_range)]
    indices = [rand(1:i) for i in p_steps]
    all_idxs = Vector{typeof(indices)}(k)

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

    B = Array{Array{Float64}}(k)
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

function sample_matrices(p_range,p_steps;k=10,simulations=50,r=10)
    matrix_array = []
    for i in 1:simulations
        mat = generate_design_matrix(p_range,p_steps;k=k)
        spread = calculate_spread(mat)
        push!(matrix_array,MatSpread(mat,spread))
    end
    sort!(matrix_array,by = x -> x.spread,rev=true)
    matrices = [i.mat for i in matrix_array[1:r]]
    matrices
end

function morris_sensitivity(prob::DEProblem,alg,t,p_range,p_steps;kwargs...)
    f = function (p)
      prob1 = remake(prob;p=p)
      y1 = solve(prob1,alg;saveat=t)
    end
    morris_sensitivity(f,p_range,p_steps;kwargs...)
end

function morris_sensitivity(f,p_range,p_steps;kwargs...)
    design_matrices = sample_matrices(p_range,p_steps;kwargs...)
    y1 = Array(f(design_matrices[1][1]))
    effects = [typeof(y1)[] for i in 1:length(p_range)]
    for i in design_matrices
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
            elem_effect = (y1-y2)/del
            push!(effects[change_index],elem_effect)
        end
    end
    effects
    means = eltype(effects[1])[]
    variances = eltype(effects[1])[]
    for k in 1:length(effects)
      sense_series = [effects[k][i][j] for i in 1:length(effects[k]), j in 1:length(effects[k][1])]
      push!(means,reshape(mean(sense_series,1),size(effects[k][1])))
      push!(variances,reshape(var(sense_series,1),size(effects[k][1])))
    end
    MorrisSensitivity(means,variances,effects)
end
