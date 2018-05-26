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

function MorrisGlobalSensitivity(prob::DEProblem,alg,t;kwargs...)
    sensitivity_function= function(p_range,p_steps;kwargs...)
        design_matrices = sample_matrices(p_range,p_steps;kwargs...)
        effects = [Array{Float64}[] for i in 1:length(prob.p)]
        prob2 = remake(prob;p=design_matrices[1][1])
        y1 = solve(prob2,alg;saveat=t)
        for i in design_matrices
            for j in 1:length(i)-1
                y2 = y1
                prob1 = remake(prob;p=i[j+1]) 
                del = i[j+1] - i[j]
                change_index = 0
                for k in 1:length(del)
                    if abs(del[k]) > 0
                        change_index = k
                        break
                    end
                end
                del = sum(del)
                y1 = solve(prob1,alg;saveat=t)
                elem_effect = (y1-y2)/del
                push!(effects[change_index],elem_effect)
            end
        end
        means = []
        variances = [[] for j in 1:length(prob.p)]
        for i in effects
            push!(means,mean(i))
        end
        for i in 1:length(effects)
            u = VectorOfArray(effects[i])
            vars = [[] for i in 1:length(prob.u0)]
            for j in 1:length(prob.u0)
                for k in 1:length(t)
                    push!(vars[j],var(u[j,k,:]))
                end
            end
            push!(variances[i],vars)
        end
        variances = VectorOfArray(variances)'
        MorrisSensitivity(means,variances,effects)
    end
end
