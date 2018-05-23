struct MatSpread
    mat::Array{Array{Float64}}
    spread::Float64
end

function generate_design_matrix(prob::DEProblem,p_range,p_steps;k = 10)
    ps = [linspace(p_range[i][1],p_range[i][2],p_steps[i]) for i in 1:length(p_range)]
    indices = [rand(1:i) for i in p_steps]
    all_idxs = Vector{typeof(indices)}(k)
    i = 1
    while i <= k
        flag = 0
        j = rand(1:length(p_range))
        indices[j] += (rand() < 0.5 ? -1 : 1)
        if indices[j] > p_steps[j] 
            flag = 1
            indices[j] -=1 
        elseif indices[j] < 1.0
            flag = 1
            indices[j] =1
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

function calculate_spread(matrix)
    spread = 0.0
    for i in 2:length(matrix)
        spread += sqrt(sum(abs2.(matrix[i] - matrix[i-1])))
    end
    spread
end

function sample_matrices(prob::DEProblem,p_range,p_steps;k=10,simulations=50,r=10)
    matrix_array = []
    for i in 1:simulations
        mat = generate_design_matrix(prob,p_range,p_steps;k=4)
        spread = calculate_spread(mat)
        push!(matrix_array,MatSpread(mat,spread))
    end
    sort!(matrix_array,by = x -> x.spread,rev=true)
    matrices = [i.mat for i in matrix_array[1:r]]
    matrices
end