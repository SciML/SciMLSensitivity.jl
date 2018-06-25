using RecursiveArrayTools, DataFrames, GLM
function regression_sensitivity(f,p_range,p_fixed,n)
    pearson_coeffs = []
    src_coeffs = []
    for i in 1:length(p_range)
        pcs = []
        srcs = []
        params = []
        for j in 1:length(p_range)
            if i != j
                push!(params,p_fixed[j])
            else
                push!(params,0)
            end
        end
        xis = []
        yis = Array{Float64}[]
        for j in 1:n
            params[i] = (p_range[i][2]-p_range[i][1])*rand() + p_range[i][1]
            x = params
            y = f(params)
            push!(xis,copy(x))
            push!(yis,copy(y))
        end
        yis = VectorOfArray(yis)
        x_mean = mean(xis)
        x_vrs = [sum(xis[i] - x_mean) for i in 1:length(xis)]
        x_lm = [j[i] for j in xis]
        x_var = var(x_lm)
        for k in 1:size(yis)[1]
            pc = []
            src = []
            for j in 2:size(yis)[2]
                pear_coeff_num = sum(x_vrs .* (yis[k,j,:] - mean(yis[k,j,:])))
                pear_coeff_deno = sqrt(sum(x_vrs.^2)) * sqrt(sum((yis[k,j,:] - mean(yis[k,j,:]).^2)))
                pear_coeff = pear_coeff_num/pear_coeff_deno
                push!(pc,pear_coeff)
                df = DataFrame(X=x_lm,Y=yis[k,j,:])
                lin_model = lm(@formula(Y ~ X),df)
                push!(src,coef(lin_model)[2].*sqrt.(x_var ./ var(yis[k,j,:])))
            end
            push!(pcs,pc)
            push!(srcs,src)
        end
        push!(pearson_coeffs,pcs)
        push!(src_coeffs,srcs)
    end
    src_coeffs
end