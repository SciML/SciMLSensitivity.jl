using RecursiveArrayTools, DataFrames, GLM
struct Regression_Sensitivity_Coefficients
    Pearson
    Standard_Regression
    Partial_Correlation
    Spearman
    Standard_Rank_Regression
    Partial_Rank_Correlation
end

function regression_sensitivity(f,p_range,p_fixed,n;coeffs=:rank)

    pearson_coeffs = []
    src_coeffs = []
    partial_coeffs = []
    spearman_coeffs = []
    srr_coeffs = []
    partial_rank_coeffs = []

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
        x_vrs = [sum(xis[i] - x_mean) for o in 1:length(xis)]
        x_lm = [j[i] for j in xis]
        x_rnk = sortperm(x_lm)
        x_rnk_mean = mean(x_rnk)
        x_rnk_vrs = [x_rnk[i] - x_rnk_mean for o in 1:length(x_rnk)]
        x_var = var(x_lm)
        x_rnk_var = var(x_rnk)
        pcs,srcs = pcs_and_srcs(yis,x_lm,x_vrs,x_var)
        pcc = pcc_f(x_lm,yis)
        yis_rnk = zero(yis)
        for k in 1:size(yis)[1]
            for j in 1:size(yis)[2]
                yis_rnk[k,j,:] = sortperm(yis[k,j,:])
            end
        end
        if coeffs == :rank
            prcc = pcc_f(x_rnk,yis_rnk)
            spear_rcc, srrc = pcs_and_srcs(yis_rnk,x_rnk,x_rnk_vrs,x_rnk_var)
        end
        push!(pearson_coeffs,pcs)
        push!(src_coeffs,srcs)
        push!(partial_coeffs,pcc)
        if coeffs == :rank
            push!(partial_rank_coeffs,prcc)
            push!(spearman_coeffs,spear_rcc)
            push!(srr_coeffs,srrc)
        end
    end
    if coeffs == :rank
        regre_coeff = Regression_Sensitivity_Coefficients(pearson_coeffs,src_coeffs,partial_coeffs,spearman_coeffs,partial_rank_coeffs,srr_coeffs)
    else
        regre_coeff = Regression_Sensitivity_Coefficients(pearson_coeffs,src_coeffs,partial_coeffs,nothing,nothing,nothing)
    end
    regre_coeff
end

function pcs_and_srcs(yis,x_lm,x_vrs,x_var)
    pcs = []
    srcs = []
    for k in 1:size(yis)[1]
        pc = []
        src = []
        for j in 2:size(yis)[2]
            pear_coeff_num = sum(x_vrs .* (yis[k,j,:] .- mean(yis[k,j,:])))
            pear_coeff_deno = sqrt(sum(x_vrs.^2)) * sqrt(sum((yis[k,j,:] .- mean(yis[k,j,:])).^2))
            pear_coeff = pear_coeff_num/pear_coeff_deno
            push!(pc,pear_coeff)
            df = DataFrame(X=x_lm,Y=yis[k,j,:])
            lin_model = lm(@formula(Y ~ X),df)
            push!(src,coef(lin_model)[2].*sqrt.(x_var ./ var(yis[k,j,:])))
        end
        push!(pcs,pc)
        push!(srcs,src)
    end
    pcs,srcs
end

function pcc_f(x_lm,yis)
    df_arr = Array{Float64}[Float64[] for i in 1:length(x_lm)]
    for o in 1:length(x_lm)
        for j in 1:length(x_lm)
            push!(df_arr[o],x_lm[(j+o)%length(x_lm)+1])
        end
    end
    df = DataFrame(df_arr)
    last_ind = length(x_lm)
    formula = "@formula(x$last_ind ~ "
    for j in 1:length(x_lm)-1
        formula *= "x$j +"
    end
    formula = formula[1:end-1] * ")"
    ols_x = lm(eval(Meta.parse(formula)),df)
    x_cap = x_lm .- predict(ols_x)
    y_caps = []
    for k in 1:size(yis)[1]
        y_cps = []
        for j in 2:size(yis)[2]
            df_arr = Array{Float64}[Float64[] for i in 1:length(yis[k,j,:])]
            for o in 1:length(yis[k,j,:])
                for l in 1:length(yis[k,j,:])
                    push!(df_arr[o],yis[k,j,:][(l+o)%length(yis[k,j,:])+1])
                end
            end
            df = DataFrame(df_arr)
            last_ind = length(yis[k,j,:])
            formula = "@formula(x$last_ind ~ "
            for j in 1:length(yis[k,j,:])-1
                formula *= "x$j +"
            end
            formula = formula[1:end-1] * ")"
            ols_y = lm(eval(Meta.parse(formula)),df)
            y_cap = yis[k,j,:] .- predict(ols_y)
            push!(y_cps,y_cap)
        end
        push!(y_caps,y_cps)
    end
    pcc = []
    for k in 1:length(y_caps)
        pcc_ = []
        for j in 1:length(y_caps[k])
            num = sum(x_cap .* y_caps[k][j])
            deno = sqrt(sum(x_cap.^2)*sum(y_caps[k][j].^2))
            push!(pcc_,num/deno)
        end
        push!(pcc,pcc_)
    end
    pcc
end

function regression_sensitivity(prob::DiffEqBase.DEProblem,alg,t,p_range,p_fixed,n;coeffs=:rank)
    f = function (p)
        prob1 = remake(prob;p=p)
        Array(solve(prob1,alg;saveat=t))
    end
    regression_sensitivity(f,p_range,p_fixed,n,coeffs=coeffs)
end
