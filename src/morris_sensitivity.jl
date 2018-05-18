function generate_design_matrix(prob::DEProblem,param_bounds,p)
    init_param_vector = prob.p
    B = Array{Array{Float64}}(length(prob.p)+1)
    r = Float64[]
    for i in 1:length(param_bounds)
        dels = [j/p for j in param_bounds[i][1]:param_bounds[i][2]]
        del_i = rand(dels)
        push!(r,init_param_vector[i]+del_i)
        # println(r)
    end

    for i in 1:(length(prob.p)+1)
        j = rand(1:length(prob.p))
        s = rand([-1,1])
        r[j] = r[j]+ s/p
        t = [i for i in r]
        B[i] = t
    end
    B
end

f = @ode_def LotkaVolterra begin
    dx = a*x - b*x*y
    dy = -c*y + x*y
end a b c
  
p = [1.5,1.0,3.0]
prob = ODEProblem(f,[1.0;1.0],(0.0,10.0),p)
t =generate_design_matrix(prob,[[0.0,5.0],[0.0,5.0],[1.0,5.0]],10.0)