using DiffEqSensitivity, QuasiMonteCarlo, Test

function linear_batch(X)
    A= 7
    B= 0.1
    @. A*X[1,:]+B*X[2,:]
end
function linear(X)
    A= 7
    B= 0.1
    A*X[1]+B*X[2]
end

p_range = [[-1, 1], [-1, 1]]
reg = gsa(linear_batch, Regression(), p_range; batch = true)

reg = gsa(linear, Regression(), p_range; batch = false)
reg = gsa(linear, Regression(true), p_range; batch = false)

f1m(x) = [x[1], -x[1]]
fn1(x) = 7x[1]
fnm(x) = [x[1], -x[2]]

@testset "f: R -> R^m" begin
    reg = gsa(f1m, Regression(rank = true), p_range; batch = false)
    @test reg.pearson[1, 1] ≈ 1
    @test reg.standard_regression[1, 1] ≈ 1
    @test reg.partial_correlation[1, 1] ≈ -1
    @test reg.pearson_rank[1, 1] ≈ 1
    @test reg.standard_rank_regression[1, 1] ≈ 1
    @test reg.partial_rank_correlation[1, 1] ≈ -1

    # loose tolerances, exact in limit
    @test reg.pearson[1, 2] ≈ -1
    @test reg.standard_regression[1, 2] ≈ -1
    @test reg.partial_correlation[1, 2] ≈ 1
    @test reg.standard_rank_regression[2, 1] ≈ 0 atol=1e-2
end

@testset "f: R^n -> R" begin
    reg = gsa(fn1, Regression(rank = true), p_range; batch = false)
    @test reg.pearson[1, 1] ≈ 1
    @test reg.standard_regression[1, 1] ≈ 1
    @test reg.partial_correlation[1, 1] ≈ -1
    @test reg.pearson_rank[1, 1] ≈ 1
    @test reg.standard_rank_regression[1, 1] ≈ 1
    @test reg.partial_rank_correlation[1, 1] ≈ -1

    # loose tolerances, exact in limit
    @test reg.pearson[2, 1] ≈ 0 atol=1e-2
    @test reg.standard_regression[2, 1] ≈ 0 atol=1e-2
    @test reg.partial_correlation[2, 1] ≈ 0 atol=1e-2
    @test reg.standard_rank_regression[2, 1] ≈ 0 atol=1e-2
end

@testset "f: R^n -> R^m" begin
    reg = gsa(fnm, Regression(rank = true), p_range; batch = false)

    @test reg.pearson[2, 2] ≈ -1
    @test reg.standard_regression[2, 2] ≈ -1
    @test reg.partial_correlation[2, 2] ≈ 1

    # loose tolerances, exact in limit
    @test reg.pearson[2, 1] ≈ 0 atol=1e-2
    @test reg.standard_regression[2, 1] ≈ 0 atol=1e-2
    @test reg.partial_correlation[2, 1] ≈ 0 atol=1e-2
    @test reg.pearson[1, 2] ≈ 0 atol=1e-2
    @test reg.standard_regression[1, 2] ≈ 0 atol=1e-2
    @test reg.partial_correlation[1, 2] ≈ 0 atol=1e-2
end
