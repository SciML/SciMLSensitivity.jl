"""
    Regression(rank) <: GSAMethod

Regression methods for global sensitivity analysis. Providing this to `gsa` results
in a calculation of the following statistics, provided as a `RegressionResult`. If
the function f to be analyzed is of dimensionality f: R^n -> R^m, then these coefficients
are returned as a matrix, with the corresponding statistic in the (i, j) entry.

- `pearson`: This is equivalent to the correlation coefficient matrix between input and output
- `standard_regression`: Standard regression coefficients, also known as sigma-normalized
derivatives
- `partial_correlation`: Partial correlation coefficients, related to the precision matrix
and a measure of the correlation of linear models of the

# Arguments
- `rank::Bool = false`: Flag determining whether to also run a rank regression analysis
"""
@with_kw mutable struct Regression <: GSAMethod
    rank::Bool = false
end

struct RegressionResult{T, TR}
    pearson::T
    standard_regression::T
    partial_correlation::T
    pearson_rank::TR
    standard_rank_regression::TR
    partial_rank_correlation::TR
end

function gsa(f, method::Regression, p_range::AbstractVector, samples::Int=1000, batch::Bool = false)
    lb = [i[1] for i in p_range]
    ub = [i[2] for i in p_range]
    X = QuasiMonteCarlo.sample(samples, lb, ub, QuasiMonteCarlo.SobolSample())

    if batch
        Y = f(X)
        multioutput = Y isa AbstractMatrix
    else
        _y = [f(X[:,j]) for j in axes(X, 2)]
        multioutput = !(eltype(_y) <: Number)
        Y = multioutput ? reduce(hcat,_y) : _y
    end

    srcs = _calculate_standard_regression_coefficients(X, Y)
    corr = _calculate_correlation_matrix(X, Y)
    partials = _calculate_partial_correlation_coefficients(X, Y)

    if method.rank
        X_rank = vcat((sortperm(view(X, i, :))' for i in axes(X, 1))...)
        Y_rank = vcat((sortperm(view(Y, i, :))' for i in axes(Y, 1))...)

        srcs_rank = _calculate_standard_regression_coefficients(X_rank, Y_rank)
        corr_rank = _calculate_standard_regression_coefficients(X_rank, Y_rank)
        partials_rank = _calculate_partial_correlation_coefficients(X_rank, Y_rank)

        return RegressionResult(
            corr,
            srcs,
            partials,
            corr_rank,
            srcs_rank,
            partials_rank
        )
    end

    return RegressionResult(
        corr,
        srcs,
        partials,
        nothing, nothing, nothing
    )
end

function _calculate_standard_regression_coefficients(X, Y)
    β̂ = X' \ Y'
    srcs = β̂ .* std(X, dims = 2) ./ std(Y, dims = 2)
    return srcs
end

function _calculate_correlation_matrix(X, Y)
    corr = cov(X, Y, dims = 2) ./ (std(X, dims = 2) .* std(Y, dims = 2)')
    return corr
end

function _calculate_partial_correlation_coefficients(X, Y)
    XY = vcat(X, Y)
    corr = cov(XY, dims = 2) ./ (std(XY, dims = 2) .* std(XY, dims = 2)')
    prec = inv(corr) # precision matrix
    pcc_XY = -prec ./ sqrt.(diag(prec) .* diag(prec)')
    # return partial correlation matrix relating f: X -> Y model values
    return pcc_XY[axes(X, 1), lastindex(X, 1) .+ axes(Y, 1)]
end
