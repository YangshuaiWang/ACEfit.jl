module ACEfit_PythonCall_ext

using ACEfit
using PythonCall


function ACEfit.solve(solver::ACEfit.SKLEARN_BRR, A, y)
    @info "Entering SKLEARN_BRR"
    BRR = pyimport("sklearn.linear_model")."BayesianRidge"
    clf = BRR(n_iter = solver.n_iter, tol = solver.tol, fit_intercept = true,
              compute_score = true)
    clf.fit(A, y)
    if length(clf.scores_) < solver.n_iter
        @info "BRR converged to tol=$(solver.tol) after $(length(clf.scores_)) iterations."
    else
        @warn "\nBRR did not converge to tol=$(solver.tol) after n_iter=$(solver.n_iter) iterations.\n"
    end
    c = clf.coef_
    return Dict{String, Any}("C" => pyconvert(Array, c) )
end


function ACEfit.solve(solver::ACEfit.SKLEARN_ARD, A, y)
    ARD = pyimport("sklearn.linear_model")."ARDRegression"
    clf = ARD(n_iter = solver.n_iter, threshold_lambda = solver.threshold_lambda,
              tol = solver.tol,
              fit_intercept = true, compute_score = true)
    clf.fit(A, y)
    if length(clf.scores_) < solver.n_iter
        @info "ARD converged to tol=$(solver.tol) after $(length(clf.scores_)) iterations."
    else
        @warn "\n\nARD did not converge to tol=$(solver.tol) after n_iter=$(solver.n_iter) iterations.\n\n"
    end
    c = clf.coef_
    return Dict{String, Any}("C" => pyconvert(Array, c) )
end

function ACEfit.solve(solver::ACEfit.SKLEARN_LassoReg, A, y)
    Lasso = pyimport("sklearn.linear_model")."Lasso"
    clf = Lasso(alpha = solver.alpha, fit_intercept = false)
    clf.fit(A, y)
    c = clf.coef_
    return Dict{String, Any}("C" => pyconvert(Array, c) )
end

function ACEfit.solve(solver::ACEfit.SKLEARN_RidgeReg, A, y)
    Ridge = pyimport("sklearn.linear_model")."Ridge"
    clf = Ridge(alpha = solver.alpha)
    clf.fit(A, y)
    c = clf.coef_
    return Dict{String, Any}("C" => pyconvert(Array, c) )
end

function ACEfit.solve(solver::ACEfit.SKLEARN_LinearReg, A, y)
    Linear = pyimport("sklearn.linear_model")."LinearRegression"
    clf = Linear()
    clf.fit(A, y)
    c = clf.coef_
    return Dict{String, Any}("C" => pyconvert(Array, c) )
end

function ACEfit.solve(solver::ACEfit.SKLEARN_ElasticNetReg, A, y)
    ElasticNet = pyimport("sklearn.linear_model")."ElasticNet"
    clf = ElasticNet(alpha = solver.alpha, l1_ratio = solver.l1_ratio)
    clf.fit(A, y)
    c = clf.coef_
    return Dict{String, Any}("C" => pyconvert(Array, c) )
end

function ACEfit.solve(solver::ACEfit.SKLEARN_LassoLarsReg, A, y)
    LassoLars = pyimport("sklearn.linear_model")."LassoLars"
    clf = LassoLars(alpha = solver.alpha)
    clf.fit(A, y)
    c = clf.coef_
    return Dict{String, Any}("C" => pyconvert(Array, c) )
end

function ACEfit.solve(solver::ACEfit.SKLEARN_QuantileReg, A, y)
    quantile = pyimport("sklearn.linear_model")."QuantileRegressor"
    clf = quantile(alpha = solver.alpha, solver="highs")
    clf.fit(A, y)
    c = clf.coef_
    return Dict{String, Any}("C" => pyconvert(Array, c) )
end

function ACEfit.solve(solver::ACEfit.SKLEARN_SGDReg, A, y)
    sgd = pyimport("sklearn.linear_model")."SGDRegressor"
    clf = sgd(loss = solver.loss, max_iter = solver.max_iter, verbose = solver.verbose)
    clf.fit(A, y)
    c = clf.coef_
    return Dict{String, Any}("C" => pyconvert(Array, c) )
end

function ACEfit.solve(solver::ACEfit.SKLEARN_RansacReg, A, y)
    ransac = pyimport("sklearn.linear_model")."RANSACRegressor"
    clf = ransac(min_samples = solver.min_samples)
    clf.fit(A, y)
    c = clf.estimator_.coef_
    return Dict{String, Any}("C" => pyconvert(Array, c) )
end

function ACEfit.solve(solver::ACEfit.SKLEARN_HuberReg, A, y)
    huber = pyimport("sklearn.linear_model")."HuberRegressor"
    clf = huber(epsilon = solver.epsilon, max_iter = solver.max_iter, alpha = solver.alpha)
    clf.fit(A, y)
    c = clf.coef_
    return Dict{String, Any}("C" => pyconvert(Array, c) )
end

end