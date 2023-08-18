var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = ACEfit","category":"page"},{"location":"#ACEfit","page":"Home","title":"ACEfit","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for ACEfit.","category":"page"},{"location":"#Scikit-learn-solvers","page":"Home","title":"Scikit-learn solvers","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"To use Python based Scikit-learn solvers you need to load PythonCall in addition to ACEfit.","category":"page"},{"location":"","page":"Home","title":"Home","text":"using ACEfit\nusing PythonCall","category":"page"},{"location":"#MLJ-solvers","page":"Home","title":"MLJ solvers","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"To use MLJ solvers you need to load MLJ in addition to ACEfit","category":"page"},{"location":"","page":"Home","title":"Home","text":"using ACEfit\nusing MLJ","category":"page"},{"location":"","page":"Home","title":"Home","text":"After that you need to load an appropriate MLJ solver. Take a look on available MLJ solvers. Note that only MLJScikitLearnInterface.jl and MLJLinearModels.jl have extension available. To use other MLJ solvers please file an issue.","category":"page"},{"location":"","page":"Home","title":"Home","text":"You need to load the solver and then create a solver structure","category":"page"},{"location":"","page":"Home","title":"Home","text":"# Load ARD solver\nARDRegressor = @load ARDRegressor pkg=MLJScikitLearnInterface\n\n# Create the solver itself and give it parameters\nsolver = ARDRegressor(\n    n_iter = 300,\n    tol = 1e-3,\n    threshold_lambda = 10000\n)","category":"page"},{"location":"","page":"Home","title":"Home","text":"After this you can use the MLJ solver like any other solver.","category":"page"},{"location":"#Index","page":"Home","title":"Index","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [ACEfit]","category":"page"},{"location":"#ACEfit.AbstractData","page":"Home","title":"ACEfit.AbstractData","text":"ACEfit users should define a type of the form:     UserData <: AbstractData\n\nSeveral functions acting on such a type should be implemented:     countobservations     featurematrix     targetvector     weightvector\n\n\n\n\n\n","category":"type"},{"location":"#ACEfit.BLR","page":"Home","title":"ACEfit.BLR","text":"struct BLR : Bayesian linear regression\n\nRefer to bayesianlinear.jl (for now) for kwarg definitions.\n\n\n\n\n\n","category":"type"},{"location":"#ACEfit.LSQR","page":"Home","title":"ACEfit.LSQR","text":"LSQR\n\n\n\n\n\n","category":"type"},{"location":"#ACEfit.QR","page":"Home","title":"ACEfit.QR","text":"struct QR : linear least squares solver, using standard QR factorisation;  this solver computes \n\n θ = argmin  A theta - y ^2 + lambda  P theta ^2\n\nConstructor\n\nACEfit.QR(; lambda = 0.0, P = nothing)\n\nwhere \n\nλ : regularisation parameter \nP : right-preconditioner / tychonov operator\n\n\n\n\n\n","category":"type"},{"location":"#ACEfit.RRQR","page":"Home","title":"ACEfit.RRQR","text":"struct RRQR : linear least squares solver, using rank-revealing QR  factorisation, which can sometimes be more robust / faster than the  standard regularised QR factorisation. This solver first transforms the  parameters theta_P = P theta, then solves\n\n θ = argmin  A P^-1 theta_P - y ^2\n\nwhere the truncation tolerance is given by the rtol parameter, and  finally reverses the transformation. This uses the pqrfact of LowRankApprox.jl;  For further details see the documentation of  LowRankApprox.jl.\n\nCrucially, note that this algorithm is not deterministic; the results can change  slightly between applications.\n\nConstructor\n\nACEfit.RRQR(; rtol = 1e-15, P = I)\n\nwhere \n\nrtol : truncation tolerance\nP : right-preconditioner / tychonov operator\n\n\n\n\n\n","category":"type"},{"location":"#ACEfit.SKLEARN_ARD","page":"Home","title":"ACEfit.SKLEARN_ARD","text":"SKLEARN_ARD\n\n\n\n\n\n","category":"type"},{"location":"#ACEfit.SKLEARN_BRR","page":"Home","title":"ACEfit.SKLEARN_BRR","text":"SKLEARN_BRR\n\n\n\n\n\n","category":"type"},{"location":"#ACEfit.assemble-Tuple{AbstractVector{<:ACEfit.AbstractData}, Any}","page":"Home","title":"ACEfit.assemble","text":"Assemble feature matrix and target vector for given data and basis.\n\n\n\n\n\n","category":"method"},{"location":"#ACEfit.assemble_weights-Tuple{AbstractVector{<:ACEfit.AbstractData}}","page":"Home","title":"ACEfit.assemble_weights","text":"Assemble full weight vector for vector of data elements.\n\n\n\n\n\n","category":"method"},{"location":"#ACEfit.count_observations-Tuple{ACEfit.AbstractData}","page":"Home","title":"ACEfit.count_observations","text":"Returns the corresponding number of rows in the design matrix.\n\n\n\n\n\n","category":"method"},{"location":"#ACEfit.feature_matrix-Tuple{ACEfit.AbstractData}","page":"Home","title":"ACEfit.feature_matrix","text":"Returns the corresponding design matrix (A) entries.\n\n\n\n\n\n","category":"method"},{"location":"#ACEfit.target_vector-Tuple{ACEfit.AbstractData}","page":"Home","title":"ACEfit.target_vector","text":"Returns the corresponding target vector (Y) entries.\n\n\n\n\n\n","category":"method"},{"location":"#ACEfit.weight_vector-Tuple{ACEfit.AbstractData}","page":"Home","title":"ACEfit.weight_vector","text":"Returns the corresponding weight vector (W) entries.\n\n\n\n\n\n","category":"method"}]
}
