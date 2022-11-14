
using ACEfit 
using LinearAlgebra

@info("Test Solver on overdetermined system")
Nobs = 10_000
Nfeat = 100 
A = randn(Nobs, Nfeat) / sqrt(Nobs)
y = randn(Nobs)
P = Diagonal(1.0 .+ rand(Nfeat))

@info(" ... QR")
solver = ACEfit.QR()
results = ACEfit.linear_solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)

@info(" ... regularised QR, λ = 1.0")
solver = ACEfit.QR(lambda = 1e0, P = P)
results = ACEfit.linear_solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)

@info(" ... regularised QR, λ = 10.0")
solver = ACEfit.QR(lambda = 1e1, P = P)
results = ACEfit.linear_solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)

@info(" ... RRQR, rtol = 1e-15")
solver = ACEfit.RRQR(rtol = 1e-15, P = P)
results = ACEfit.linear_solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)

@info(" ... RRQR, rtol = 0.5")
solver = ACEfit.RRQR(rtol = 0.5, P = P)
results = ACEfit.linear_solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)

@info(" ... RRQR, rtol = 0.99")
solver = ACEfit.RRQR(rtol = 0.99, P = P)
results = ACEfit.linear_solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)

@info(" ... LSQR")
solver = ACEfit.LSQR(damp=0, atol=1e-6)
results = ACEfit.linear_solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)

@info(" ... SKLEARN_BRR")
solver = ACEfit.SKLEARN_BRR()
results = ACEfit.linear_solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)

@info(" ... SKLEARN_ARD")
solver = ACEfit.SKLEARN_ARD()
results = ACEfit.linear_solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)

@info(" ... Bayesian Linear")
solver = ACEfit.BL()
results = ACEfit.linear_solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)

@info(" ... Bayesian ARD")
solver = ACEfit.BARD()
results = ACEfit.linear_solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)

@info(" ... Bayesian Linear Regression SVD")
solver = ACEfit.BayesianLinearRegressionSVD()
results = ACEfit.linear_solve(solver, A, y)
C = results["C"]
@show norm(A * C - y)
@show norm(C)
