
using Distributed
# addprocs(5, exeflags="--project=$(Base.active_project())")

@everywhere using ACEpotentials 

using JuLIP, LinearAlgebra, ACEfit
# using MLJ
# using MLJScikitLearnInterface
using Plots

Solvers = [:QR, :RRQR, :LSQR, :TruncatedSVD, :BLR, :ARD, :BRR, :RansacReg]

data_keys = (energy_key = "energy", force_key = "force", virial_key="none")
train = read_extxyz("../Zuo/Si_train.xyz")
test = read_extxyz("../Zuo/Si_test.xyz")
model = acemodel(elements = [:Si], order = 3, totaldegree = 16, rcut = 6.0);

Err = Dict()
SOLVERS = Dict(:QR => ACEfit.QR(), :RRQR => ACEfit.RRQR(), :LSQR => ACEfit.LSQR(), :TruncatedSVD => ACEfit.TruncatedSVD(),
               :BLR => ACEfit.BLR(), :ARD => ACEfit.SKLEARN_ARD(), :BRR => ACEfit.SKLEARN_BRR(),
               :LinearReg => ACEfit.SKLEARN_LinearReg(), :LassoReg => ACEfit.SKLEARN_LassoReg(), :RidgeReg => ACEfit.SKLEARN_RidgeReg(), :ElasticNet => ACEfit.SKLEARN_ElasticNetReg(), 
               :LassoLarsReg => ACEfit.SKLEARN_LassoLarsReg(), :QuantileReg => ACEfit.SKLEARN_QuantileReg(), 
               :SGDReg => ACEfit.SKLEARN_SGDReg(), :HuberReg => ACEfit.SKLEARN_HuberReg(), :RansacReg => ACEfit.SKLEARN_RansacReg())
for sol in Solvers
    solver = SOLVERS[sol]
    acename = string("../results/Si/", string(sol), ".json")
    figname = string("../results/Si/", string(sol), ".png")
    acefit!(model, train; solver=solver, data_keys...);
    @info("Training Errors")
    ACEpotentials.linear_errors(train, model; data_keys...);
    @info("Testing Errors")
    configerr = ACEpotentials.linear_errors(test, model; data_keys...);
    frmse = round.(configerr["rmse"]["set"]["F"] * 1000, digits=2)

    Err[sol] = frmse

    ace = model.potential;
    Fref = []; Face = [];
    for tr in train
        exact = mat(tr.data["force"].data)[:]
        estim = mat(forces(ace, tr))[:]
        push!(Fref, exact...)
        push!(Face, estim...)
    end
    fmin = minimum(Fref); fmax = maximum(Fref);
    Fref_te = []; Face_te = [];
    for te in test
        exact = mat(te.data["force"].data)[:]
        estim = mat(forces(ace, te))[:]
        push!(Fref_te, exact...)
        push!(Face_te, estim...)
    end
    p = scatter(Fref, Face, color=:red, alpha=0.4, label="Train")
    scatter!(Fref, Face, color=:blue, alpha=0.4, label="Test")
    plot!(fmin:0.001:fmax, fmin:0.001:fmax, color=:black)
    plot!(xlabel="Reference Force", ylabel="ACE Predicted Force", legend=:topright, framestyle=:box, axis_ratio=:equal, size=(600, 600))
    annotate!(fmax÷3, fmin, text("RMSE = $frmse meV/A", :black, :left, 12))
    savefig(figname)
    ACEpotentials.export2json(acename, model)
    
end


