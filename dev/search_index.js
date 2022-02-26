var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = ACEfit","category":"page"},{"location":"#ACEfit","page":"Home","title":"ACEfit","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for ACEfit.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [ACEfit]","category":"page"},{"location":"#ACEfit.Dat","page":"Home","title":"ACEfit.Dat","text":"Dat: store one configuration (input, e.g., structure, state, ...)  that can  have multiple observations attached to it. Fields:\n\nconfig::Any : the structure \nconfigtype::String : Each dat::Dat belongs to a group identified by a string dat.configtype to allow filtering, and grouping. \nobs::Vector{Any}  : list of observations \nmeta::Dict{String, Any} : any additional meta information that we may want to attach to this data point; this needs to be raw json.\n\n\n\n\n\n","category":"type"},{"location":"#ACEfit.QR","page":"Home","title":"ACEfit.QR","text":"struct QR : linear least squares solver, using standard QR factorisation;  this solver computes \n\n θ = argmin  A theta - y ^2 + lambda  P theta ^2\n\nConstructor\n\nACEfit.QR(; λ = 0.0, P = nothing)\n\nwhere \n\nλ : regularisation parameter \nP : right-preconditioner / tychonov operator\n\n\n\n\n\n","category":"type"},{"location":"#ACEfit.RRQR","page":"Home","title":"ACEfit.RRQR","text":"struct RRQR : linear least squares solver, using rank-revealing QR  factorisation, which can sometimes be more robust / faster than the  standard regularised QR factorisation. This solver first transforms the  parameters theta_P = P theta, then solves\n\n θ = argmin  A P^-1 theta_P - y ^2\n\nwhere the truncation tolerance is given by the rtol parameter, and  finally reverses the transformation. This uses the pqrfact of LowRankApprox.jl;  For further details see the documentation of  LowRankApprox.jl.\n\nCrucially, note that this algorithm is not deterministic; the results can change  slightly between applications.\n\nConstructor\n\nACEfit.RRQR(; rtol = 1e-15, P = I)\n\nwhere \n\nrtol : truncation tolerance\nP : right-preconditioner / tychonov operator\n\n\n\n\n\n","category":"type"},{"location":"#ACEfit.basis_obs","page":"Home","title":"ACEfit.basis_obs","text":"function basis_obs end - evaluate an observation on a basis and a config;  used for llsq assembly, \n\nbasis_obs(obstype, basis, cfg)\n\nFor examle, for a potential energy the implementatin might look like this: \n\nbasis_obs(::Type{TOBS}, basis, at) where {TOBS <: ObsPotentialEnergy} = \n   TOBS.( energy(basis, at) )\n\nNote how the TOBS operation is now broadcast since energy(basis, at)  returns a vector of observations (one for each  basis functions) whereas  for a model we would use eval_obs.\n\n\n\n\n\n","category":"function"},{"location":"#ACEfit.devec_obs","page":"Home","title":"ACEfit.devec_obs","text":"convert a Vector{T} to some real data, e.g.,\n\nx::Vector{Float64}\ndevec_obs(::Type{ObsVirial}, x) = [ x[1] x[2] x[3]; \n                                    x[2] x[4] x[5];\n                                    x[3] x[5] x[6] ]\n\n\n\n\n\n","category":"function"},{"location":"#ACEfit.eval_obs","page":"Home","title":"ACEfit.eval_obs","text":"Evaluate a specific observation type: Given an observation obs,  a model model and  a configuration cfg = dat.config, the call \n\neval_obs(obstype, model, cfg)\n\nmust return the corresponding observation. For example, if  obs::ObsPotentialEnergy and cfg = at::Atoms, and model is an interatomic  potential, then \n\neval_obs(::Type{TOBS}, model, cfg) where {TOBS <: ObsPotentialEnergy} = \n      TOBS( energy(model, cfg) )\n\n\n\n\n\n","category":"function"},{"location":"#ACEfit.titerate-Tuple{Any, Any}","page":"Home","title":"ACEfit.titerate","text":"titerate(f, data; kwargs...) Multi-threaded map loop. At each iteration the function f(dat) is executed, where dat in data. The order is not necessarily preserved. In fact the  costs array is used to sort the data by decreasing cost to ensure the  most costly configurations are encountered first. This helps avoid threads  without work at the end of the loop.\n\n\n\n\n\n","category":"method"},{"location":"#ACEfit.vec_obs","page":"Home","title":"ACEfit.vec_obs","text":"convert some real data, in some generic format, into a vector to be stored in a Dat or Lsq system. E.g.,\n\nV = virial(...)::Matrix{Float64}\nobsV = ObsVirial(V)\nvec_obs(obsV::ObsVirial) = obsV.V[ [1,2,3,5,6,9] ]  # NB: V is symmetric\n\n\n\n\n\n","category":"function"}]
}
