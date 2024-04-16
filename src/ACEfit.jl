module ACEfit

include("bayesianlinear.jl")
include("data.jl")
include("assemble.jl")
include("solvers.jl")
include("fit.jl")
# include("../ext/ACEfit_MLJLinearModels_ext.jl")
include("../ext/ACEfit_PythonCall_ext.jl")

end
