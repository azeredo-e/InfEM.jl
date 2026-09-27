module DiffusionModels

using Distributions
using Graphs
using SimpleWeightedGraphs

abstract type AbstractDiffusionModel end

include("independent_cascade.jl")
include("linear_threshold.jl")

export AbstractDiffusionModel, IndependentCascade, LinearThreshold, run_diffusion_process

end # module DiffusionModels
