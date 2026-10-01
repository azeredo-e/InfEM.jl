module DiffusionModels

using Distributions
using Graphs
using SimpleWeightedGraphs

"""
Defines the interface for diffusion models.

A diffusion model is the process through which information spreads through the network. Usually it is based on
a epidemiological model. Custom models can be created by subtyping `AsbtractDiffusionModel` and implementing
the `run_diffusion_process` function.
"""
abstract type AbstractDiffusionModel end

include("independent_cascade.jl")
include("linear_threshold.jl")

export AbstractDiffusionModel, IndependentCascade, LinearThreshold, run_diffusion_process

end # module DiffusionModels
