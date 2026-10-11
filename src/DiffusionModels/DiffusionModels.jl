module DiffusionModels

using Distributions
using Graphs
using SimpleWeightedGraphs

"""
Defines the interface for diffusion models.

A diffusion model is the process through which information spreads through the network. Usually it is based on
an epidemiological model. Custom models can be created by subtyping `AsbtractDiffusionModel` and implementing
the `run_diffusion_process` function.
"""
abstract type AbstractDiffusionModel end

abstract type AbstractDiffusionProcessResult end

"""
    DiffusionResult{F<:AbstractFloat}(diffusion_history::Vector{BitVector}, average_spread::F) <: AbstractDiffusionProcessResult

Represents the result of a diffusion process. It contains the history of activated nodes at each iteration.

# Fields
`diffusion_history::Vector{BitVector}`: A vector with each entry containing a `BitVector` where each of its entries
represent the state of that node on that specific diffusion, 0 for inactive and 1 for active.
`average_spread::AbstractFloat`: The average spread of the network through all runs of diffusions.
"""
struct DiffusionResult{
    F <: AbstractFloat
} <: AbstractDiffusionProcessResult
    diffusion_history::Vector{BitVector}
    average_spread::F
end
function DiffusionResult(; diffusion_history::Vector{BitVector})
    DiffusionResult(diffusion_history, 0.0)
end

include("independent_cascade.jl")
include("linear_threshold.jl")

export AbstractDiffusionModel, DiffusionResult, IndependentCascade, LinearThreshold, run_diffusion_process

end # module DiffusionModels
