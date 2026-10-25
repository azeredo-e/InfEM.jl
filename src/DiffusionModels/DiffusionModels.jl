module DiffusionModels

using Distributions
using Graphs
using SimpleWeightedGraphs

include("../docstrings/diffusion.jl")

@doc ABSTRACT_DIFFUSION_MODEL_DOC
abstract type AbstractDiffusionModel end

abstract type AbstractDiffusionProcessResult end

@doc DIFFUSION_RESULT_DOC
struct DiffusionResult{
    H <: AbstractVector{<:AbstractVector},
    F <: AbstractFloat
} <: AbstractDiffusionProcessResult
    diffusion_history::H
    average_spread::F
end
function DiffusionResult(; diffusion_history::AbstractVector{<:AbstractVector})
    return DiffusionResult(diffusion_history, 0.0)
end

include("independent_cascade.jl")
include("linear_threshold.jl")

export
       AbstractDiffusionModel,
       DiffusionResult,
       IndependentCascade,
       LinearThreshold,
       run_diffusion_process, FinalStatus, TimedCascade

end # module DiffusionModels
