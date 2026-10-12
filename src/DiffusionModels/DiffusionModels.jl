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

export
       AbstractDiffusionModel,
       DiffusionResult,
       IndependentCascade,
       LinearThreshold,
       run_diffusion_process

end # module DiffusionModels
