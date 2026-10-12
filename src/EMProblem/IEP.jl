module IEP
using Graphs
using SimpleWeightedGraphs
using Distributions
using Printf
using Random
using DataStructures

using ..DiffusionModels

import ..Infem: solve

include("../docstrings/IE.jl")

@doc ABSTRACT_IE_PROBLEM_DOC
abstract type AbstractIEProblem end

@doc ABSTRACT_TOPOLOGICAL_IEP_DOC
abstract type AbstractTopologicalIEP <: AbstractIEProblem end

@doc ABSTRACT_DIFFUSION_IEP_DOC
abstract type AbstractDiffusionIEP <: AbstractIEProblem end

@doc ABSTRACT_DIFFUSION_COMPLETE_IEP
abstract type AbstractDiffusionCompleteIEP <: AbstractDiffusionIEP end

@doc ABSTRACT_DIFFUSION_VERTEX_IEP
abstract type AbstractDiffusionVertexIEP <: AbstractDiffusionIEP end

@doc IE_PROBLEM_DOC
struct IEProblem{
    T,
    G <: AbstractGraph{T}
} <: AbstractDiffusionVertexIEP
    g::G
    d::DiffusionResult
end

abstract type AbstractIESolution end

@doc IE_SOLUTION_DOC
struct IESolution{
    T,
    G <: AbstractGraph{T},
    F <: AbstractFloat
} <: AbstractIESolution
    g::G
    d::DiffusionResult{F}
    ret::Any
    time::NamedTuple{(:start, :end, :elapsed), Tuple{F, F, F}}
end

# include("Topological/degree.jl")
# include("DiffusionComplete/")
include("DiffusionVertex/mimi.jl")
include("DiffusionVertex/lipn.jl")
include("DiffusionVertex/tends.jl")

export
       AbstractIEProblem,
       AbstractTopologicalIEP,
       AbstractDiffusionIEP,
       AbstractDiffusionCompleteIEP,
       AbstractDiffusionVertexIEP,
       IEProblem,
       AbstractIESolution,
       IESolution,
       LIPN

end # module IEP
