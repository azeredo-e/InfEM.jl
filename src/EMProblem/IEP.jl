module IEP
using Graphs
using SimpleWeightedGraphs
using Distributions
using Printf
using Random
using DataStructures

using ..DiffusionModels

import ..Infem: solve
    
abstract type AbstractIEProblem end

"""
Topological based measures, such as degree, closeness, betweenness, etc.
"""
abstract type AbstractTopologicalIEP <: AbstractIEProblem end

"""
Algorithms that are based on the result of the cascade
"""
abstract type AbstractDiffusionIEP <: AbstractIEProblem end

"""
For when we have G = (V, E)
"""
abstract type AbstractDiffusionCompleteIEP <: AbstractDiffusionIEP end

"""
For when we have the set of vertices but not the edges
"""
abstract type AbstractDiffusionVertexIEP <: AbstractDiffusionIEP end


#! Need to fix this type shenanigans before pushing to master
struct IEProblem{
    T,
    G <: AbstractGraph{T},
} <: AbstractDiffusionVertexIEP
    g::G
    d::DiffusionResult
end

abstract type AbstractIESolution end

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