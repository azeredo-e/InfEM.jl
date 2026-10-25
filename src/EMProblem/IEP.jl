module IEP
using Graphs
using SimpleWeightedGraphs
using Distributions
using Printf
using Random
using DataStructures
using SparseArrays
using SpecialFunctions: loggamma
# using Optim

using ..DiffusionModels

import ..Infem: solve

include("../docstrings/IE.jl")

@doc ABSTRACT_IE_PROBLEM_DOC
abstract type AbstractIEProblem end

@doc IE_PROBLEM_DOC
struct IEProblem{
    T,
    G <: AbstractGraph{T}
} <: AbstractIEProblem
    g::G
    d::DiffusionResult
end

@doc ABSTRACT_IEP_SOLVER_DOC
abstract type AbstractIEPSolver end

abstract type AbstractIESolution end

@doc IE_SOLUTION_DOC
struct IESolution{
    T,
    G <: AbstractGraph{T},
    H <: AbstractVector{<:AbstractVector},
    F <: AbstractFloat
} <: AbstractIESolution
    g::G
    d::DiffusionResult{H, F}
    ret::Any
    time::NamedTuple{(:start, :end, :elapsed), Tuple{F, F, F}}
end

include("utils.jl")
include("DiffusionVertex/utils.jl")
# include("Topological/degree.jl")
include("DiffusionComplete/netrate.jl")
# include("DiffusionComplete/dings.jl")
include("DiffusionVertex/mimi.jl")
include("DiffusionVertex/lipn.jl")
include("DiffusionVertex/tends.jl")

export
    AbstractIEProblem,
    AbstractIESolution,
    IEProblem,
    IESolution,
    LIPN,
    NetRate,
    TENDS
export AbstractTransmissionModel, Exponential, PowerLaw, Rayleigh

end # module IEP
