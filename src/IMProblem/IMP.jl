module IMP

using Graphs
using SimpleWeightedGraphs
using Distributions
using Printf
using Random
using DataStructures

using ..DiffusionModels

import ..Infem: solve # I f*cking hate this module system

include("../docstrings/IM.jl")

@doc ABSTRACT_IM_PROBLEM_DOC
abstract type AbstractIMProblem end

@doc ABSTRACT_TRADITIONAL_IMP_DOC
abstract type AbstractTraditionalIMP <: AbstractIMProblem end

@doc IM_PROBLEM_DOC
struct IMProblem{
    T,
    G <: AbstractGraph{T},
    I <: Integer,
    VF <: AbstractVector{<:AbstractFloat}
} <: AbstractTraditionalIMP
    g::G
    k::I
    costs::VF
end
function IMProblem(g, k)
    return IMProblem(g, k, ones(Float64, nv(g)))
end

@doc ABSTRACT_IM_SOLUTION_DOC
abstract type AbstractIMSolution end

@doc IM_SOLUTION_DOC
struct IMSolution{
    T,
    G <: AbstractGraph{T},
    I <: Integer,
    F <: AbstractFloat,
    VI <: AbstractVector{<:Integer}
} <: AbstractIMSolution
    g::G
    k::I
    diffusion_model::DiffusionModels.AbstractDiffusionModel
    S::VI
    spread::F
    time::NamedTuple{(:start, :end, :elapsed), Tuple{F, F, F}}
    solution::Union{NamedTuple, Nothing}
end

@doc ABSTRACT_IM_SOLVER_DOC
abstract type AbstractIMSolver end
# At each algorithm file there is the type definittion for that solver

include("solve.jl")
include("SimulationBased/greedy.jl")
include("SimulationBased/celf.jl")
include("SimulationBased/celfpp.jl")

export
       IMProblem,
       AbstractIMProblem,
       AbstractTraditionalIMP,
       AbstractIMSolution,
       IMSolution,
       AbstractIMSolver,
       Greedy,
       CELF,
       CELFpp

end # module IMP
