module IMP

using Graphs
using SimpleWeightedGraphs
using Distributions
using Printf
using Random
using DataStructures

using ..DiffusionModels

abstract type AbstractIMProblem end
abstract type AbstractTraditionalIMP <: AbstractIMProblem end
struct IMProblem{G <: AbstractGraph, I <: Integer} <: AbstractTraditionalIMP
    g::G
    k::I
end

abstract type AbstractIMSolution end
struct IMSolution{
    G <: AbstractGraph,
    I <: Integer,
    F <: AbstractFloat,
    VF <: AbstractVector{F}
} <: AbstractIMSolution
    g::G
    k::I
    diffusion_model::DiffusionModels.AbstractDiffusionModel
    S::Vector{I}
    spread::VF
    time::NamedTuple{(:start, :end, :elapsed), Tuple{F, F, F}}
end

abstract type AbstractIMSolver end
# At each algorithm file there is the type definittion for that solver

include("SimulationBased/greedy.jl")
# include("SimulationBased/celf.jl")
# include("SimulationBased/celfpp.jl")

export
       IMProblem,
       AbstractIMProblem,
       AbstractTraditionalIMP,
       AbstractIMSolution,
       IMSolution,
       AbstractIMSolver,
       Greedy,
       solve

end # module IMP
