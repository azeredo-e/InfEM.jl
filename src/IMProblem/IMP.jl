module IMP

using Graphs
using SimpleWeightedGraphs
using Distributions
using Printf
using Random
using DataStructures

using ..DiffusionModels

"""
Defines the interface for Influence Maximization Problems (IMPs).
"""
abstract type AbstractIMProblem end

"""
Defines the interface for traditional Influence Maximizations Problems (IMPs).
"""
abstract type AbstractTraditionalIMP <: AbstractIMProblem end

"""
    IMProblem{G <: AbstractGraph, I <: Integer}(g::G, k::I)

Defines a traditional Influence Maximization Problem (IMP). Given a graph `g` and a budget of seed nodes `k`,
the goal is to find the optimal set of nodes `S` so to maximize the spread of influence in the network.
The parameter `k` controls the solution as `k ≥ |S|`.

The optional parameter `costs` controls the individual cost of influence for each node in the network. In the passed
`Vector` each index match the cost of the node with the same index. This is used in some solvers. If all nodes
have the same cost, a vector of ones can be passed to this parameter.

# Fields
- `g::<:AbstractGraph`: The graph representing the network.
- `k::<:Integer`: The budget of seed nodes.
- `costs::<:AbstractVector{<:AbstractFloat}`=|V|: The costs of each node in the network.
"""
struct IMProblem{
    G <: AbstractGraph,
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

"""
Defines the interface for the solutions of Influence Maximization Problems (IMPs).
"""
abstract type AbstractIMSolution end

"""
    IMSolution{G <: AbstractGraph, I <: Integer, F <: AbstractFloat, VI <: AbstractVector{<:AbstractFloat}}(g::G, k::I, diffusion_model::DiffusionModels.AbstractDiffusionModel, S::VI, spread::F, time::NamedTuple{(:start, :end, :elapsed), Tuple{F, F, F}})

Solution for a Influece Maximization Problem (IMP). It is not expected to be used directly by the user, only
by the appropriate functions from the *Infem.jl* API.

# Fields
- `g::<:AbstractGraph`: The graph representing the network.
- `k::<:Integer`: The budget of seed nodes.
- `diffusion_model::<:AbstractDiffusionModel`: The diffusion model used for generating this result.
- `S::<:AbstractVector{<:Integer}`: The set of seed nodes selected by the algorithm.
- `spread::<:AbstractFloat`: The final estimated spread of the algorithm.
- `time::NamedTuple{(:start, :end, :elapsed), Tuple{<:AbstractFloat, <:AbstractFloat, <:AbstractFloat}}`: The
start, end and elapsed time for execution of the algorithm. It uses the `time()` function for measuring.
- `solution::NamedTuple`: Other information computed by the algorithm. It is solver specific so no specific 
fields are guaranteed to be present. If no additional information is computed, this field will be `nothing`.
"""
struct IMSolution{
    G <: AbstractGraph,
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

"""
Defines the interface for the Influece Maximization Problem (IMP) solver. Custom solvers can be created by
subtyping `AbstractIMSolver` and implementing the `solve` function for that solver type.
"""
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
       CELFpp,
       solve

end # module IMP
