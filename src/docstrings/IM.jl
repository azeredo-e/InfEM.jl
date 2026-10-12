### ABSTRACT TYPES DEFINITIONS ### 

const ABSTRACT_IM_PROBLEM_DOC = """
Defines the interface for Influence Maximization Problems (IMPs).
"""

const ABSTRACT_TRADITIONAL_IMP_DOC = """
Defines the interface for traditional Influence Maximizations Problems (IMPs).
"""

const ABSTRACT_IM_SOLUTION_DOC = """
Defines the interface for the solutions of Influence Maximization Problems (IMPs).
"""

const ABSTRACT_IM_SOLVER_DOC = """
Defines the interface for the Influece Maximization Problem (IMP) solver. Custom solvers can be created by
subtyping `AbstractIMSolver` and implementing the `solve` function for that solver type.
"""

### SOLUTIONS AND PROBLEM TYPES DEFINITIONS ###

const IM_SOLUTION_DOC = """
    IMSolution{...}

Solution for a Influece Maximization Problem (IMP). It is not expected to be used directly by the user, only
by the appropriate functions from the *Infem.jl* API.

# Fields
- `g::<:AbstractGraph{T}`: The graph representing the network.
- `k::<:Integer`: The budget of seed nodes.
- `diffusion_model::<:AbstractDiffusionModel`: The diffusion model used for generating this result.
- `S::<:AbstractVector{<:Integer}`: The set of seed nodes selected by the algorithm.
- `spread::<:AbstractFloat`: The final estimated spread of the algorithm.
- `time::NamedTuple{(:start, :end, :elapsed), Tuple{<:AbstractFloat, <:AbstractFloat, <:AbstractFloat}}`: The
    start, end and elapsed time for execution of the algorithm. It uses the `time()` function for measuring.
- `solution::NamedTuple`: Other information computed by the algorithm. It is solver specific so no specific 
    fields are guaranteed to be present. If no additional information is computed, this field will be `nothing`.
"""

const IM_PROBLEM_DOC = """
    IMProblem{T, G <: AbstractGraph{T}, I <: Integer}(g::G, k::I)

Defines a traditional Influence Maximization Problem (IMP). Given a graph `g` and a budget of seed nodes `k`,
the goal is to find the optimal set of nodes `S` so to maximize the spread of influence in the network.
The parameter `k` controls the solution as `k ≥ |S|`.

The optional parameter `costs` controls the individual cost of influence for each node in the network. In the passed
`Vector` each index match the cost of the node with the same index. This is used in some solvers. If all nodes
have the same cost, a vector of ones can be passed to this parameter.

# Fields
- `g::<:AbstractGraph{T}`: The graph representing the network.
- `k::<:Integer`: The budget of seed nodes.
- `costs::<:AbstractVector{<:AbstractFloat}`=|V|: The costs of each node in the network.
"""

### ALGORITHMS DEFINITIONS ###

const SOLVE_IMP_DOC = """
    solve{...}(im_problem::IM,solver::S,diffusion_model::D)

Defines the basic interface for solving influence maximization problems (IMPs). The function is expected to be
implemented for each solver type, and it should return an `IMSolution` object.

# Arguments
- `im_problem::<:AbstractTraditionalIMP`: The influence maximization problem to be solved.
- `solver::<:AbstractIMSolver`: The solver to be used for solving the problem.
- `diffusion_model::<:AbstractDiffusionModel`: The diffusion model to be used for simulating the spread of influence in the network.

# Returns
- `IMSolution`: The solution of the influence maximization problem, containing the selected seed nodes and the estimated spread of influence.

# Throws
- `ArgumentError`: If the solver is not implemented for the given problem, solver and diffusion model.
"""

const GREEDY_DOC = """
    Greedy{B<:Bool}(verbose::B)}

Defines the Greedy solver for Influence Maximization problems (IMPs). It can be passed to the `solve` function
to solve an IMP.

Based on the original greedy algorithm from Kempe et al. (2003).

# Arguments
- verbose::<:Bool=false: Verbose output.
"""

const CELF_DOC = """
    CELF{B <: Bool}(verbose::B = false)

Defines the CELF solver for Influence Maximization problems (IMPs). It can be passed to the `solve` function
to solve an IMP.

CELF is an evolution on the *Greedy* algorithm, it treats our number `k` as a budget and allows node to have
different costs.

When running `solve` the optional `solution` field of the returned `IMSolution` will contain a `NamedTuple`
with the following fields:
- `winner`: The winner of the two strategies, either `:UC` or `:CB`.
- `A_UC`: The set of seed nodes selected by the *Uniform Cost* strategy.
- `R_UC`: The final estimated spread of the *Uniform Cost* strategy.
- `A_CB`: The set of seed nodes selected by the *Cost Benefit* strategy.
- `R_CB`: The final estimated spread of the *Cost Benefit* strategy.

Based on the original CELF algorithm from Leskovec et al. (2007).

# Arguments
- verbose::<:Bool=false: Verbose output.
"""

const CELFPP_DOC = """
    CELFpp{B <: Bool}(verbose::B = false)

Defines the CELF++ solver for Influence Maximization problems (IMPs). It can be passed to the `solve` function
to solve an IMP.

CELF++ is an evolution on the *CELF* algorithm.

When running `solve` the optional `solution` field of the returned `IMSolution` will contain a `NamedTuple`
with the following fields:
- lookups::<:Integer: The number of times the diffusion process was run during the execution of the algorithm.

Based on the original CELF++ algorithm from Goyal et al. (2011).

# Arguments
- verbose::<:Bool=false: Verbose output.
"""
