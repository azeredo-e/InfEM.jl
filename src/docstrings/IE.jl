### ABSTRACT TYPES DEFINITIONS ### 

const ABSTRACT_IE_PROBLEM_DOC = """
Defines the interface for Influence Estimation Problems (IMPs).
"""

const ABSTRACT_TOPOLOGICAL_IEP_DOC = """
Basic interface for topological based measures, such as degree, closeness, betweenness, etc.
"""

const ABSTRACT_DIFFUSION_IEP_DOC = """
Basic interface for IE algorithms that build weights or edges based on the results of the cascades.
"""

const ABSTRACT_DIFFUSION_COMPLETE_IEP = """
Basic interface for IE algorithms when we have both the set of vertices and edges and are just estimating the weights.
"""

const ABSTRACT_DIFFUSION_VERTEX_IEP = """
Basic interface for IE algorithms when we have the set of vertices but not the edges and our aim is to estimate both.
"""

### SOLUTIONS AND PROBLEM TYPES DEFINITIONS ###

const IE_PROBLEM_DOC = """
    IEProblem(g::AbstractGraph{T}, d::DiffusionResult)

Defines a Influence Estimation Problem (IEP). Given a graph `g` estimate the weights of edges and/or the edges
itself.

# Fields
- `g::AbstractGraph{T}`: The graph object with the set of vertices with each to build the complete graph with
    edges and/or weights.
- `d::DiffusionResult`: The diffusion result object from an executed diffusion process.
"""

const IE_SOLUTION_DOC = """
    IESolution{...}

The solution object from a Influence Estimation Problem (IEP). It is not expected to be used directly by the user, only
by the appropriate functions from the *Infem.jl* API.

# Fields
- `g::<:AbstractGraph{T}`: The graph representing the network.
- `d::DiffusionResult`: The diffusion result object used for estimation.
- `ret::Any`: the result
- `time::NamedTuple{(:start, :end, :elapsed), Tuple{<:AbstractFloat, <:AbstractFloat, <:AbstractFloat}}`: The
    start, end and elapsed time for execution of the algorithm. It uses the `time()` function for measuring.
"""

### ALGORITHMS DEFINITIONS ###

const SOLVE_IEP_DOC = """

"""

const LIPN_DOC = """
    LIPN{...}(...)

Defines the LIPN (Learning Influence Probabilities in diffusion Networks without timestamps) solver. An influence
and edge estimation algorithm based on final cascade results od a diffusion process, as the name implies, it doesnt
requires the timestamp of contagion between nodes, only the final result of the diffusion process in order to generate
the most likely edge set and weights for a given network.

Based on the original LIPN algorithm from Wang et al. (2025).

# Fields
- `e_threshold::AbstractFloat=0.5`: Since LIPN is a probabilistic model, this parameter sets the threshohold for
    which edges to keep. 
- `em_tol::AbstractFloat=1e-6`: Stopping crieteria for convergence in the EM algorithm.
- `em_maxiter::Integer=100`: Max number of iteration for the EM algorithm.
- `sa_temp_ini::Number=100`: Initial temperature for the SA algorithm used for tuning the final result and ensure
    convergence.
- `sa_temp_final::Number=1`: Final temperature for the SA algorithm.
- `sa_epochs::Integer=200`: Max number of iteration for the SA algorithm.
- `sa_temp_decay::Number=8`: Temperature decays in each step of the SA algorithm.
- `verbose::Bool=false`: Verbose output.
"""
