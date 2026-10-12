### ABSTRACT TYPES DEFINITIONS ### 

const ABSTRACT_DIFFUSION_MODEL_DOC = """
Defines the interface for diffusion models.

A diffusion model is the process through which information spreads through the network. Usually it is based on
an epidemiological model. Custom models can be created by subtyping `AsbtractDiffusionModel` and implementing
the `run_diffusion_process` function.
"""

### SOLUTIONS AND PROBLEM TYPES DEFINITIONS ###

const DIFFUSION_RESULT_DOC = """
    DiffusionResult{F<:AbstractFloat}(diffusion_history::Vector{BitVector}, average_spread::F) <: AbstractDiffusionProcessResult

Represents the result of a diffusion process. It contains the history of activated nodes at each iteration.

# Fields
`diffusion_history::Vector{BitVector}`: A vector with each entry containing a `BitVector` where each of its entries
    represent the state of that node on that specific diffusion, 0 for inactive and 1 for active.
`average_spread::AbstractFloat`: The average spread of the network through all runs of diffusions.
"""

### ALGORITHMS DEFINITIONS ###

const INDEPENDENT_CASCADE_DOC = """
    IndependentCascade{N <: Integer,D <: UnivariateDistribution}(n_iters::N, rng::D) <: AbstractDiffusionModel

Independent Cascade diffusion model.

Originally an epidemiological model, it has been adapted to be used in the IM and IE scenario. It considers
influence as a stochastic problem. Nodes, given their neighbors "stubbornes", have a chance to influence
other nodes following a random trial based on a probability distribution.  
Each time step, for each edge, a random trial is run, if succeded the receiving node will be from now on
conisidered active, and, on the next step, passible of influencing its neighbors.

The process is then run as a Monte Carlo simulation over multiple runs, the average number of active nodes is
then considered to be the result of the diffusion process.

# Fields
- `n_iters::Integer=10_000`: Number of iterations of the MC simulation to run.
- `rng::Distributions.UnivariateDistributions=Uniform(0,1)`: The rng generator to use for the influence trial in
    each step. It defaults to a *Distributions.jl* Uniform(0,1) univariate distributions.

# Returns
-  `::DiffusionResult`: A `DiffusionResult` struct with fields `diffusion_history::Vector{BitVector}` and
`average_spreap::AbstractFloat`.
"""
