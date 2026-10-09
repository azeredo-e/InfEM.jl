"""
*Infem.jl* is a package for Influence Estimation and Maximization in networks.
"""
module Infem

using Printf
using Random
using Distributions
using Graphs
using SimpleWeightedGraphs
using DataStructures

const STD_N_ITERS = 10_000

include("DiffusionModels/DiffusionModels.jl")
include("IMProblem/IMP.jl")

using .DiffusionModels
using .IMP

export # Problem and solution definitions
       IMProblem,
       AbstractIMProblem,
       AbstractTraditionalIMP,
       AbstractIMSolution,
       IMSolution
export AbstractIMSolver, Greedy, CELF # IM models
export IndependentCascade, LinearThreshold # Diffusion processes
export run_diffusion_process, solve # Solve functions

end # module Infem
