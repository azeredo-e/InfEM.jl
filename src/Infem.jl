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

include("solve.jl") # Needs to be the first import otherwise everthing goes to...
include("DiffusionModels/DiffusionModels.jl") # Needs to be imported second because of type dependencies
include("IMProblem/IMP.jl")
include("EMProblem/IEP.jl")

using .DiffusionModels
using .IMP
using .IEP

export # Problem and solution definitions
       IMProblem,
       AbstractIMProblem,
       AbstractTraditionalIMP
export
       AbstractIEProblem,
       AbstractTopologicalIEP,
       AbstractDiffusionIEP,
       AbstractDiffusionCompleteIEP,
       AbstractDiffusionVertexIEP,
       IEProblem,
       LIPN,
       TENDS
export AbstractIMSolver, Greedy, CELF, CELFpp # IM models
export IndependentCascade, LinearThreshold # Diffusion processes
export run_diffusion_process, solve # Solve functions

end # module Infem
