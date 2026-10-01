"""
    solve{IM <: AbstractTraditionalIMP,S <: AbstractIMSolution,D <: DiffusionModels.AbstractDiffusionModel}(im_problem::IM,solver::S,diffusion_model::D)

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
function solve() end

function solve(
        im_problem::IM,
        solver::S,
        diffusion_model::D
)::IMSolution where {
        IM <: AbstractTraditionalIMP,
        S <: AbstractIMSolution,
        D <: DiffusionModels.AbstractDiffusionModel
}
    throw(ArgumentError("The solver $(typeof(solver)) is not implemented for the problem $(typeof(im_problem)) with diffusion model $(typeof(diffusion_model))."))
end
