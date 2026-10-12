# @doc SOLVE_IEP_DOC
# function solve(
#         im_problem::IM,
#         solver::S,
#         diffusion_model::D
# )::IMSolution where {
#         IM <: AbstractTraditionalIMP,
#         S <: AbstractIMSolution,
#         D <: DiffusionModels.AbstractDiffusionModel
# }
#     throw(ArgumentError("The solver $(typeof(solver)) is not implemented for the problem $(typeof(im_problem)) with diffusion model $(typeof(diffusion_model))."))
# end
