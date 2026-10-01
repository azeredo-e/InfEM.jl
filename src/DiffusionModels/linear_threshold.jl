struct LinearThreshold <: AbstractDiffusionModel end

function run_diffusion_process(
        g::G,
        seed_nodes::VI,
        diffusion::LinearThreshold
) where {
        G <: AbstractGraph,
        VI <: AbstractVector{<:Integer}
}
    throw("Linear Threshold diffusion model is not implemented yet.")
end
