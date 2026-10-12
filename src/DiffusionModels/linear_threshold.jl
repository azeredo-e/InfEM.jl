struct LinearThreshold <: AbstractDiffusionModel end

function run_diffusion_process(
        g::G,
        seed_nodes::VI,
        diffusion_model::LinearThreshold
) where {
        T,
        G <: AbstractGraph{T},
        VI <: AbstractVector{<:Integer}
}
    throw("Linear Threshold diffusion model is not implemented yet.")
end
