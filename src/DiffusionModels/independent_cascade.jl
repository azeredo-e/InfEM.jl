
"""

"""
struct IndependentCascade{
    N <: Integer,
    D <: UnivariateDistribution
} <: AbstractDiffusionModel
    n_iters::N
    rng::D
end
function IndependentCascade(; n_iters = STD_N_ITERS, rng = Uniform(0, 1))
    IndependentCascade(n_iters, rng)
end

function run_diffusion_process(
    g::G,
    seed_nodes::VI,
    diffusion_model::IndependentCascade
) where {T, G <: AbstractGraph{T}, VI <: AbstractVector{<:Integer}}
    n = nv(g)
    total_activated = 0
    #! This will explode really fast... Problem for future me
    diffusion_history = Vector{BitVector}(undef, diffusion_model.n_iters)
    activated = falses(n)
    frontier = Vector{Int}(undef, n)
    next_frontier = Vector{Int}(undef, n)

    for epoch in 1:diffusion_model.n_iters
        fill!(activated, false)
        f_len = 0
        n_activated = 0
        for s in seed_nodes
            activated[s] = true
            f_len += 1
            frontier[f_len] = s
            n_activated += 1
        end
        while f_len > 0
            nf_len = 0

            for i in 1:f_len
                u = frontier[i]
                for v in outneighbors(g, u)
                    if !activated[v] && rand(diffusion_model.rng) < get_weight(g, u, v)
                        activated[v] = true
                        nf_len += 1
                        next_frontier[nf_len] = v
                        n_activated += 1
                    end
                end
            end
            frontier, next_frontier = next_frontier, frontier
            f_len = nf_len
        end

        total_activated += n_activated
        diffusion_history[epoch] = copy(activated)
    end

    return DiffusionResult(diffusion_history, total_activated / diffusion_model.n_iters)
end
