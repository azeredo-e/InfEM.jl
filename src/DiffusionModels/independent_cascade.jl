abstract type AbstractHistoryMode end

"""Store, for each epoch, only the final infection status (`BitVector`)."""
struct FinalStatus <: AbstractHistoryMode end

"""Store, for each epoch, the activation step of every node (`Inf` = never activated)."""
struct TimedCascade <: AbstractHistoryMode end

@doc INDEPENDENT_CASCADE_DOC
struct IndependentCascade{
    N <: Integer,
    D <: UnivariateDistribution,
    H <: AbstractHistoryMode
} <: AbstractDiffusionModel
    n_iters::N
    rng::D
    history::H
end

function IndependentCascade(; n_iters = STD_N_ITERS, rng = Uniform(0, 1),
        history = FinalStatus())
    return IndependentCascade(n_iters, rng, history)
end

# keeps the old two-argument constructor working
IndependentCascade(n_iters, rng) = IndependentCascade(n_iters, rng, FinalStatus())

# --- history helpers (one method per mode) ---------------------------------------------
init_history(::FinalStatus, n_iters) = Vector{BitVector}(undef, n_iters)
init_history(::TimedCascade, n_iters) = Vector{Vector{Float64}}(undef, n_iters)

init_times(::FinalStatus, n) = Float64[]         # never touched in this mode
init_times(::TimedCascade, n) = fill(Inf, n)

reset_times!(::FinalStatus, times) = nothing
reset_times!(::TimedCascade, times) = fill!(times, Inf)

mark_activated!(::FinalStatus, times, v, step) = nothing
mark_activated!(::TimedCascade, times, v, step) = (@inbounds times[v] = step; nothing)

function store_epoch!(history, epoch, ::FinalStatus, activated, times)
    @inbounds history[epoch] = copy(activated)
    return nothing
end
function store_epoch!(history, epoch, ::TimedCascade, activated, times)
    @inbounds history[epoch] = copy(times)
    return nothing
end

function run_diffusion_process(g::G,
        seed_nodes::VI,
        diffusion_model::IndependentCascade) where {
        T, G <: AbstractGraph{T}, VI <: AbstractVector{<:Integer}}
    n = nv(g)
    mode = diffusion_model.history
    total_activated = 0
    #! This will explode really fast... Problem for future me
    diffusion_history = init_history(mode, diffusion_model.n_iters)
    activated = falses(n)
    activation_time = init_times(mode, n)
    frontier = Vector{Int}(undef, n)
    next_frontier = Vector{Int}(undef, n)

    for epoch in 1:diffusion_model.n_iters
        fill!(activated, false)
        reset_times!(mode, activation_time)
        f_len = 0
        n_activated = 0
        step = 0                                   # seeds are activated at time 0
        for s in seed_nodes
            activated[s] = true
            mark_activated!(mode, activation_time, s, step)
            f_len += 1
            frontier[f_len] = s
            n_activated += 1
        end
        while f_len > 0
            nf_len = 0
            step += 1                              # everything reached now has time `step`
            for i in 1:f_len
                u = frontier[i]
                for v in outneighbors(g, u)
                    if !activated[v] && rand(diffusion_model.rng) < get_weight(g, u, v)
                        activated[v] = true
                        mark_activated!(mode, activation_time, v, step)
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
        store_epoch!(diffusion_history, epoch, mode, activated, activation_time)
    end

    return DiffusionResult(diffusion_history, total_activated / diffusion_model.n_iters)
end
