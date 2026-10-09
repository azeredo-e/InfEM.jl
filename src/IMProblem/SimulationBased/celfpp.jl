
"""
    CELFpp{B <: Bool}(verbose::B = false)

Defines the CELF++ solver for Influence Maximization problems (IMPs). It can be passed to the `solve` function
to solve an IMP.

CELF++ is an evolution on the *CELF* algorithm. TOTO: Add more details about CELF++ and how it differs from CELF.

When running `solve` the optional `solution` field of the returned `IMSolution` will contain a `NamedTuple`
with the following fields:
- lookups::<:Integer: The number of times the diffusion process was run during the execution of the algorithm.

Based on the original CELF++ algorithm from Goyal et al. (2011).

# Arguments
- verbose::<:Bool=false: Verbose output.
"""
struct CELFpp{B <: Bool} <: AbstractIMSolver
    verbose::B
end
function CELFpp(; verbose = false)
    return CELFpp(verbose)
end

function solve(
    im_problem::IM,
    solver::CELFpp,
    diffusion_model::D
)::IMSolution where {
    IM <: AbstractTraditionalIMP,
    D <: DiffusionModels.AbstractDiffusionModel
}
    t0 = time()

    n = nv(im_problem.g)
    @assert 1 ≤ im_problem.k ≤ n "k must satisfy 1 ≤ k ≤ nv(g)"

    # Per-node fields
    mg1 = zeros(Float64, n)
    mg2 = zeros(Float64, n)
    prev_best = zeros(Int, n)
    flag = zeros(Int, n)

    # Algorithm state
    S = Int[]
    sizehint!(S, im_problem.k)
    last_seed = 0
    cur_best = 0
    lookups = 0

    Q = PriorityQueue{Int, Float64}(Base.Order.Reverse)
    pair_buf = Vector{Int}(undef, 2)
    single_seed = Vector{Int}(undef, 1)
    tmp_seeds = Vector{Int}(undef, n)

    len_S_cache = -1   # -1 = invalid / needs refresh

    for u in 1:n
        # mg1[u] = σ({u})
        single_seed[1] = u
        mg1[u] = run_diffusion_process(im_problem.g, single_seed, diffusion_model)
        lookups += 1

        prev_best[u] = cur_best # u.prev_best = cur_best 

        if cur_best == 0 # u.mg2 = σ({cur_best})
            mg2[u] = mg1[u]
        else
            pair_buf[1] = u
            pair_buf[2] = cur_best
            mg2[u] = run_diffusion_process(
                im_problem.g,
                pair_buf,
                diffusion_model
            ) - mg1[cur_best]
            lookups += 1
        end

        flag[u] = 0
        enqueue!(Q, u, mg1[u])

        if cur_best == 0 || mg1[u] > mg1[cur_best]
            cur_best = u
        end
    end

    spread_S = 0.0
    spread_S_valid = true

    while length(S) < im_problem.k # |S| < k
        u = peek(Q)[1] # u = top (root) element in Q

        if flag[u] == length(S) # if u.flag == |S|
            dequeue!(Q) # S ← S ∪ {u}; Q ← Q − {u}
            push!(S, u)
            last_seed = u # last_seed = u
            cur_best = 0
            spread_S_valid = false
            len_S_cache = -1

            solver.verbose &&
                @printf("  [CELF++] step %2d | node %4d | total lookups: %d\n",
                    length(S),
                    u,
                    lookups)
            continue
        elseif prev_best[u] == last_seed # else if u.prev_best == last_seed
            mg1[u] = mg2[u]
        else
            len_S = length(S)
            if len_S != len_S_cache
                copyto!(tmp_seeds, 1, S, 1, len_S)
                len_S_cache = len_S
            end
            if !spread_S_valid
                spread_S = run_diffusion_process(im_problem.g, S, diffusion_model)
                spread_S_valid = true
                lookups += 1
            end

            @inbounds tmp_seeds[len_S + 1] = u # u.mg1 = Δu(S) = σ(S ∪ {u}) − σ(S)
            mg1[u] = run_diffusion_process(
                im_problem.g,
                @view(tmp_seeds[1:(len_S + 1)]),
                diffusion_model
            ) - spread_S
            lookups += 1

            prev_best[u] = cur_best # u.prev_best = cur_best

            if cur_best == 0 # u.mg2 = Δu(S ∪ {cur_best}) = σ(S ∪ {u, cur_best}) − σ(S ∪ {cur_best})
                mg2[u] = mg1[u]
            else
                # σ(S ∪ {cur_best}): reuse S prefix, write cur_best at len_S+1
                @inbounds tmp_seeds[len_S + 1] = cur_best
                spread_S_cb = run_diffusion_process(
                    im_problem.g,
                    @view(tmp_seeds[1:(len_S + 1)]),
                    diffusion_model
                )

                # σ(S ∪ {u, cur_best}): write u at len_S+1, cur_best at len_S+2
                @inbounds tmp_seeds[len_S + 1] = u
                @inbounds tmp_seeds[len_S + 2] = cur_best
                mg2[u] = run_diffusion_process(
                    im_problem.g,
                    @view(tmp_seeds[1:(len_S + 2)]),
                    diffusion_model
                ) - spread_S_cb
                lookups += 2
            end
        end

        flag[u] = length(S) # u.flag = |S|; Update cur best.
        if cur_best == 0 || mg1[u] > mg1[cur_best]
            cur_best = u
        end

        Q[u] = mg1[u] # Reinsert u into Q and heapify
    end

    return IMSolution(
        im_problem.g,
        im_problem.k,
        diffusion_model,
        S,
        spread_S,
        NamedTuple{(:start, :end, :elapsed)}((t0, time(), time() - t0)),
        (lookups = lookups,)
    )
end
