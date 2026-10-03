"""
    Greedy{B<:Bool}(verbose::B)}

Defines the Greedy solver for Influence Maximization problems (IMPs). It can be passed to the `solve` function
to solve an IMP.

Based on the original greedy algorithm from Kempe et al. (2003).

# Arguments
- verbose::<:Bool=false: Verbose output.
"""
struct Greedy{B <: Bool} <: AbstractIMSolver
    verbose::B
end
function Greedy(; verbose = false)
    return Greedy(verbose)
end

function solve(
        im_problem::IM,
        solver::Greedy,
        diffusion_model::D
)::IMSolution where {
        IM <: AbstractTraditionalIMP,
        D <: DiffusionModels.AbstractDiffusionModel
}
    t0 = time()

    N = nv(im_problem.g)
    @assert 1 ≤ im_problem.k ≤ N "K must satisfy 1 ≤ K ≤ nv(g)  (got K=$(im_problem.k), N=$N)"

    in_S = falses(N)
    tmp_seeds = Vector{Int}(undef, N)

    # φ₀ = ∅
    S = Int[]
    sizehint!(S, im_problem.k)
    spreads = Float64[]
    sizehint!(spreads, im_problem.k)
    t0 = time()

    for q in 1:im_problem.k # for q = 1 to K
        len_S = length(S)
        spread_S = len_S == 0 ? 0.0 :
                   run_diffusion_process(im_problem.g, S, diffusion_model)
        copyto!(tmp_seeds, 1, S, 1, len_S)
        best_node = 0
        best_gain = -Inf

        # i = argmax_{j ∈ V\φ₀} {σ(φ₀ ∪ {j}) − σ(φ₀)}
        @inbounds for j in 1:N
            in_S[j] && continue # O(1) bit-test; original was O(|φ₀|) scan

            tmp_seeds[len_S + 1] = j
            gain = run_diffusion_process(
                im_problem.g,
                @view(tmp_seeds[1:(len_S + 1)]),
                diffusion_model
            ) - spread_S

            if gain > best_gain
                best_gain = gain
                best_node = j
            end
        end

        push!(S, best_node) # φ₀ = φ₀ ∪ {i}
        in_S[best_node] = true
        push!(spreads, spread_S + best_gain)

        solver.verbose &&
            @printf("  step %2d | node %4d | marginal gain %7.3f | σ(S) = %7.3f\n",
                q,
                best_node,
                best_gain,
                last(spreads))
    end

    return IMSolution(
        im_problem.g,
        im_problem.k,
        diffusion_model,
        S,
        last(spreads),
        NamedTuple{(:start, :end, :elapsed)}((t0, time(), time() - t0)),
        nothing
    )
end
