function lazy_forward(
        im_problem::IM,
        type::Symbol,
        diffusion_model::DM;
)::Vector{Int} where {
        IM <: AbstractTraditionalIMP,
        DM <: DiffusionModels.AbstractDiffusionModel
}
    n = nv(im_problem.g)
    A = Int[]
    sizehint!(A, n)
    δ = fill(Inf, n)
    cur = falses(n)
    in_A = falses(n)
    budget_used = 0.0
    tmp_seeds = Vector{Int}(undef, n)
    use_uc = (type === :UC)

    while true # while ∃ affordable s ∉ A
        # Quick check for performance: if no candidate is affordable, we can break early without recomputing
        # the candidates list
        any_candidate = false
        @inbounds for s in 1:n
            if !in_A[s] && budget_used + im_problem.costs[s] ≤ im_problem.k
                any_candidate = true
                break
            end
        end
        any_candidate || break

        fill!(cur, false) # foreach s ∈ V\A do  curs ← false
        R_A = isempty(A) ? 0.0 :
              run_diffusion_process(im_problem.g, A, diffusion_model).average_spread
        len_A = length(A)
        copyto!(tmp_seeds, 1, A, 1, len_A)

        while true
            s_star = 0
            best_p = -Inf
            @inbounds for s in 1:n
                (in_A[s] || budget_used + im_problem.costs[s] > im_problem.k) && continue
                p = use_uc ? δ[s] : δ[s] / im_problem.costs[s]
                if p > best_p
                    best_p = p
                    s_star = s
                end
            end

            if @inbounds cur[s_star]  # if curs*  then  A ← A ∪ {s*};  break
                push!(A, s_star)
                @inbounds in_A[s_star] = true
                budget_used += im_problem.costs[s_star]
                # budget_used += 1
                break
            else # else  δs* ← R(A ∪ {s*}) − R(A);  curs* ← true
                @inbounds tmp_seeds[len_A + 1] = s_star
                @inbounds δ[s_star] = run_diffusion_process(
                    im_problem.g,
                    @view(tmp_seeds[1:(len_A + 1)]),
                    diffusion_model
                ).average_spread - R_A
                @inbounds cur[s_star] = true
            end
        end
    end # while ∃ affordable candidate (outer loop)

    return A
end

struct CELF{B <: Bool} <: AbstractIMSolver
    verbose::B
end
function CELF(; verbose = false)
    return CELF(verbose)
end

function solve(
    im_problem::IM,
    solver::CELF,
    diffusion_model::D
)::IMSolution where {
    IM <: AbstractTraditionalIMP,
    D <: DiffusionModels.AbstractDiffusionModel
}
    t0 = time()
    solver.verbose && println("CELF ▸ LazyForward [UC] …")
    A_UC = lazy_forward(im_problem, :UC, diffusion_model)
    R_UC = (
        isempty(A_UC) ?
        0.0 :
        run_diffusion_process(im_problem.g, A_UC, diffusion_model).average_spread
    )

    # When budgeted algorithms are implemented this will 
    # Check if all costs are 1.0, if so, we can skip the CB strategy because it will be the same as UC
    if all(im_problem.costs .== 1.0)
        solver.verbose && println("CELF ▸ LazyForward [CB] …")
        A_CB = lazy_forward(im_problem, :CB, diffusion_model)
        R_CB = (
            isempty(A_CB) ?
            0.0 :
            run_diffusion_process(im_problem.g, A_CB, diffusion_model).average_spread
        )
    else
        A_CB = Int[]
        R_CB = 0.0
    end

    solution, spread, winner = R_UC ≥ R_CB ? (A_UC, R_UC, :UC) : (A_CB, R_CB, :CB)

    if solver.verbose
        @printf("  UC → nodes %-20s  spread = %.3f\n", string(A_UC), R_UC)
        # @printf("  CB → nodes %-20s  spread = %.3f\n", string(A_CB), R_CB)
        # @printf("  Winner: %s\n", winner)
    end

    return IMSolution(
        im_problem.g,
        im_problem.k,
        diffusion_model,
        solution,
        spread,
        NamedTuple{(:start, :end, :elapsed)}((t0, time(), time() - t0)),
        (winner = winner, A_UC = A_UC, R_UC = R_UC, A_CB = A_CB, R_CB = R_CB)
    )
end
