function magnetization(S::Matrix{<:Real}, h::Vector{<:Real}, J::Matrix{<:Real})
    h + [sum([J[i, j] * S[3, j] for j in 1:size(S)[2]]) for i in 1:size(S)[2]]
end

function evolve_mean_field(h::Vector{<:Real}, J::Matrix{<:Real}, T_final::Float64, schedule::Function; rtol=1e-4, atol=1e-6)

    function mf_eom(dS, S, _, t)
        magnetization = h + [sum([J[i, j] * S[3, j] for j in 1:size(S)[2]]) for i in 1:size(S)[2]]
        dS .= reduce(hcat, [[-2 * schedule(t) * magnetization[i] * S[2, i], 
                             -2 * (1 - schedule(t)) * S[3, i] + 2 * schedule(t) * magnetization[i] * S[1, i],
                              2 * (1 - schedule(t)) * S[2, i]] for i in 1:size(S)[2]])
    end

    S₀ = reduce(hcat, [[1., 0., 0.] for _ in 1:size(h)[1]])
    prob = ODEProblem(mf_eom, S₀, (0.0, T_final))
    sol = solve(prob, Tsit5(), reltol=rtol, abstol=atol)
    sol
end

function complex_coordinate(spin_idx::Int, nx_coarse::Matrix{Float64}, ny_coarse::Matrix{Float64}, nz_coarse::Matrix{Float64})
    (nx_coarse[spin_idx, :] .+ 1.0im .*  ny_coarse[spin_idx, :]) ./ (1 .+ sign(nz_coarse[spin_idx, end]) .* nz_coarse[spin_idx, :])
end