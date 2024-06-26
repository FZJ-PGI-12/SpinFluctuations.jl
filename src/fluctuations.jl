"""
    fluctuation_matrix(problem::Problem, S::Vector{<:Vector{<:Real}}, solutions::Vector{<:Real}, β::Real, γ::Real)

Returns the Gaussian fluctuation matrix for a given point along the mean-field trajectories. 
"""
function fluctuation_matrix(problem::Problem, S::Matrix{<:Real}, solutions::Vector{<:Real}, β::Real, γ::Real)
    num_qubits = problem.num_qubits
    
    A = zeros(ComplexF64, (num_qubits, num_qubits))
    B = zeros(ComplexF64, (num_qubits, num_qubits))
    τ_3 = diagm(vcat(ones(num_qubits), -ones(num_qubits)))

    # helper function to construct A and B    
    n_ij_pm = (idx, pm) -> solutions[idx] * S[1, idx] + pm * 1im * S[2, idx]

    for i in 1:num_qubits
        # we exclude a factor of 2 here because we symmetrize below
        # signs in front of β, γ are reversed relative to the original paper
        A[i, i] = -β * S[1, i] / (1 + solutions[i] * S[3, i]) - γ * solutions[i] * magnetization(S, problem.local_fields, problem.couplings)[i]
        for j in i + 1:num_qubits
            A[i, j] = γ * problem.couplings[i, j] * n_ij_pm(i, 1) * n_ij_pm(j, -1)
            B[i, j] = γ * problem.couplings[i, j] * n_ij_pm(i, 1) * n_ij_pm(j,  1)        
        end
    end

    # symmetrize
    A += transpose(conj.(A))
    B += transpose(B)

    L = τ_3 * [A                   B;
               transpose(conj.(B)) conj.(A)]

    L
end

function bogoliubov_spectrum(problem::Problem, lyapunov_parameters::LyapunovParameters)
    @unpack_LyapunovParameters lyapunov_parameters
    
    # evolution
    schedule(t) = t / T_final
    sol = evolve_mean_field(problem.local_fields, problem.couplings, T_final, schedule, rtol=rtol, atol=atol)  
    
    # coarse times for the transfer matrix
    # (sufficient to capture the relevant low frequencies)
    times = range(0, T_final, npts + 1)
    Δt = times[2] - times[1]    
    
    # solution (rounded S_z values)
    solution = S -> sign.([S[3, i] for i in 1:size(S)[2]])
    solutions = solution(sol(T_final)) 
    
    bogol_spec = []
    for (k, t) in enumerate(times[2:end])        
        L = fluctuation_matrix(problem, sol(t), solutions, 1 - schedule(t), schedule(t))   
        push!(bogol_spec, eigvals(L))
    end
    bogol_spec
end

# ======================================================================================================================================
# statistical-function methods

function statistical_green_function(problem::Problem, lyapunov_parameters::LyapunovParameters)
    @unpack_LyapunovParameters lyapunov_parameters
    
    # evolution
    schedule(t) = t / T_final
    sol = evolve_mean_field(problem.local_fields, problem.couplings, T_final, schedule, rtol=rtol, atol=atol)  
    
    # coarse times for the transfer matrix
    # (sufficient to capture the relevant low frequencies)
    times = range(0, T_final, npts + 1)
    Δt = times[2] - times[1]    
    
    # solution (rounded S_z values)
    solution = S -> sign.([S[3, i] for i in 1:size(S)[2]])
    solutions = solution(sol(T_final)) 
    
    F_0 = Diagonal(vcat(-1.0im .* ones(problem.num_qubits), 1.0im .* ones(problem.num_qubits))) |> Matrix
    F = [F_0 for _ in 1:npts+1]
    M = 1.0I(2problem.num_qubits)
    M_inv = 1.0I(2problem.num_qubits)
    
    for (k, t) in enumerate(times[2:end])        
        L = fluctuation_matrix(problem, sol(t), solutions, 1 - schedule(t), schedule(t))   
        M = exp(-1im .* Δt .* L) * M
        M_inv = M_inv * exp(1im .* Δt .* L)        
        # M_inv = inv(M)

        # evolve GF
        F[k + 1] = M * F_0 * M_inv
    end

    sol, F
end


function evolve_statistical_function(problem::Problem, lyapunov_parameters::LyapunovParameters)
    @unpack_LyapunovParameters lyapunov_parameters
    
    # evolution
    schedule(t) = t / T_final
    sol = evolve_mean_field(problem.local_fields, problem.couplings, T_final, schedule, rtol=rtol, atol=atol)  
    
    # solution (rounded S_z values)
    solution = S -> sign.([S[3, i] for i in 1:size(S)[2]])
    solutions = solution(sol(T_final)) 
    
    # statistical equation of motion
    function statistical_eom(dF, F, p, t)
        L = fluctuation_matrix(problem, sol(t), solutions, 1 - schedule(t), schedule(t))
        # L = fluctuation_matrix(problem, sol(T_final * t), solutions, 1 - schedule(T_final * t), schedule(T_final * t))
        dF .= -1.0im .* (L * F - F * L)
    end    
    
    # solve with DifferentialEquations.jl on the forward-time diagonal
    statistical_sols = []
    F0 = Diagonal(vcat(-1.0im .* ones(problem.num_qubits), 1.0im .* ones(problem.num_qubits))) |> Matrix
    prob = ODEProblem(statistical_eom, F0, (0.0, T_final))
    # prob = ODEProblem(statistical_eom, F0, (0.0, 1.0))
    statistical_sol = solve(prob, Tsit5(), reltol=rtol, abstol=atol)
    push!(statistical_sols, statistical_sol)

    statistical_sols
end

# ======================================================================================================================================
# spectral-function methods

# less memory-intensive version of the spectral evolution
# the mean-field solution needs to be "padded" because the τ-dynamics walks "outside" of the usual (t, t') square
function evolve_spectral_sum(problem::Problem, T_final::Real, τ_final::Real, T_range; rtol=1e-4, atol=1e-6)
    # if s is smaller than zero, return zero
    # else, if s is bigger than one, return one
    # otherwise return s
    padded_schedule(s) = s < 0. ? 0. : (s > 1. ? 1. : s)
    
    # evolve beyond T_final to get all necessary values for τ-dynamics
    sol = evolve_mean_field(problem.local_fields, problem.couplings, T_final + τ_final/2, t -> padded_schedule(t / T_final), rtol=rtol, atol=atol)
    
    # for negative t, return initial values (there is no evolution anyway)
    # otherwise, return solution
    padded_sol(t) = t < 0. ? sol(0.) : sol(t)
    
    # get mean-field solution
    solution = S -> sign.([S[3, i] for i in 1:size(S)[2]])
    solutions = solution(sol(T_final)) 
    
    # spectral equation of motion
    function spectral_eom(dρ, ρ, T, τ)
        x_p = padded_schedule((T + τ / 2) / T_final)
        x_m = padded_schedule((T - τ / 2) / T_final)

        L_p = fluctuation_matrix(problem, padded_sol(T + τ / 2), solutions, 1 - x_p, x_p)
        L_m = fluctuation_matrix(problem, padded_sol(T - τ / 2), solutions, 1 - x_m, x_m)

        dρ .= -0.5im .* (L_p * ρ + ρ * L_m)
    end    
    
    # solve with DifferentialEquations.jl for npts_diag points on the forward-time diagonal
    spectral_sums = []
    ρ0 = Diagonal(vcat(-1.0im .* ones(problem.num_qubits), 1.0im .* ones(problem.num_qubits))) |> Matrix
    for T_diag in T_range
        prob = ODEProblem(spectral_eom, ρ0, (0.0, τ_final), T_diag)
        spectral_sol = solve(prob, Tsit5(), reltol=rtol, abstol=atol)
        push!(spectral_sums, spectral_sum(spectral_sol))
    end    
    solutions, spectral_sums
end

# the mean-field solution needs to be "padded" because the τ-dynamics walks "outside" of the usual (t, t') square
function evolve_spectral_function(problem::Problem, T_final::Real, τ_final::Real, T_range; rtol=1e-4, atol=1e-6)
    # if s is smaller than zero, return zero
    # else, if s is bigger than one, return one
    # otherwise return s
    padded_schedule(s) = s < 0. ? 0. : (s > 1. ? 1. : s)
    
    # evolve beyond T_final to get all necessary values for τ-dynamics
    sol = evolve_mean_field(problem.local_fields, problem.couplings, T_final + τ_final/2, t -> padded_schedule(t / T_final), rtol=rtol, atol=atol)
    
    # for negative t, return initial values (there is no evolution anyway)
    # otherwise, return solution
    padded_sol(t) = t < 0. ? sol(0.) : sol(t)
    
    # get mean-field solution
    solution = S -> sign.([S[3, i] for i in 1:size(S)[2]])
    solutions = solution(sol(T_final)) 
    
    # spectral equation of motion
    function spectral_eom(dρ, ρ, T, τ)
        x_p = padded_schedule((T + τ / 2) / T_final)
        x_m = padded_schedule((T - τ / 2) / T_final)

        L_p = fluctuation_matrix(problem, padded_sol(T + τ / 2), solutions, 1 - x_p, x_p)
        L_m = fluctuation_matrix(problem, padded_sol(T - τ / 2), solutions, 1 - x_m, x_m)

        dρ .= -0.5im .* (L_p * ρ + ρ * L_m)
    end    
    
    # solve with DifferentialEquations.jl for npts_diag points on the forward-time diagonal
    spectral_sols = []
    ρ0 = Diagonal(vcat(-1.0im .* ones(problem.num_qubits), 1.0im .* ones(problem.num_qubits))) |> Matrix
    for T_diag in T_range
        prob = ODEProblem(spectral_eom, ρ0, (0.0, τ_final), T_diag)
        spectral_sol = solve(prob, Tsit5(), reltol=rtol, abstol=atol)
        push!(spectral_sols, spectral_sol)
    end    
    solutions, spectral_sols
end

# ======================================================================================================================================
# old Lyapunov method

function maximal_lyapunov_exponent(problem::Problem, lyapunov_parameters::LyapunovParameters)
    @unpack_LyapunovParameters lyapunov_parameters
    
    # evolution
    schedule(t) = t / T_final
    sol = evolve_mean_field(problem.local_fields, problem.couplings, T_final, schedule, rtol=rtol, atol=atol)  
    
    # coarse times for the Lyapunov exponent
    # (sufficient to capture the relevant low frequencies)
    times = range(0, T_final, npts + 1)
    Δt = times[2] - times[1]    
    
    # solution (rounded S_z values)
    solution = S -> sign.([S[3, i] for i in 1:size(S)[2]])
    solutions = solution(sol(T_final)) 
    
    max_lyapunov_exponent = [0. for _ in 1:size(times[2:end])[1]]
    M = 1.0I(2problem.num_qubits)
    
    # Δts = map(x -> x[2] - x[1], zip(sol.t[1:end-1], sol.t[2:end]))
    
    for (k, t) in enumerate(times[2:end])        
        L = fluctuation_matrix(problem, sol(t), solutions, 1 - schedule(t), schedule(t))   
        M = exp(-1im .* Δt .* L) * M

        # use Arpack to get largest eigenvalue only
        exp_lambda, _ = eigs(M * transpose(conj.(M)), nev=1, which=:LM, maxiter=512)
        
        lambda = log.((1.0 + 0.0im) * exp_lambda) ./ 2 .|> real
        max_lyapunov_exponent[k] = lambda[1]        
    end

    sol, max_lyapunov_exponent
end

