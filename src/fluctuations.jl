function magnetization(S::Matrix{<:Real}, h::Vector{<:Real}, J::Matrix{<:Real})
    h + [sum([J[i, j] * S[3, j] for j in 1:size(S)[2]]) for i in 1:size(S)[2]]
end

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


function statistical_green_function(problem::Problem, lyapunov_parameters::LyapunovParameters)
    @unpack_LyapunovParameters lyapunov_parameters
    
    # evolution
    schedule(t) = t / T_final
    sol = evolve(problem.local_fields, problem.couplings, T_final, schedule, rtol=rtol, atol=atol)  
    
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
        M_inv = inv(M)

        # evolve GF
        F[k + 1] = M * F_0 * M_inv
    end

    sol, F
end


# current method!
function maximal_lyapunov_exponent(problem::Problem, lyapunov_parameters::LyapunovParameters)
    @unpack_LyapunovParameters lyapunov_parameters
    
    # evolution
    schedule(t) = t / T_final
    sol = evolve(problem.local_fields, problem.couplings, T_final, schedule, rtol=rtol, atol=atol)  
    
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



function maximal_lyapunov_exponent(problem::Problem, T_final::Real, npts::Int; rtol=1e-4, atol=1e-6)
    
    # evolution
    schedule(t) = t / T_final
    sol = evolve(problem.local_fields, problem.couplings, T_final, schedule, rtol=rtol, atol=atol)  
    
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

    sol.t, solutions, max_lyapunov_exponent
end








function maximal_lyapunov_exponent(problem::Problem, T_final::Real; rtol=1e-4, atol=1e-6)
    
    # evolution
    schedule(t) = t / T_final
    sol = evolve(problem.local_fields, problem.couplings, T_final, schedule, rtol=rtol, atol=atol)  
    
    # solution (rounded S_z values)
    solution = S -> sign.([S[3, i] for i in 1:size(S)[2]])
    # solution = S -> [S[3, i] for i in 1:size(S)[2]]
    solutions = solution(sol(T_final)) 
    
    max_lyapunov_exponent = [0. for _ in 1:size(sol.t[2:end])[1]]
    M = 1.0I(2problem.num_qubits)
    
    Δts = map(x -> x[2] - x[1], zip(sol.t[1:end-1], sol.t[2:end]))
    
    for (k, t) in enumerate(sol.t[2:end])
        L = fluctuation_matrix(problem, sol(t), solutions, 1 - schedule(t), schedule(t))   
        
        # could one truncate this SVD style?
        omega_eig, omega_eigvec = eigen(L)
        
        M = omega_eigvec * diagm(exp.(-1im .* Δts[k] .* omega_eig)) * inv(omega_eigvec) * M

        # use Arpack to get largest eigenvalue only
        exp_lambda, _ = eigs(M * transpose(conj.(M)), nev=1, which=:LM, maxiter=512)
        
        lambda = log.((1.0 + 0.0im) * exp_lambda) ./ 2 .|> real
        max_lyapunov_exponent[k] = lambda[1]        
    end

    sol.t, solutions, max_lyapunov_exponent
end


"""
    evolve_fluctuations_full(problem::Problem, T_final::Real; rtol=1e-4, atol=1e-6)

### Input
- `problem::Problem`: A `Problem` instance defining the optimization problem.
-

### Output
- The Lyapunov exponents characterizing the dynamics of the Gaussian fluctuations.
"""
function evolve_fluctuations_full(problem::Problem, T_final::Real; rtol=1e-4, atol=1e-6)
    num_qubits = problem.num_qubits    
    
    # evolution
    schedule(t) = t / T_final
    sol = evolve(problem.local_fields, problem.couplings, T_final, schedule, rtol=rtol, atol=atol)  
    
    # solution (rounded S_z values)
    solution = S -> sign.([S[3, i] for i in 1:size(S)[2]])
    solutions = solution(sol(T_final)) 
    
    lyapunov_exponents = [zeros(2num_qubits) for _ in 1:size(sol.t[2:end])[1]]
    M = 1.0I(2num_qubits)
    
    Δts = map(x -> x[2] - x[1], zip(sol.t[1:end-1], sol.t[2:end]))
    
    for (k, t) in enumerate(sol.t[2:end])
        L = fluctuation_matrix(problem, sol(t), solutions, 1 - schedule(t), schedule(t))        
        omega_eig, omega_eigvec = eigen(L)
        
        M = omega_eigvec * diagm(exp.(-1im .* Δts[k] .* omega_eig)) * inv(omega_eigvec) * M

        lyapunov_exponential_eig = eigvals(M * transpose(conj.(M)))

        lyapunov_exponent_eig = log.((1.0 + 0.0im) * lyapunov_exponential_eig) ./ 2 .|> real
        lyapunov_exponent_eig = sort(lyapunov_exponent_eig)

        lyapunov_exponents[k] = lyapunov_exponent_eig        
    end

    sol.t, solutions, lyapunov_exponents
end