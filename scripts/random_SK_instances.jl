# script to generate 'hard' random SK instances
using QAOA, SpinFluctuations
using LinearAlgebra, Arpack, Random, Distributions, Printf, HDF5
using Dates
using PyPlot

PATH = "/home/ubuntu/Archives/"

# nev = 100
# N = 19
# keep_EVs = 3

nev = 80
N = 17
keep_EVs = 3

# nev = 70
# N = 15
# keep_EVs = 4

# nev=64
# N = 13
# keep_EVs = 5

# nev=50
# N = 11 
# keep_EVs = 5

# nev=32
# N = 9
# keep_EVs = 5

folder_name = PATH * @sprintf("data/SK_model/N_%i/", N)

# minigap cutoff conditions for saving
lower_cutoff = (minigap, data) -> (minigap < 0.01 && findfirst(x -> x == minigap, data) < length(data))
upper_cutoff = minigap -> minigap > 0.5 # 0.75

# command-line argument
loop_var = parse(Int, ARGS[1])

for seed in loop_var:loop_var+99
    printstyled(Dates.format(now(), "HH:MM") * ": seed = ", seed, "\n", color=:blue) 

    # create coupling matrix
    Random.seed!(seed)
    J = rand(Distributions.Normal(0, 1), N, N) ./ sqrt(N) 
    J[diagind(J)] .= 0.0
    J = UpperTriangular(J)
    J_mat = J + transpose(J)

    # problem instance
    mf_problem = Problem(0, J_mat)

    # get spectrum
    exact_times = range(0, 1, 33)
    eigeninfo = map(s -> (eigs(-SpinFluctuations.hamiltonian(1 - s, s, mf_problem.local_fields, mf_problem.couplings), nev=nev, which=:LM, maxiter=10000)), exact_times)
    λs = [vals[1] for vals in eigeninfo]
    λ = sort(reduce(hcat, λs), dims=1)

    all_eigvecs = zeros(length(exact_times), 2^(N-1), keep_EVs)
    for k in 1:length(exact_times)
        sorting_perm = sortperm(λs[k])
        all_eigvecs[k, :, :] .= eigeninfo[k][2][:, sorting_perm[1:keep_EVs]]
    end
    
    minigap = minimum(λ[2, :] .- λ[1, :])
    
    if lower_cutoff(minigap, λ[2, :] .- λ[1, :]) # || upper_cutoff(minigap)
        printstyled("\t", Dates.format(now(), "HH:MM") * ": Minigap is ", string(minigap), "\n", color=:green)
        printstyled("\t", Dates.format(now(), "HH:MM") * ": Saving...", "\n", color=:green)
        
        h5write(folder_name * @sprintf("random_SK_instance_N_%i_seed_%i.h5", N, seed), "J", J_mat)
        h5write(folder_name * @sprintf("random_SK_instance_N_%i_seed_%i.h5", N, seed), "exact_ARPACK_LM_eigvals", λ)
        h5write(folder_name * @sprintf("random_SK_instance_N_%i_seed_%i.h5", N, seed), "exact_ARPACK_LM_lowest_eigvecs", all_eigvecs)   
        printstyled("\t", Dates.format(now(), "HH:MM") * ": Spectrum done.", "\n", color=:green) 
    else 
        printstyled("\t", Dates.format(now(), "HH:MM") * ": Minigap is ", string(minigap), "\n", color=:red)
    end
end
