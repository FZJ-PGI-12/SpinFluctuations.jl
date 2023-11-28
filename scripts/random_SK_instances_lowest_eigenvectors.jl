# script to generate 'hard' random SK instances
using QAOA, SpinFluctuations
using LinearAlgebra, Arpack, Random, Distributions, Printf, HDF5
using Dates

PATH = "/home/ubuntu/Archives/"

# nev = 100
# N = 19
# pattern = r"random_SK_instance_N_19_seed_(\d+)\.h5"
# keep_EVs = 3

# nev = 80
# N = 17
# pattern = r"random_SK_instance_N_17_seed_(\d+)\.h5"
# keep_EVs = 4

# nev = 64
# N = 15
# pattern = r"random_SK_instance_N_15_seed_(\d+)\.h5"
# keep_EVs = 5

# nev = 64
# N = 13
# pattern = r"random_SK_instance_N_13_seed_(\d+)\.h5"
# keep_EVs = 5

# nev = 50
# N = 11 
# pattern = r"random_SK_instance_N_11_seed_(\d+)\.h5"
# keep_EVs = 5

nev = 32
N = 9
pattern = r"random_SK_instance_N_9_seed_(\d+)\.h5"
keep_EVs = 5

subdir = "small_gaps"
# subdir = "large_gaps"
folder_name = PATH * @sprintf("data/SK_model/N_%i/%s/", N, subdir)
instance_names = readdir(folder_name)
filter!(x -> !occursin("results", x), instance_names)
filter!(x -> !occursin("undecided", x), instance_names)

# command-line argument
loop_var = parse(Int, ARGS[1])

for (k, instance_name) in enumerate(instance_names[loop_var:loop_var+99])
    try
        h5read(folder_name * instance_name, "exact_ARPACK_LM_lowest_eigvecs")
    catch
        seed = match(pattern, instance_name)[1]
        printstyled(Dates.format(now(), "HH:MM") * ": ", instance_name, @sprintf(" is loop number %i", k), "\n", color=:blue)
        printstyled("\t", Dates.format(now(), "HH:MM") * ": Getting eigenvectors...", "\n", color=:white)

        # problem instance
        J_mat = h5read(folder_name * instance_name, "J")
        mf_problem = Problem(0, J_mat)

        # get spectrum
        exact_times = range(0, 1, 33)
        eigeninfo = map(s -> (eigs(-SpinFluctuations.hamiltonian(1 - s, s, mf_problem.local_fields, mf_problem.couplings), nev=nev, which=:LM, maxiter=10000)), exact_times)
        λs = [vals[1] for vals in eigeninfo]

        all_eigvecs = zeros(length(exact_times), 2^(N-1), keep_EVs)
        for k in 1:length(exact_times)
            sorting_perm = sortperm(λs[k])
            all_eigvecs[k, :, :] .= eigeninfo[k][2][:, sorting_perm[1:keep_EVs]]
        end
        h5write(folder_name * instance_name, "exact_ARPACK_LM_lowest_eigvecs", all_eigvecs)
        printstyled("\t", Dates.format(now(), "HH:MM") * ": Eigenvectors done.", "\n", color=:green)
    end
end
