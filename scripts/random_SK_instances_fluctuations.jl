using QAOA, SpinFluctuations
using LinearAlgebra, Arpack, Random, Distributions, Printf, HDF5
using Dates
using PyPlot

loop_var = parse(Int, ARGS[1])
PATH = "/home/ubuntu/Archives/"

# N = 19
# pattern = r"random_SK_instance_N_19_seed_(\d+)\.h5"

# N = 17
# pattern = r"random_SK_instance_N_17_seed_(\d+)\.h5"

# N = 15
# pattern = r"random_SK_instance_N_15_seed_(\d+)\.h5"

# N = 13
# pattern = r"random_SK_instance_N_13_seed_(\d+)\.h5"

# N = 11 
# pattern = r"random_SK_instance_N_11_seed_(\d+)\.h5"

N = 9
pattern = r"random_SK_instance_N_9_seed_(\d+)\.h5"

subdir = "small_gaps"
# subdir = "large_gaps"
folder_name = PATH * @sprintf("data/SK_model/N_%i/%s/", N, subdir)
instance_names = readdir(folder_name)
filter!(x -> !occursin("results", x), instance_names)
filter!(x -> !occursin("undecided", x), instance_names)
filter!(x -> !occursin("frustrated", x), instance_names)

# missing_seeds = ["9943"]

for (k, instance_name) in enumerate(instance_names[loop_var:loop_var+99])
# for (k, instance_name) in enumerate(instance_names)
#     seed = match(pattern, instance_name)[1]
#     if seed ∉ missing_seeds
#         continue
#     end
    try
        h5read(folder_name * "results_" * instance_name, @sprintf("fluctuations_T_final_%.0f_tol_1e%.0f_npts_%i", 65536., log10(1e-8), 8192))
        printstyled(Dates.format(now(), "HH:MM") * " |> ", instance_name, " exists\n", color=:green)
    catch
        printstyled(Dates.format(now(), "HH:MM") * " |> ", instance_name, @sprintf(" is loop number %i", k), "\n", color=:white)

        # λ = h5read(folder_name * instance_name, "exact_ARPACK_LM_eigvals")
        # gap = λ[2, :] .- λ[1, :];
        # exact_times = range(0, 1, 33)
        # gaploc = exact_times[findfirst(x -> x == minimum(gap), gap)] 
        # printstyled("\t", Dates.format(now(), "HH:MM") * ": Gap is located around ", gaploc, "\n", color=:blue)

        couplings = h5read(folder_name * instance_name, "J")
        mf_problem = Problem(0, couplings)

        # write to results file
        instance_name = "results_" * instance_name

        # T_final = 32000.
        # T_final = 32768.
        T_final = 65536.
        tol = 1e-8

        # # Bogoliubov spectrum
        # npts_bogo = 32
        # bogo_spec = bogoliubov_spectrum(mf_problem, LyapunovParameters(T_final, npts_bogo, tol, tol))
        # bogo_spec = reduce(hcat, bogo_spec)
        # bogo_spec = sort(bogo_spec .|> real, dims=1)

        # h5write(folder_name * instance_name, @sprintf("bogoliubov_spectrum_T_final_%.0f_tol_1e%.0f_npts_%i", T_final, log10(tol), npts_bogo), bogo_spec)
        # printstyled("\t", Dates.format(now(), "HH:MM") * "|> Bogoliubov spectrum done.", "\n", color=:green)

        # statistical Green function
        # npts = 2048
        # npts = 4096
        npts = 8192
        coarse_times = range(0, 1, npts + 1)
        lyapunov_parameters = LyapunovParameters(T_final, npts, tol, tol)
        mf_sol, stat_GF = statistical_green_function(mf_problem, lyapunov_parameters)

        flucs = k -> (real.(1.0im .* diag(stat_GF[k])[1:mf_problem.num_qubits]) .- 1.0) ./ 2;
        all_flucs = reduce(hcat, map(flucs, 1:npts+1))

        h5write(folder_name * instance_name, @sprintf("fluctuations_T_final_%.0f_tol_1e%.0f_npts_%i", T_final, log10(tol), npts), all_flucs)
        printstyled("\t", Dates.format(now(), "HH:MM") * " |> Fluctuations done.", "\n", color=:green)
    end
end


