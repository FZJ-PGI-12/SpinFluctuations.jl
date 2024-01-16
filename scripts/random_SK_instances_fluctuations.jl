using QAOA, SpinFluctuations
using LinearAlgebra, Arpack, Random, Distributions, Printf, HDF5
using Dates
using PyPlot

loop_var = parse(Int, ARGS[1])
PATH = "/home/ubuntu/Archives/"

N = 19
pattern = r"random_SK_instance_N_19_seed_(\d+)\.h5"

# N = 17
# pattern = r"random_SK_instance_N_17_seed_(\d+)\.h5"

# N = 15
# pattern = r"random_SK_instance_N_15_seed_(\d+)\.h5"

# N = 13
# pattern = r"random_SK_instance_N_13_seed_(\d+)\.h5"

# N = 11 
# pattern = r"random_SK_instance_N_11_seed_(\d+)\.h5"

# N = 9
# pattern = r"random_SK_instance_N_9_seed_(\d+)\.h5"

subdir = "small_gaps"
# missing_seeds = string.([1302, 1315, 1336, 1359, 1400, 1402, 1430, 1456, 1478, 1493, 1498, 1509, 1515, 1521, 1523, 1542, 1557, 1603, 1621, 1624, 1633, 1656, 1659, 1666, 1683, 1688, 1690, 1691, 1695, 1726, 1750, 1760])

# subdir = "large_gaps"
# missing_seeds = string.([1367, 1369, 1379, 1416, 1447, 1552])

folder_name = PATH * @sprintf("data/SK_model/N_%i/%s/", N, subdir)
instance_names = readdir(folder_name)
filter!(x -> !occursin("results", x), instance_names)
filter!(x -> !occursin("undecided", x), instance_names)
filter!(x -> !occursin("frustrated", x), instance_names)
filter!(x -> !occursin("main_df", x), instance_names)

for (k, instance_name) in enumerate(instance_names[loop_var:loop_var+9])
# for (k, instance_name) in enumerate(instance_names)
    # seed = match(pattern, instance_name)[1]
    # if seed ∉ missing_seeds
    #     continue
    # end
    try
        h5read(folder_name * "results_" * instance_name, @sprintf("fluctuations_T_final_%.0f_tol_1e%.0f_npts_%i", 32000., log10(1e-8), 2048))
        printstyled(Dates.format(now(), "HH:MM") * " |> ", instance_name, " exists\n", color=:green)
    catch
        printstyled(Dates.format(now(), "HH:MM") * " |> ", instance_name, @sprintf(" is loop number %i", k), "\n", color=:white)

        couplings = h5read(folder_name * instance_name, "J")
        mf_problem = Problem(0, couplings)

        # write to results file
        instance_name = "results_" * instance_name

        T_final = 32000.
        # T_final = 32768.
        # T_final = 65536.
        tol = 1e-8

        # Bogoliubov spectrum
        npts_bogo = 32
        bogo_spec = bogoliubov_spectrum(mf_problem, LyapunovParameters(T_final, npts_bogo, tol, tol))
        bogo_spec = reduce(hcat, bogo_spec)
        bogo_spec = sort(bogo_spec .|> real, dims=1)

        h5write(folder_name * instance_name, @sprintf("bogoliubov_spectrum_T_final_%.0f_tol_1e%.0f_npts_%i", T_final, log10(tol), npts_bogo), bogo_spec)
        printstyled("\t", Dates.format(now(), "HH:MM") * "|> Bogoliubov spectrum done.", "\n", color=:green)

        # statistical Green function
        npts = 2048
        # npts = 4096
        # npts = 8192
        coarse_times = range(0, 1, npts + 1)
        lyapunov_parameters = LyapunovParameters(T_final, npts, tol, tol)
        mf_sol, stat_GF = statistical_green_function(mf_problem, lyapunov_parameters)

        flucs = k -> (real.(1.0im .* diag(stat_GF[k])[1:mf_problem.num_qubits]) .- 1.0) ./ 2;
        all_flucs = reduce(hcat, map(flucs, 1:npts+1))

        h5write(folder_name * instance_name, @sprintf("fluctuations_T_final_%.0f_tol_1e%.0f_npts_%i", T_final, log10(tol), npts), all_flucs)
        printstyled("\t", Dates.format(now(), "HH:MM") * " |> Fluctuations done.", "\n", color=:green)
    end
end


