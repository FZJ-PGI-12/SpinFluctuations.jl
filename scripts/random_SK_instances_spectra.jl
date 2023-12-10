using QAOA, SpinFluctuations
using LinearAlgebra, Arpack, Random, Distributions, Printf, HDF5
using Dates, Crayons
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
# subdir = "large_gaps"
folder_name = PATH * @sprintf("data/SK_model/N_%i/%s/", N, subdir)
instance_names = readdir(folder_name)
filter!(x -> !occursin("results", x), instance_names)

for (k, instance_name) in enumerate(instance_names[loop_var:loop_var+99])
    seed = match(pattern, instance_name)[1]
    # printstyled(Dates.format(now(), "HH:MM") * ": ", instance_name, @sprintf(" is loop number %i", k), "\n", color=:blue)

# for (k, instance_name) in enumerate(instance_names)
#     seed = match(pattern, instance_name)[1]
#     if seed ∉ missing_seeds
#         continue
#     end
#     printstyled(Dates.format(now(), "HH:MM") * ": ", instance_name, @sprintf(" is loop number %i", k), "\n", color=:red)

    λ = h5read(folder_name * instance_name, "exact_ARPACK_LM_eigvals")
    gap = λ[2, :] .- λ[1, :]
    small_idxs = findall(x -> x < 0.1, gap) 
    mingap = minimum(gap) 

    exact_times = range(0, 1, 33)
    gaploc = exact_times[findfirst(x -> x == mingap, gap)] 
    # printstyled("\t", Dates.format(now(), "HH:MM") * ": Gap is located around ", gaploc, "\n", color=:blue)

    couplings = h5read(folder_name * instance_name, "J")
    mf_problem = Problem(0, couplings)

    T_final = 32768.
    tol = 1e-6

    # ======= Spectra =======

    # write to results file
    instance_name = "results_" * instance_name

    npts_diag = 16
    T_diags = T_final .* range(0.5, 1.0, npts_diag + 1)
    # T_diags = T_final .* [gaploc - 0.1, gaploc - 0.05, gaploc]

    # look at a few points leading up to the gap and one point after
    # T_diags = T_final .* exact_times[small_idxs[1:findfirst(x -> x == gap_idx, small_idxs) + 1]]

    # τ_final = 2048.
    τ_final = 8192.

    try
        h5read(folder_name * instance_name, @sprintf("spectra_T_final_%i_tau_final_%i/T_%0.5f/data", T_final, τ_final, T_diags[1] / T_final))
        printstyled(Dates.format(now(), "HH:MM") * ": ", instance_name, @sprintf(" is loop number %i", k), "\n", color=:light_green)
    catch
        printstyled(Dates.format(now(), "HH:MM") * ": ", instance_name, @sprintf(" is loop number %i", k), "\n", color=:light_red)
        printstyled("\t", Dates.format(now(), "HH:MM") * ": Getting spectral function...", "\n", color=:white)

        mf_sol, spectral_sums = evolve_spectral_sum(mf_problem, T_final, τ_final, T_diags, rtol=1e-2*tol, atol=tol)
        # h5write(folder_name * instance_name, @sprintf("mean_field_sol_T_final_%.0f_tol_1e%.0f", T_final, log10(tol)), mf_sol)
        
        for k in 1:length(T_diags)
            # printstyled("\t\t", Dates.format(now(), "HH:MM") * ": Getting spectrum at ", T_diags[k] / T_final, "\n", color=:blue)
            ωs, spec_sum = spectral_fft(spectral_sums[k])
            h5write(folder_name * instance_name, @sprintf("spectra_T_final_%i_tau_final_%i/T_%0.5f/omegas", T_final, τ_final, T_diags[k] / T_final), ωs)
            h5write(folder_name * instance_name, @sprintf("spectra_T_final_%i_tau_final_%i/T_%0.5f/data", T_final, τ_final, T_diags[k] / T_final), spec_sum)
        end
        printstyled("\t", Dates.format(now(), "HH:MM") * ": Spectra done.", "\n", color=:green)
    end   
end