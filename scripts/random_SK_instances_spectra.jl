using QAOA, SpinFluctuations
using LinearAlgebra, Arpack, Random, Distributions, Printf, HDF5
using Dates
using PyPlot

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
folder_name = PATH * @sprintf("data/SK_model/N_%i/%s/", N, subdir)
instance_names = readdir(folder_name)

loop_var = parse(Int, ARGS[1])

for instance_name in instance_names[loop_var:loop_var+99]
    printstyled(Dates.format(now(), "HH:MM") * ": ", instance_name, "\n", color=:blue)
    seed = match(pattern, instance_name)[1]    

    λ = h5read(folder_name * instance_name, "exact_ARPACK_LM_eigvals")

    gap = λ[2, :] .- λ[1, :];
    exact_times = range(0, 1, 33)
    gaploc = exact_times[findfirst(x -> x == minimum(gap), gap)] 
    printstyled("\t", Dates.format(now(), "HH:MM") * ": Gap is located around ", gaploc, "\n", color=:blue)

    couplings = h5read(folder_name * instance_name, "J")
    mf_problem = Problem(0, couplings)

    T_final = 16000.
    T_final = 32000.
    tol = 1e-8

    # spectra
    # npts_diag = 20
    # T_diags = T_final .* range(0.5, 1.0, npts_diag+1)
    T_diags = T_final .* [gaploc - 0.1, gaploc - 0.05, gaploc]
    # τ_final = 1000.
    τ_final = 2000.
    # τ_final = 4000.
    spectral_sols = evolve_spectral_function(mf_problem, T_final, τ_final, T_diags)
    for k in 1:length(T_diags)
        printstyled("\t\t", Dates.format(now(), "HH:MM") * ": Getting spectrum at ", T_diags[k] / T_final, "\n", color=:blue)
        ωs, spec_sum = spectral_sum(spectral_sols[k])
        h5write(folder_name * instance_name, @sprintf("spectra_T_final_%i_tau_final_%i/T_%0.3f/omegas", T_final, τ_final, T_diags[k] / T_final), ωs)
        h5write(folder_name * instance_name, @sprintf("spectra_T_final_%i_tau_final_%i/T_%0.3f/data", T_final, τ_final, T_diags[k] / T_final), spec_sum)
    end    
    printstyled("\t", Dates.format(now(), "HH:MM") * ": Spectra done.", "\n", color=:green)
end


