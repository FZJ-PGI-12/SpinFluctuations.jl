using QAOA, SpinFluctuations
using LinearAlgebra, Arpack, Random, Distributions, Printf, HDF5
using Dates, Crayons
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

missing_seeds = ["9943"]

# for (k, instance_name) in enumerate(instance_names[loop_var:loop_var+24])
for (k, instance_name) in enumerate(instance_names)
    seed = match(pattern, instance_name)[1]
    if seed ∉ missing_seeds
        continue
    end
    printstyled(Dates.format(now(), "HH:MM") * ": ", instance_name, @sprintf(" is loop number %i", k), "\n", color=:blue)
    printstyled("\t", Dates.format(now(), "HH:MM") * ": Getting mean-field...", "\n", color=:white)

    couplings = h5read(folder_name * instance_name, "J")
    mf_problem = Problem(0, couplings)

    T_final = 32768.
    tol = 1e-6

    schedule(t) = t / T_final
    sol = evolve_mean_field(mf_problem.local_fields, mf_problem.couplings, T_final, schedule, rtol=1e2tol, atol=tol) 
    
    # get mean-field solution
    solution = S -> sign.([S[3, i] for i in 1:size(S)[2]])
    mf_sol = solution(sol(T_final)) 

    # write to results file
    instance_name = "results_" * instance_name

    writable_data = zeros(length(sol.u), size(sol.u[1])...)
    for i in 1:length(sol.u)
        writable_data[i, :, :] .= sol.u[i]
    end
    h5write(folder_name * instance_name, @sprintf("mean_field_sol_T_final_%.0f_tol_1e%.0f", T_final, log10(tol)), mf_sol)
    h5write(folder_name * instance_name, @sprintf("mean_field_T_final_%.0f_tol_1e%.0f/times", T_final, log10(tol)), sol.t)
    h5write(folder_name * instance_name, @sprintf("mean_field_T_final_%.0f_tol_1e%.0f/trajectories", T_final, log10(tol)), writable_data)
    printstyled("\t", Dates.format(now(), "HH:MM") * ": Mean-field trajectories done.", "\n", color=:green)
end