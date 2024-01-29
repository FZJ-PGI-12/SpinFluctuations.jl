using QAOA
using HDF5, Printf
using Revise, SpinFluctuations

# ================================================================================================

PATH = "/home/ubuntu/Archives/"

N = parse(Int, ARGS[1])

patterns_dict = Dict(
    9  => r"random_SK_instance_N_9_seed_(\d+)\.h5",
    11 => r"random_SK_instance_N_11_seed_(\d+)\.h5",
    13 => r"random_SK_instance_N_13_seed_(\d+)\.h5",
    15 => r"random_SK_instance_N_15_seed_(\d+)\.h5",
    17 => r"random_SK_instance_N_17_seed_(\d+)\.h5",
    19 => r"random_SK_instance_N_19_seed_(\d+)\.h5"
)

T_final = 32768.
tol = 1e-6
npts = 2048
coarse_times = range(0, 1, npts + 1)

subdir = "small_gaps"
# subdir = "large_gaps"
folder_name = PATH * @sprintf("data/SK_model/N_%i/%s/", N, subdir)
instance_names = readdir(folder_name)
filter!(x -> !occursin("results", x), instance_names)
filter!(x -> !occursin("undecided", x), instance_names)
filter!(x -> !occursin("frustrated", x), instance_names)
filter!(x -> !occursin("main_df", x), instance_names)

printstyled(Dates.format(now(), "HH:MM") * "|> Loading up...", "\n", color=:red) 

# ================================================================================================

all_most_frustrated_spins = Dict()
all_most_undecided_spins = Dict()

for (k, instance_name) in enumerate(instance_names)
    seed = match(patterns_dict[N], instance_name)[1]  
    print(k, "\t")  
    
    # Mean-field trajectories
    sol_t = h5read(folder_name * "results_" * instance_name, @sprintf("mean_field_T_final_%.0f_tol_1e%.0f/times", T_final, log10(tol)))
    sol_u = h5read(folder_name * "results_" * instance_name, @sprintf("mean_field_T_final_%.0f_tol_1e%.0f/trajectories", T_final, log10(tol)))

    # mean-field
    J_mat = h5read(folder_name * instance_name, "J")
    mf_problem = Problem(0, J_mat)

    # From area under magnetization
    nz_vals = n_vals("z", sol_u)
    nxy_coarse = zeros(N-1)
    nz_coarse = n_coarse(nz_vals, sol_t, coarse_times)

    S_vals = [transpose(reduce(hcat, [nxy_coarse, nxy_coarse, nz_coarse[:, k]])) |> Matrix for k in 1:npts+1]
    magnetizations = reduce(hcat, map(S -> magnetization(S, mf_problem.local_fields, mf_problem.couplings), S_vals));  

    # Get most frustrated spins from area under magnetization
    areas = Dict()
    dts = [(x[2] - x[1]) / T_final for x in zip(coarse_times[1:end-1], coarse_times[2:end])]
    for spin_idx in 1:N-1
        areas[spin_idx] = sum(dts .* magnetizations[spin_idx, 2:end]) |> abs
    end
    all_most_frustrated_spins[seed] = [k for (k, v) in sort(areas |> collect, by=x->x[2])]

    # From area under Edwards-Anderson order parameter
    nzs = reduce(hcat, [sol_u[k, 3, :] for k in 1:size(sol_u)[1]])

    # Get "most undecided spin" from area under z components
    EA_param = Dict()
    dts = [(x[2] - x[1]) / T_final for x in zip(sol_t[1:end-1], sol_t[2:end])]
    for spin_idx in 1:N-1
        EA_param[spin_idx] = sum(dts .* nzs[spin_idx, 2:end] .^ 2) |> abs
    end
    all_most_undecided_spins[seed] = [k for (k, v) in sort(EA_param |> collect, by=x->x[2])]    
end

h5write(folder_name * @sprintf("most_frustrated_spins_N_%i.h5", N), 
@sprintf("T_final_%.0f_tol_1e%.0f/seeds", T_final, log10(tol)), [k |> string for (k, v) in all_most_frustrated_spins])

h5write(folder_name * @sprintf("most_frustrated_spins_N_%i.h5", N), 
@sprintf("T_final_%.0f_tol_1e%.0f/spin_idxs", T_final, log10(tol)), reduce(hcat, [v for (k, v) in all_most_frustrated_spins]) |> transpose |> Matrix)


h5write(folder_name * @sprintf("most_undecided_spins_N_%i.h5", N), 
@sprintf("T_final_%.0f_tol_1e%.0f/seeds", T_final, log10(tol)), [s |> string for (s, v) in all_most_undecided_spins])

h5write(folder_name * @sprintf("most_undecided_spins_N_%i.h5", N), 
@sprintf("T_final_%.0f_tol_1e%.0f/spin_idxs", T_final, log10(tol)), reduce(hcat, [v for (s, v) in all_most_undecided_spins]) |> transpose |> Matrix)

printstyled(Dates.format(now(), "HH:MM") * "|> Done writing!", "\n", color=:green) 