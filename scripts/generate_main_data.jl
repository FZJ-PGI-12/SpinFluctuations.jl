using QAOA, Distributions, Interpolations
using DataFrames, Arrow, HDF5, Printf, Dates
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

npts = 2048
coarse_times = range(0, 1, npts + 1)
exact_times = range(0, 1, 33)

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

most_frustrated_spins_seeds = h5read(folder_name * @sprintf("most_undecided_spins_N_%i.h5", N), @sprintf("T_final_%.0f_tol_1e%.0f/seeds", 32768., log10(1e-6)))
most_frustrated_spins_idxs = h5read(folder_name * @sprintf("most_undecided_spins_N_%i.h5", N), @sprintf("T_final_%.0f_tol_1e%.0f/spin_idxs", 32768., log10(1e-6)));
most_frustrated_spins = Dict(zip(most_frustrated_spins_seeds, eachrow(most_frustrated_spins_idxs)))
df_filename = "main_df_undecided"

# most_frustrated_spins_seeds = h5read(folder_name * @sprintf("most_frustrated_spins_N_%i.h5", N), @sprintf("T_final_%.0f_tol_1e%.0f/seeds", 32768., log10(1e-6)))
# most_frustrated_spins_idxs = h5read(folder_name * @sprintf("most_frustrated_spins_N_%i.h5", N), @sprintf("T_final_%.0f_tol_1e%.0f/spin_idxs", 32768., log10(1e-6)));
# most_frustrated_spins = Dict(zip(most_frustrated_spins_seeds, eachrow(most_frustrated_spins_idxs)))
# df_filename = "main_df_frustrated"


# ================================================================================================
# LOADING

ordered_seeds = []

mean_fields = Dict()
all_magnetizations = Dict()
mean_scaled_flucs = Dict()
most_frustrated_flucs = Dict()

all_Hs = Dict()
all_eigenvals = Dict()
all_eigenstates = Dict()

for (k, instance_name) in enumerate(instance_names)
    seed = match(patterns_dict[N], instance_name)[1]    
    print(k, ", ")

    # eigenvalues and -vectors
    λ = h5read(folder_name * instance_name, "exact_ARPACK_LM_eigvals")
    all_eigvecs = h5read(folder_name * instance_name, "exact_ARPACK_LM_lowest_eigvecs")

    J_mat = h5read(folder_name * instance_name, "J")
    mf_problem = Problem(0, J_mat)

    # # mean-field solutions
    T_final = 32768.
    tol = 1e-6
    mf_sol = h5read(folder_name * "results_" * instance_name, @sprintf("mean_field_sol_T_final_%.0f_tol_1e%.0f", T_final, log10(tol)))
    sigma_star = sign.(mf_sol)
    h = mf_problem.local_fields
    J = mf_problem.couplings
    E_star = sum([-h[l] * sigma_star[l] for l in 1:N-1]) + sum([-J[i, j] * sigma_star[i] * sigma_star[j] for i in 1:N-1 for j in (i+1):N-1]) 
    
    # continue if mean-fields finds optimal solution
    if isapprox(E_star, λ[1, end], atol=1e-5)
        continue
    end
        
    sol_t = h5read(folder_name * "results_" * instance_name, @sprintf("mean_field_T_final_%.0f_tol_1e%.0f/times", T_final, log10(tol)))
    sol_u = h5read(folder_name * "results_" * instance_name, @sprintf("mean_field_T_final_%.0f_tol_1e%.0f/trajectories", T_final, log10(tol)))

    nx_coarse, ny_coarse, nz_coarse = [n_coarse(n_vals(xyz, sol_u), sol_t, coarse_times) for xyz in 1:3]
    mean_fields[seed] = [nx_coarse, ny_coarse, nz_coarse]
    S_vals = [transpose(reduce(hcat, [nx_coarse[:, k], ny_coarse[:, k], nz_coarse[:, k]])) |> Matrix for k in 1:npts+1]
    magnetizations = reduce(hcat, map(S -> magnetization(S, mf_problem.local_fields, mf_problem.couplings), S_vals))
    all_magnetizations[seed] = magnetizations    

    # fluctuations
    T_final = 32000
    tol = 1e-8
    all_flucs = h5read(folder_name * "results_" * instance_name, @sprintf("fluctuations_T_final_%.0f_tol_1e%.0f_npts_%i", T_final, log10(tol), npts))
    lyapunov_exponent = sum(all_flucs, dims=1)
    if sum(lyapunov_exponent) |> abs < 1e4 # discard non-converged ones

        scale_factors = [1 .+ abs.(complex_coordinate(i, mean_fields[seed]...)).^2 for i in 1:N-1]
        mean_scaled_flucs[seed] = mean([scale_factors[i].^2 .* all_flucs[i, :] for i in 1:N-1], dims=1)[1]

        # keep the two most frustrated spins
        most_frustrated_flucs[seed] = [all_flucs[most_frustrated_spins[seed][1], :], all_flucs[most_frustrated_spins[seed][2], :]]
    end

    # adiabatic theorem...
    H_x = SpinFluctuations.hamiltonian(1, 0, mf_problem.local_fields, mf_problem.couplings)
    H_z = SpinFluctuations.hamiltonian(0, 1, mf_problem.local_fields, mf_problem.couplings)
    all_Hs[seed] = [H_x, H_z]

    eigenstate(n) = [all_eigvecs[k, :, n] for k in 1:length(exact_times)]
    num_eig_vecs = size(all_eigvecs)[3]
    all_eigenstates[seed] = [eigenstate(n) for n in 1:num_eig_vecs]
    all_eigenvals[seed] = [λ[n, :] for n in 1:num_eig_vecs]

    push!(ordered_seeds, seed)
end

print("\n")

# ================================================================================================
# DATA EVALUATION

minigap_locs = Dict()
all_overlaps = Dict()
all_gaps = Dict()
all_frustrated_flucs = Dict()
all_inv_mags = Dict()
seeds_to_max_fracs = Dict()

for seed in ordered_seeds
    try
        λs = all_eigenvals[seed]
        mingap = minimum(λs[2] .- λs[1])
        gap_idx = findfirst(x -> x == mingap, λs[2] .- λs[1]) 
        gaploc = exact_times[gap_idx]
        minigap_locs[seed] = gaploc
        
        nx_coarse, ny_coarse, nz_coarse = mean_fields[seed]
        
        all_frustrated_flucs[seed] = []
        top_idx = most_frustrated_spins[seed][1]
        scale_factor = 1 .+ abs.(complex_coordinate(top_idx, mean_fields[seed]...)).^2
        push!(all_frustrated_flucs[seed], scale_factor.^2 .* most_frustrated_flucs[seed][1])

        sec_idx = most_frustrated_spins[seed][2]
        scale_factor = 1 .+ abs.(complex_coordinate(sec_idx, mean_fields[seed]...)).^2
        push!(all_frustrated_flucs[seed], scale_factor.^2 .* most_frustrated_flucs[seed][2])   
        
        H_x, H_z = all_Hs[seed]
        eigenstates = all_eigenstates[seed]
        
        overlap(n, H) = [eigenstates[n][k]' * H * eigenstates[1][k] for k in 1:length(exact_times)]
        
        all_overlaps[seed] = [abs.(overlap(n, H_z) .- overlap(n, H_x)) for n in 2:3]
        all_gaps[seed] = [1 ./ (λs[n] .- λs[1]) for n in 2:3]
        
        seeds_to_max_fracs[seed] = maximum(all_overlaps[seed][1] .* (all_gaps[seed][1]).^2)
    catch err
        println(err)
        println(seed)
    end
end

# ================================================================================================
# CREATE DATA FRAME AND WRITE WITH ARROW

seeds_and_max_fracs = sort([(k, v) for (k, v) in seeds_to_max_fracs], by=x->x[2]) |> reverse

main_df = DataFrame(seed=String[], minigap_locs=Float64[], 
                    eigvals=Vector[], eigstates=Vector[],
                    scaled_most_frustrated_flucs=Vector[], overlaps=Vector[], gaps=Vector[], mean_scaled_flucs=Vector[], 
                    mean_fields=Vector[], magnetizations=Matrix[])
for (seed, _) in seeds_and_max_fracs
    push!(main_df, [seed, minigap_locs[seed], 
                    all_eigenvals[seed], all_eigenstates[seed],
                    all_frustrated_flucs[seed], all_overlaps[seed], all_gaps[seed], mean_scaled_flucs[seed], 
                    mean_fields[seed], all_magnetizations[seed]])
end

printstyled(Dates.format(now(), "HH:MM") * "|> Writing " * folder_name * df_filename * ".arrow", "\n", color=:white) 
Arrow.write(folder_name * df_filename * ".arrow", main_df)
printstyled(Dates.format(now(), "HH:MM") * "|> Done writing " * folder_name * df_filename * ".arrow", "\n", color=:green) 