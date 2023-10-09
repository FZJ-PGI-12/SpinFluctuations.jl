# script to generate 'hard' random SK instances
using QAOA, SpinFluctuations
using LinearAlgebra, Arpack, Random, Distributions, Printf, HDF5
using Dates
using PyPlot
# PyPlot.plt.style.use("./paper.mplstyle")

PATH = "/home/ubuntu/Archives/"

nev = 100
N = 19

# nev=80
# N = 17

# nev=64
# N = 15

# nev=64
# N = 13

# nev=50
# N = 11 

# nev=32
# N = 9

folder_name = PATH * @sprintf("data/SK_model/N_%i/", N)

# parameters for getting Lyapunov exponent
#
# final time for mean-field
T_final = 10000.

# number of points to get Lyapunov exponent for
npts = 512
npts = 256

# tolerance for DifferentialEquations.jl when solving mean-field 
tol = 1e-8 

# minigap cutoff conditions for saving
lower_cutoff = (minigap, data) -> (minigap < 0.01 && findfirst(x -> x == minigap, data) < length(data))
upper_cutoff = minigap -> minigap > 0.5 # 0.75

# command-line argument
loop_var = parse(Int, ARGS[1])

for seed in loop_var:loop_var+49
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
    λ = [vals[1] for vals in eigeninfo]
    λ = sort(reduce(hcat, λ), dims=1)
    final_eigvecs = eigeninfo[end][2]

    minigap = minimum(λ[2, :] .- λ[1, :])

    if lower_cutoff(minigap, λ[2, :] .- λ[1, :]) || upper_cutoff(minigap)
        printstyled("\t", Dates.format(now(), "HH:MM") * ": Minigap is ", string(minigap), "\n", color=:green)
        printstyled("\t", Dates.format(now(), "HH:MM") * ": Saving...", "\n", color=:green)

        h5write(folder_name * @sprintf("random_SK_instance_N_%i_seed_%i.h5", N, seed), "J", J_mat)
        h5write(folder_name * @sprintf("random_SK_instance_N_%i_seed_%i.h5", N, seed), "exact_ARPACK_LM_eigvals", λ)
        h5write(folder_name * @sprintf("random_SK_instance_N_%i_seed_%i.h5", N, seed), "exact_ARPACK_LM_final_eigvecs", final_eigvecs)
    
        # get max. Lyapunov exponent
        coarse_times = range(0, 1, npts + 1)
        lyapunov_parameters = LyapunovParameters(T_final, npts, 1e-2*tol, tol)
        
        mf_sol, lyapunov_exponent = maximal_lyapunov_exponent(mf_problem, lyapunov_parameters)

        h5write(folder_name * @sprintf("random_SK_instance_N_%i_seed_%i.h5", N, seed), @sprintf("lyapunov_exponent_T_final_%.0f_tol_1e%.0f_npts_%i", T_final, log10(tol), npts), lyapunov_exponent)
        printstyled("\t", Dates.format(now(), "HH:MM") * ": Lyapunov exponent done.", "\n", color=:green)

        # ===========================================================================================================================
        # Plotting

        figure(figsize=(7, 3))
        subplot(121)
        plot(coarse_times[2:end], lyapunov_exponent, "-C0")
        xlim(0, 1)
        ylim(0, )
        xlabel("\$s\$")
        ylabel("Lyapunov Exponent")

        subplot(122)
        for i in 1:size(λ)[1]
            plot(exact_times, λ[i, :] .- λ[1, :], "-k", ms=2)
        end
        xlim(0, 1)
        ylim(0, 4)
        xlabel("\$s\$")
        ylabel("Exact Eigenvalues")

        tight_layout()
        savefig(PATH * @sprintf("plots/SK_model/N_%i/", N) * @sprintf("random_SK_instance_N_%i_seed_%i.pdf", N, seed))
        close()

        printstyled("\t", Dates.format(now(), "HH:MM") * ": Plot done.", "\n", color=:green)
    else 
        printstyled("\t", Dates.format(now(), "HH:MM") * ": Minigap is ", string(minigap), "\n", color=:red)
    end
end
