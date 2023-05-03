# script to generate 'hard' random SK instances
using QAOA, AdaptiveQuantumAnnealing, LinearAlgebra, Arpack, Random, Distributions, Printf, HDF5, PyPlot

PATH = "/Users/t.bode/.julia/dev/HardInstanceGenerator/"
N = 20

folder_name = PATH * @sprintf("data/SK_model/N_%i/", N)

for seed in 1:500
    printstyled("seed = ", seed, "\n", color=:green) 
    Random.seed!(seed)
    J = rand(Distributions.Normal(0, 1), N, N) ./ sqrt(N) 
    J[diagind(J)] .= 0.0
    J = UpperTriangular(J)
    J_mat = J + transpose(J)

    mf_problem = Problem(0, J_mat)

    # # do mean-field computations

    # h5write(folder_name * @sprintf("random_SK_instance_N_%i_seed_%i.h5", N, seed), "J", J_mat)

    # T_final = 30000.
    # tol = 1e-9
    # adap_times, solution, lyapunov_exponent = maximal_lyapunov_exponent(mf_problem, T_final, rtol=1e2*tol, atol=tol)

    # h5write(folder_name * @sprintf("random_SK_instance_N_%i_seed_%i.h5", N, seed), "times", adap_times)
    # h5write(folder_name * @sprintf("random_SK_instance_N_%i_seed_%i.h5", N, seed), "mean_field_solution", solution)
    # h5write(folder_name * @sprintf("random_SK_instance_N_%i_seed_%i.h5", N, seed), "lyapunov_exponent", lyapunov_exponent)

    # figure(figsize=(4, 3))
    # subplot(111)
    # plot(adap_times[2:end] ./ T_final, lyapunov_exponent, "-C0", label="λ")
    # xlim(0, 1)
    # ylim(0, )
    # xlabel("s")
    # legend(frameon=false)
    # tight_layout()
    # savefig(PATH * @sprintf("plots/SK_model/N_%i/", N) * @sprintf("random_SK_instance_N_%i_seed_%i.pdf", N, seed))
    # close()

    # get spectrum
    exact_times = range(0, 1, 26)
    eigeninfo = map(s -> (eigs(-AdaptiveQuantumAnnealing.hamiltonian(1 - s, s, mf_problem.local_fields, mf_problem.couplings), nev=8, which=:LM, maxiter=300)), exact_times)
    λ = [vals[1] for vals in eigeninfo]
    λ = reduce(hcat, λ)
    final_eigvecs = eigeninfo[end][2]
    h5write(folder_name * @sprintf("random_SK_instance_N_%i_seed_%i.h5", N, seed), "exact_eigvals", λ)
    h5write(folder_name * @sprintf("random_SK_instance_N_%i_seed_%i.h5", N, seed), "exact_final_eigvecs", final_eigvecs)
end