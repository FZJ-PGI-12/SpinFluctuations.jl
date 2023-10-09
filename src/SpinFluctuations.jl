module SpinFluctuations

using QAOA, DifferentialEquations
using FFTW, Interpolations, LinearAlgebra, Arpack, SparseArrays
using Parameters

include("parameters.jl")
export LyapunovParameters, MetricParameters, AdaptiveScheduleParameters

include("mean_field.jl")
export evolve_mean_field

include("fluctuations.jl")
export bogoliubov_spectrum, evolve_spectral_function, statistical_green_function, maximal_lyapunov_exponent, full_green_function, diagonal_spectral_functions, evolve_fluctuations_full

include("sparse_hamiltonian.jl")
export hamiltonian, exact_gap

include("utils.jl")
export spectral_sum, spectrum_vs_T, spectrum, meshgrid, moving_average, energies_and_probs, energy_from_max_prob, energies_and_bitstrings

end
