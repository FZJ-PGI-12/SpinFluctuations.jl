module SpinFluctuations

using QAOA, Yao, DifferentialEquations
using FFTW, Interpolations, LinearAlgebra, Arpack, SparseArrays
using Parameters

include("parameters.jl")
export LyapunovParameters

include("mean_field.jl")
export magnetization, evolve_mean_field

include("fluctuations.jl")
export bogoliubov_spectrum, evolve_spectral_sum, evolve_spectral_function, evolve_statistical_function, statistical_green_function, maximal_lyapunov_exponent

include("sparse_hamiltonian.jl")
export hamiltonian, exact_gap

include("utils.jl")
export spectral_fft, spectral_sum, spectrum_vs_T, spectrum, meshgrid, smoothen, moving_average

end
