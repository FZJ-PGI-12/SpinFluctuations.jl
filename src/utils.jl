function spectral_fft(spectral_sum)
    ωs = spectral_sum[1]
    fft_f = fftshift(fft(spectral_sum[2]))
    spec = abs.(fft_f) ./ maximum(abs.(fft_f))
    ωs, spec
end

function spectral_sum(spectral_sol)
    τ_final = spectral_sol.t[end]
    npts = 2length(spectral_sol.t)
    τ_range = range(0, τ_final, npts+1)
    ωs = [(ω - 1 - npts/2)/τ_final for ω in 1:npts+1]

    # extrapolate to uniform grid and perform FFT
    sum_f = [map(τ -> spectral_sol(τ)[idx, idx], τ_range) for idx in 1:size(spectral_sol.u[1])[1]÷2]
    ωs, sum(sum_f)
end

# spectrum as a function of T for given spin index
function spectrum_vs_T(idx, spectral_sols)
    all_ωs = []
    all_specs = []
    for (k, spectral_sol) in enumerate(spectral_sols)
        ωs, spec = spectrum(spectral_sol, idx)
        push!(all_ωs, ωs)
        push!(all_specs, spec)    
    end
    (all_ωs, all_specs)
end    

# spectrum for fixed diagonal T and spin index idx
function spectrum(spectral_sol, idx)
    τ_final = spectral_sol.t[end]
    npts = 2length(spectral_sol.t)
    τ_range = range(0, τ_final, npts+1)
    ωs = [(ω - 1 - npts/2)/τ_final for ω in 1:npts+1]

    # extrapolate to uniform grid and perform FFT
    fft_f = fftshift(fft(map(τ -> spectral_sol(τ)[idx, idx], τ_range)))
    spec = abs.(fft_f) ./ maximum(abs.(fft_f))

    ωs, spec
end

function spectrum(ρ_T::Vector{ComplexF64}, lyapunov_parameters::LyapunovParameters)
    @unpack_LyapunovParameters lyapunov_parameters
    @assert size(ρ_T, 1) == npts+1

    ωs = [(ω - 1 - npts/2)/T_final for ω in 1:npts+1]
    fft_f = fftshift(fft(ρ_T))
    spec = abs.(fft_f) ./ maximum(abs.(fft_f))
    ωs, spec
end

function meshgrid(xin, yin)
    # Get the number of elements in the input arrays
    nx = length(xin)
    ny = length(yin)
    
    # Initialize output arrays with zeros
    xout = zeros(ny, nx)
    yout = zeros(ny, nx)
    
    # Loop over columns (jx) and rows (ix)
    for jx = 1:nx
        for ix = 1:ny
            # Assign values from input arrays to output arrays
            xout[ix, jx] = xin[jx]
            yout[ix, jx] = yin[ix]
        end
    end
    
    # Return a tuple of output arrays
    return (x = xout, y = yout)
end


function moving_average(vs, n)
    res = similar(vs, length(vs) - (n-1))
    @inbounds for i in 1:length(res)
        res[i] = sum(@view vs[i:(i + n-1)]) / n
    end
    return res
end


# function energies_and_probs(final_probs, annealing_problem)
#     L = annealing_problem.num_qubits
#     h = annealing_problem.local_fields
#     J = annealing_problem.couplings
    
#     bitstrings = [string(i, base=2, pad=L) |> reverse for i in 0:(2^L - 1)]
#     bitvals = [parse.(Int, [bitstring[j] for j in 1:L]) for bitstring in bitstrings]
#     spins = [1 .- 2s for s in bitvals]
    
#     bitstring_to_energy = Dict()
#     bitstring_to_probs = Dict()
#     for (i, spin) in enumerate(spins)
#         E = sum([-h[l] * spin[l] for l in 1:L]) + sum([-J[i, j] * spin[i] * spin[j] for i in 1:L for j in (i+1):L])
#         bitstring_to_energy[bitstrings[i]] = E
#         bitstring_to_probs[bitstrings[i]] = final_probs[i]
#     end

#     energy_to_bistring = Dict((val, []) for (key, val) in bitstring_to_energy)
#     for (key, val) in bitstring_to_energy
#         push!(energy_to_bistring[val], key)
#     end
    
#     energy_to_probs = Dict{Float64, Float64}()
#     for (key, vals) in energy_to_bistring
#         energy_to_probs[key] = sum([bitstring_to_probs[val] for val in vals])
#     end
    
#     E = collect(keys(energy_to_probs)) |> sort
#     probs = [energy_to_probs[en] for en in E]
#     E = E .- minimum(E)
#     return (E, probs)
# end


# function energy_from_max_prob(final_probs, annealing_problem)
#     L = annealing_problem.num_qubits
#     h = annealing_problem.local_fields
#     J = annealing_problem.couplings

#     max_prob = maximum(final_probs)
#     idxs = findfirst(x -> x == max_prob, final_probs)
#     bitstrings = [string.(idxs .- 1, base=2, pad=L) .|> reverse]
#     bitvals = [parse.(Int, [bitstring[j] for j in 1:L]) for bitstring in bitstrings]
#     spins = [1 .- 2s for s in bitvals]

#     bistrings_to_energy = Dict()
#     for (i, spin) in enumerate(spins)
#         E = sum([-h[l] * spin[l] for l in 1:L]) + sum([-J[i, j] * spin[i] * spin[j] for i in 1:L for j in (i+1):L])
#         bistrings_to_energy[bitstrings[i]] = E
#     end
    
#     return max_prob, bistrings_to_energy
# end


# function energies_and_bitstrings(annealing_problem)
#     L = annealing_problem.num_qubits
#     h = annealing_problem.local_fields
#     J = annealing_problem.couplings
    
#     bitstrings = [string(i, base=2, pad=L) |> reverse for i in 0:(2^L - 1)]
#     bitvals = [parse.(Int, [bitstring[j] for j in 1:L]) for bitstring in bitstrings]
#     spins = [1 .- 2s for s in bitvals]
    
#     bitstring_to_energy = Dict()
#     bitstring_to_probs = Dict()
#     for (i, spin) in enumerate(spins)
#         E = sum([-h[l] * spin[l] for l in 1:L]) + sum([-J[i, j] * spin[i] * spin[j] for i in 1:L for j in (i+1):L])
#         bitstring_to_energy[bitstrings[i]] = E
#     end

#     energy_to_bistring = Dict((val, []) for (key, val) in bitstring_to_energy)
#     for (key, val) in bitstring_to_energy
#         push!(energy_to_bistring[val], key)
#     end
    
#     return (bitstring_to_energy, energy_to_bistring)
# end