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

