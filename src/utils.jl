# function moving_average(vs, n)
#     [sum(@view vs[i:(i + n-1)]) / n for i in 1:(length(vs) - (n-1))]
# end

function moving_average(vs, n)
    res = similar(vs, length(vs) - (n-1))
    @inbounds for i in 1:length(res)
        res[i] = sum(@view vs[i:(i + n-1)]) / n
    end
    return res
end


function energies_and_probs(final_probs, annealing_problem)
    L = annealing_problem.num_qubits
    h = annealing_problem.local_fields
    J = annealing_problem.couplings
    
    bitstrings = [string(i, base=2, pad=L) |> reverse for i in 0:(2^L - 1)]
    bitvals = [parse.(Int, [bitstring[j] for j in 1:L]) for bitstring in bitstrings]
    spins = [1 .- 2s for s in bitvals]
    
    bitstring_to_energy = Dict()
    bitstring_to_probs = Dict()
    for (i, spin) in enumerate(spins)
        E = sum([-h[l] * spin[l] for l in 1:L]) + sum([-J[i, j] * spin[i] * spin[j] for i in 1:L for j in (i+1):L])
        bitstring_to_energy[bitstrings[i]] = E
        bitstring_to_probs[bitstrings[i]] = final_probs[i]
    end

    energy_to_bistring = Dict((val, []) for (key, val) in bitstring_to_energy)
    for (key, val) in bitstring_to_energy
        push!(energy_to_bistring[val], key)
    end
    
    energy_to_probs = Dict{Float64, Float64}()
    for (key, vals) in energy_to_bistring
        energy_to_probs[key] = sum([bitstring_to_probs[val] for val in vals])
    end
    
    E = collect(keys(energy_to_probs)) |> sort
    probs = [energy_to_probs[en] for en in E]
    E = E .- minimum(E)
    return (E, probs)
end


function energy_from_max_prob(final_probs, annealing_problem)
    L = annealing_problem.num_qubits
    h = annealing_problem.local_fields
    J = annealing_problem.couplings

    max_prob = maximum(final_probs)
    idxs = findfirst(x -> x == max_prob, final_probs)
    bitstrings = [string.(idxs .- 1, base=2, pad=L) .|> reverse]
    bitvals = [parse.(Int, [bitstring[j] for j in 1:L]) for bitstring in bitstrings]
    spins = [1 .- 2s for s in bitvals]

    bistrings_to_energy = Dict()
    for (i, spin) in enumerate(spins)
        E = sum([-h[l] * spin[l] for l in 1:L]) + sum([-J[i, j] * spin[i] * spin[j] for i in 1:L for j in (i+1):L])
        bistrings_to_energy[bitstrings[i]] = E
    end
    
    return max_prob, bistrings_to_energy
end


function energies_and_bitstrings(annealing_problem)
    L = annealing_problem.num_qubits
    h = annealing_problem.local_fields
    J = annealing_problem.couplings
    
    bitstrings = [string(i, base=2, pad=L) |> reverse for i in 0:(2^L - 1)]
    bitvals = [parse.(Int, [bitstring[j] for j in 1:L]) for bitstring in bitstrings]
    spins = [1 .- 2s for s in bitvals]
    
    bitstring_to_energy = Dict()
    bitstring_to_probs = Dict()
    for (i, spin) in enumerate(spins)
        E = sum([-h[l] * spin[l] for l in 1:L]) + sum([-J[i, j] * spin[i] * spin[j] for i in 1:L for j in (i+1):L])
        bitstring_to_energy[bitstrings[i]] = E
    end

    energy_to_bistring = Dict((val, []) for (key, val) in bitstring_to_energy)
    for (key, val) in bitstring_to_energy
        push!(energy_to_bistring[val], key)
    end
    
    return (bitstring_to_energy, energy_to_bistring)
end