function operator(num_qubits::Int, i::Int, O)
    σ_id = 1.0I(2) |> real  
    reduce(kron, vcat([σ_id for _ in 1:i-1], [O], [σ_id for _ in i+1:num_qubits]))
end

function operator(num_qubits::Int, i::Int, j::Int, O)
    @assert i != j
    mini = min(i, j)
    maxi = max(i, j)
    
    σ_id = 1.0I(2) |> real |> SparseMatrixCSC
    
    reduce(kron, vcat([σ_id for _ in 1:mini-1], [O], [σ_id for _ in mini+1:maxi-1], [O], [σ_id for _ in maxi+1:num_qubits]))
end

function hamiltonian(β, γ, local_fields, couplings)
    σ_x = X |> Matrix |> real |> SparseMatrixCSC
    σ_z = Z |> Matrix |> real |> SparseMatrixCSC
    
    num_qubits = size(local_fields)[1]
    
    H  = γ .* sum([local_fields[i] * operator(num_qubits, i, σ_z) for i in 1:num_qubits])
    H += γ .* sum([couplings[i, j] * operator(num_qubits, i, j, σ_z) for j in 1:num_qubits for i in 1:j-1])
    H += β .* sum([operator(num_qubits, i, σ_x) for i in 1:num_qubits])
    H
end