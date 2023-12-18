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


function smoothen(data, coarse_times; navg=64)
    ninterp = size(coarse_times)[1] - 1
    avg_data = linear_interpolation(range(0, 1, ninterp + 1)[1:end - navg + 1], moving_average(data, navg), extrapolation_bc=Line())
    map(x -> avg_data(x - navg / 2ninterp), coarse_times)
end


function moving_average(vs, n)
    res = similar(vs, length(vs) - (n-1))
    @inbounds for i in 1:length(res)
        res[i] = sum(@view vs[i:(i + n-1)]) / n
    end
    return res
end


# =================================================================================================================================
# mean-field and fluctuation helpers

function n_vals(xyz::Int, sol_u)
    reduce(hcat, [sol_u[k, xyz, :] for k in 1:size(sol_u)[1]])
end

function n_vals(xyz::String, sol_u)
    component_dict = Dict("x" => 1, "y" => 2, "z" => 3)
    reduce(hcat, [sol_u[k, component_dict[xyz], :] for k in 1:size(sol_u)[1]])
end

function n_coarse(n_xyz, sol_t, coarse_times)
    reduce(hcat, [map(linear_interpolation(sol_t, n_xyz[spin_nr, :], extrapolation_bc=Line()), sol_t[end] .* coarse_times) for spin_nr in 1:size(n_xyz)[1]]) |> transpose |> Matrix
end

function shift_idx_to_center(idx, vec)
    @assert size(vec)[1] % 2 == 1
    reduce(vcat, [zeros(size(vec)[1] - idx), vec, zeros(idx - 1)])
end


# =================================================================================================================================