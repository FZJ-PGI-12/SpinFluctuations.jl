@with_kw struct LyapunovParameters
    # final time for mean-field
    T_final::Float64

    # number of points to compute fluctuations for
    npts::Int

    # tolerances
    rtol::Float64
    atol::Float64
end