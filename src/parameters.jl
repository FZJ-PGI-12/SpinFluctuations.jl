@with_kw struct LyapunovParameters
    # final time for mean-field
    T_final::Float64

    # number of points to compute fluctuations for
    npts::Int

    # tolerances
    rtol::Float64
    atol::Float64
end

@with_kw struct MetricParameters
    # interpolation
    g_ninterp::Int

    # moving average
    g_navg::Int

    # metric amplitude
    C::Float64

    # metric shift
    δx::Float64
end

@with_kw struct AdaptiveScheduleParameters
    # annealing time
    T_anneal::Float64

    # DifferentialEquations.jl algorithm for BV problem
    algorithm

    # boundary values
    u0::Vector{Float64}

    # interpolation
    Γ_ninterp::Int

    # moving average
    Γ_navg::Int

    # tolerances
    reltol::Float64
    abstol::Float64
end