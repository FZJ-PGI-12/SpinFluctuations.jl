@with_kw struct MetricParameters
    # padding
    pad::Int

    # coarse moving average
    navg_coarse::Int

    # interpolation
    g_ninterp::Int

    # fine moving average
    navg_fine::Int

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

    # Christoffel moving average
    navg::Int

    # Christoffel interpolation
    Γ_ninterp::Int

    # tolerances
    reltol::Float64
    abstol::Float64
end