"""
    ConstrainedLassoProblem(A, b, radius; constraint=:signed)

Minimize ½‖Ax-b‖² subject to ‖x‖₁ ≤ radius. With
`constraint=:nonnegative`, also require x ≥ 0. This is an L1-ball
constraint, not the penalized problem with λ = radius.
"""
struct ConstrainedLassoProblem{MA<:AbstractMatrix,VB<:AbstractVector,T}
    A::MA
    b::VB
    radius::T
    constraint::Symbol
end

function ConstrainedLassoProblem(A::AbstractMatrix, b::AbstractVector,
                                 radius; constraint=:signed)
    LassoProblem(A, b; constraint)
    radius isa Real && isfinite(radius) && radius >= 0 ||
        throw(ArgumentError("radius must be finite and nonnegative"))
    return ConstrainedLassoProblem(A, b, eltype(b)(radius), constraint)
end

"""Owned constrained-solve result; optimality is the projected-gradient mapping."""
struct ConstrainedResult{V,T,D}
    x::V
    radius::T
    objective::T
    residual_norm::T
    optimality_residual::T
    converged::Bool
    stop_reason::Symbol
    iterations::Int
    diagnostics::D
end

function _project_l1_ball!(out::AbstractVector, v::AbstractVector,
                           radius, nonnegative::Bool)
    radius == 0 && return fill!(out, zero(eltype(out)))
    magnitudes = nonnegative ? max.(v, zero(eltype(v))) : abs.(v)
    if sum(magnitudes) <= radius
        copyto!(out, nonnegative ? magnitudes : v)
        return out
    end
    sorted = sort(collect(magnitudes); rev=true)
    cumulative = zero(eltype(sorted))
    active = 0
    threshold = zero(eltype(sorted))
    for j in eachindex(sorted)
        cumulative += sorted[j]
        candidate = (cumulative - radius) / j
        if sorted[j] > candidate
            active = j
            threshold = candidate
        end
    end
    active > 0 || error("L1 projection failed to find an active coordinate")
    if nonnegative
        @. out = max(v - threshold, zero(eltype(out)))
    else
        @. out = sign(v) * max(abs(v) - threshold, zero(eltype(out)))
    end
    return out
end

function _constrained_feasible(problem::ConstrainedLassoProblem, x)
    length(x) == size(problem.A, 2) || return false
    tol = 100 * eps(eltype(x)) * max(one(eltype(x)), problem.radius)
    sum(abs, x) <= problem.radius + tol || return false
    return problem.constraint != :nonnegative || minimum(x) >= -tol
end

"""
    constrained_dual_gap(problem, x)

Fenchel primal-dual gap for least squares on an L1 ball. The dual vector
is the data residual; the support function is `radius*norm(A'u, Inf)` for
signed coefficients and `radius*max(0, maximum(-A'u))` for nonnegative
coefficients.
"""
function constrained_dual_gap(problem::ConstrainedLassoProblem, x)
    _constrained_feasible(problem, x) ||
        throw(ArgumentError("x is not feasible for the L1 ball"))
    residual = problem.A * x - problem.b
    gradient = adjoint(problem.A) * residual
    support = problem.constraint == :signed ?
              maximum(abs, gradient) :
              max(zero(eltype(gradient)), maximum(-gradient))
    gap = sum(abs2, residual) + dot(problem.b, residual) +
          problem.radius * support
    return max(zero(gap), gap)
end

"""Projected-gradient KKT mapping norm, zero exactly at a constrained optimum."""
function constrained_optimality_residual(problem::ConstrainedLassoProblem,
                                         x::AbstractVector; step=nothing)
    _constrained_feasible(problem, x) ||
        throw(ArgumentError("x is not feasible for the L1 ball"))
    t = isnothing(step) ? inv(max(opnorm(problem.A)^2, 1)) : step
    t isa Real && isfinite(t) && t > 0 ||
        throw(ArgumentError("step must be positive and finite"))
    gradient = adjoint(problem.A) * (problem.A * x - problem.b)
    projected = similar(x)
    _project_l1_ball!(projected, x .- t .* gradient, problem.radius,
                      problem.constraint == :nonnegative)
    return norm(x - projected) / t
end

_constrained_tolerance(problem, stop) =
    stop.atol + stop.rtol * max(norm(adjoint(problem.A) * problem.b), 1)

function _constrained_result(problem, x, converged, reason, iterations,
                             diagnostics; step)
    owned = copy(x)
    residual = norm(problem.A * owned - problem.b)
    mapping = constrained_optimality_residual(problem, owned; step)
    T = promote_type(typeof(residual), typeof(problem.radius))
    return ConstrainedResult(owned, T(problem.radius), T(0.5 * residual^2),
                             T(residual), T(mapping), converged, reason,
                             iterations, diagnostics)
end

"""Reusable FISTA workspace for an L1-ball constrained least-squares problem."""
struct ConstrainedFISTASolver{P,S}
    problem::P
    base::S
end

function ConstrainedFISTASolver(problem::ConstrainedLassoProblem;
                                step=:auto, stop=DualGapStop(), x0=nothing,
                                power_iterations=50, power_safety=1.2)
    problem.A isa StridedMatrix && problem.b isa StridedVector ||
        throw(ArgumentError("constrained FISTA currently requires CPU dense arrays"))
    isnothing(x0) || _constrained_feasible(problem, x0) ||
        throw(ArgumentError("x0 must satisfy the L1-ball constraint"))
    base = FISTASolver(LassoProblem(problem.A, problem.b;
                                    constraint=problem.constraint);
                       step, stop, x0, power_iterations, power_safety)
    return ConstrainedFISTASolver(problem, base)
end

function solve!(solver::ConstrainedFISTASolver; warm_start=true)
    p, base = solver.problem, solver.base
    x, y, previous = base.x, base.y, base.previous
    !warm_start && fill!(x, zero(eltype(x)))
    copyto!(y, x)
    tolerance = base.stop isa DualGapStop ?
                _dual_gap_tolerance(p.b, base.stop) :
                _constrained_tolerance(p, base.stop)
    scale = max(norm(adjoint(p.A) * p.b), 1)
    if base.stop isa KKTStop
        mapping = constrained_optimality_residual(p, x; step=base.step)
        if mapping <= tolerance
            return _constrained_result(p, x, true, :optimality, 0,
                                       (step=base.step,); step=base.step)
        end
    end
    momentum_t = one(base.step)
    for iteration in 1:base.stop.maxiter
        copyto!(previous, x)
        mul!(base.residual, p.A, y)
        @. base.residual = base.residual - p.b
        mul!(base.gradient, adjoint(p.A), base.residual)
        @. base.prox_input = y - base.step * base.gradient
        _project_l1_ball!(x, base.prox_input, p.radius,
                          p.constraint == :nonnegative)
        if iteration % base.stop.check_every == 0 ||
           iteration == base.stop.maxiter
            if base.stop isa KKTStop
                mapping = constrained_optimality_residual(p, x; step=base.step)
                if mapping <= tolerance
                    return _constrained_result(p, x, true, :optimality,
                                               iteration, (step=base.step,);
                                               step=base.step)
                end
            elseif base.stop isa DualGapStop
                change = norm(x - previous, Inf)
                if change <= base.stop.tol * norm(x, Inf)
                    gap = constrained_dual_gap(p, x)
                    if gap <= tolerance
                        return _constrained_result(p, x, true, :dual_gap,
                                                   iteration,
                                                   (step=base.step,
                                                    iterate_change=change,
                                                    dual_gap=gap);
                                                   step=base.step)
                    end
                end
            else
                @. base.prox_input = (y - x) / base.step
                mapping = norm(base.prox_input)
                if mapping <= tolerance
                    result = _constrained_result(p, x, true, :gradient_mapping,
                                                 iteration,
                                                 (step=base.step,
                                                  gradient_mapping_residual=mapping,
                                                  gradient_mapping_scale=scale);
                                                 step=base.step)
                    _warn_fista_optimality(result.optimality_residual / scale,
                                           base.stop, :projected_gradient_over_scale)
                    return result
                end
            end
        end
        next_t = (one(momentum_t) + sqrt(one(momentum_t) + 4momentum_t^2)) / 2
        momentum = (momentum_t - one(momentum_t)) / next_t
        @. y = x + momentum * (x - previous)
        momentum_t = next_t
    end
    result = _constrained_result(p, x, false, :maxiter, base.stop.maxiter,
                                 (step=base.step,); step=base.step)
    base.stop isa GradientMappingStop &&
        _warn_fista_optimality(result.optimality_residual / scale,
                               base.stop, :projected_gradient_over_scale)
    return result
end

"""Reusable CPU ADMM workspace for an L1-ball constrained least-squares problem."""
struct ConstrainedADMMSolver{P,S,T}
    problem::P
    base::S
    mapping_step::T
end

function ConstrainedADMMSolver(problem::ConstrainedLassoProblem;
                               rho=1.0, stop=DualGapStop(), x0=nothing,
                               feasibility_atol=1e-8,
                               feasibility_rtol=1e-4)
    problem.A isa StridedMatrix && problem.b isa StridedVector ||
        throw(ArgumentError("constrained ADMM currently requires CPU dense arrays"))
    isnothing(x0) || _constrained_feasible(problem, x0) ||
        throw(ArgumentError("x0 must satisfy the L1-ball constraint"))
    base = ADMMSolver(LassoProblem(problem.A, problem.b;
                                   constraint=problem.constraint);
                      rho, stop, x0, feasibility_atol, feasibility_rtol)
    t = inv(max(opnorm(problem.A)^2, 1))
    return ConstrainedADMMSolver(problem, base, t)
end

function solve!(solver::ConstrainedADMMSolver; warm_start=true)
    p, base = solver.problem, solver.base
    if !warm_start
        fill!(base.z, zero(eltype(base.z)))
        fill!(base.u, zero(eltype(base.u)))
    end
    tolerance = base.stop isa DualGapStop ?
                _dual_gap_tolerance(p.b, base.stop) :
                _constrained_tolerance(p, base.stop)
    if base.stop isa DualGapStop
        gap = constrained_dual_gap(p, base.z)
        if gap <= tolerance
            return _constrained_result(p, base.z, true, :dual_gap, 0,
                                       (primal_residual=zero(base.rho),
                                        dual_residual=zero(base.rho),
                                        rho=base.rho, dual_gap=gap);
                                       step=solver.mapping_step)
        end
    else
        mapping = constrained_optimality_residual(p, base.z;
                                                   step=solver.mapping_step)
        if mapping <= tolerance
            return _constrained_result(p, base.z, true, :optimality, 0,
                                       (primal_residual=zero(base.rho),
                                        dual_residual=zero(base.rho),
                                        rho=base.rho); step=solver.mapping_step)
        end
    end
    primal = oftype(base.rho, Inf)
    dual = oftype(base.rho, Inf)
    for iteration in 1:base.stop.maxiter
        _admm_x_step!(base)
        copyto!(base.z_previous, base.z)
        @. base.q = base.x + base.u
        _project_l1_ball!(base.z, base.q, p.radius,
                          p.constraint == :nonnegative)
        @. base.u = base.u + base.x - base.z
        if iteration % base.stop.check_every == 0 ||
           iteration == base.stop.maxiter
            primal = norm(base.x - base.z)
            dual = base.rho * norm(base.z - base.z_previous)
            primal_tol = sqrt(length(base.z)) * base.feasibility_atol +
                         base.feasibility_rtol * max(norm(base.x), norm(base.z))
            dual_tol = sqrt(length(base.z)) * base.feasibility_atol +
                       base.feasibility_rtol * base.rho * norm(base.u)
            if primal <= primal_tol && dual <= dual_tol
                if base.stop isa DualGapStop
                    gap = constrained_dual_gap(p, base.z)
                    if gap <= tolerance
                        return _constrained_result(p, base.z, true,
                            :feasibility_and_dual_gap, iteration,
                            (primal_residual=primal, dual_residual=dual,
                             rho=base.rho, dual_gap=gap);
                            step=solver.mapping_step)
                    end
                else
                    mapping = constrained_optimality_residual(p, base.z;
                                                               step=solver.mapping_step)
                    if mapping <= tolerance
                        return _constrained_result(p, base.z, true,
                            :feasibility_and_optimality, iteration,
                            (primal_residual=primal, dual_residual=dual,
                             rho=base.rho); step=solver.mapping_step)
                    end
                end
            end
        end
    end
    return _constrained_result(p, base.z, false, :maxiter,
                               base.stop.maxiter,
                               (primal_residual=primal, dual_residual=dual,
                                rho=base.rho); step=solver.mapping_step)
end

initialize(problem::ConstrainedLassoProblem, method::FISTA; x0=nothing) =
    ConstrainedFISTASolver(problem; step=method.step, stop=method.stop, x0,
                            power_iterations=method.power_iterations,
                            power_safety=method.power_safety)

initialize(problem::ConstrainedLassoProblem, method::ADMM; x0=nothing) =
    ConstrainedADMMSolver(problem; rho=method.rho, stop=method.stop, x0,
                           feasibility_atol=method.feasibility_atol,
                           feasibility_rtol=method.feasibility_rtol)

function solve(problem::ConstrainedLassoProblem, method::Union{FISTA,ADMM};
               x0=nothing)
    return solve!(initialize(problem, method; x0))
end

export ConstrainedLassoProblem, ConstrainedResult
export ConstrainedFISTASolver, ConstrainedADMMSolver
export constrained_optimality_residual, constrained_dual_gap
