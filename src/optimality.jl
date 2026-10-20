"""
    lasso_kkt_residual(problem, x, lambda)

Largest coordinatewise violation of the exact LASSO KKT conditions at `x`.
This is evaluated at the **returned coefficients**, not a FISTA extrapolate.
For `constraint=:nonnegative`, the feasible-point condition at zero is
`gradient[j] + lambda ≥ 0`.
"""
function lasso_kkt_residual(problem::LassoProblem, x::AbstractVector, lambda)
    _validate_lambda(lambda)
    length(x) == size(problem.A, 2) ||
        throw(DimensionMismatch("x must have one entry per column of A"))
    g = adjoint(problem.A) * (problem.A * x - problem.b)
    if problem.constraint == :nonnegative
        minimum(x) >= 0 || throw(ArgumentError("x violates nonnegativity"))
        violations = ifelse.(x .> 0, abs.(g .+ lambda), max.(-g .- lambda, 0))
    else
        violations = ifelse.(x .> 0, abs.(g .+ lambda),
                     ifelse.(x .< 0, abs.(g .- lambda),
                             max.(abs.(g) .- lambda, 0)))
    end
    return maximum(violations)
end

# Internal no-allocation check using the FISTA workspace. The gradient at y has
# already served its purpose after the prox step and may be overwritten here.
function _lasso_kkt_residual!(problem::LassoProblem, x, lambda,
                              residual, gradient, violations)
    mul!(residual, problem.A, x)
    @. residual = residual - problem.b
    mul!(gradient, adjoint(problem.A), residual)
    if problem.constraint == :nonnegative
        @. violations = ifelse(x > 0, abs(gradient + lambda),
                                max(-gradient - lambda, 0))
    else
        @. violations = ifelse(x > 0, abs(gradient + lambda),
                                ifelse(x < 0, abs(gradient - lambda),
                                       max(abs(gradient) - lambda, 0)))
    end
    return maximum(violations)
end

"""Evaluate the penalized least-squares objective at a coefficient vector."""
function objective_value(problem::LassoProblem, x::AbstractVector, lambda)
    _validate_lambda(lambda)
    length(x) == size(problem.A, 2) ||
        throw(DimensionMismatch("x must have one entry per column of A"))
    problem.constraint == :nonnegative && minimum(x) < 0 &&
        return Inf
    return 0.5 * norm(problem.A * x - problem.b)^2 + lambda * sum(abs, x)
end

"""
    lasso_dual_gap(problem::LassoProblem, x, lambda)

Nonnegative primal-dual gap for signed or nonnegative penalized LASSO.
For `lambda > 0`, scale the residual `u = Ax-b` until the dual
constraint is feasible: `norm(A'u, Inf) <= lambda` for signed coefficients,
or `A'u >= -lambda` for nonnegative coefficients. At `lambda == 0`,
the zero dual vector gives a valid but usually loose bound; the default
stop uses a KKT fallback in that case.
"""
function lasso_dual_gap(problem::LassoProblem, x::AbstractVector, lambda)
    _validate_lambda(lambda)
    length(x) == size(problem.A, 2) ||
        throw(DimensionMismatch("x must have one entry per column of A"))
    problem.constraint == :nonnegative && minimum(x) < 0 &&
        throw(ArgumentError("x violates nonnegativity"))
    residual = problem.A * x - problem.b
    primal = 0.5 * sum(abs2, residual) + lambda * sum(abs, x)
    iszero(lambda) && return primal
    gradient = adjoint(problem.A) * residual
    dual_violation = problem.constraint == :signed ?
                     maximum(abs, gradient) :
                     max(zero(lambda), maximum(-gradient))
    scale = max(one(float(lambda)), dual_violation / lambda)
    gap = primal + 0.5 * sum(abs2, residual) / scale^2 +
          dot(problem.b, residual) / scale
    return max(zero(gap), gap)
end

_dual_gap_tolerance(b, stop::DualGapStop) = stop.tol * sum(abs2, b)
