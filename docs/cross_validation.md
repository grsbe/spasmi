# Grouped cross-validation

`cross_validate` runs a LASSO path separately on each training fold and
scores held-out observations. The caller controls which rows belong together;
Spasmi never shuffles timestamps or guesses what a scientifically independent
group is.

```julia
using Spasmi

# One group per measurement row. Supply sorted dates explicitly when input rows
# are not in chronological order.
folds = ordered_group_folds(day_per_row, 5;
                           order=sort(unique(day_per_row)),
                           assignment=:round_robin)
cv = cross_validate(LassoProblem(A, b; constraint=:nonnegative),
                    FISTA(); fold_ids=folds.fold_ids,
                    lambda_ratios=10.0 .^ range(0, -4; length=30))

cv.best_index               # minimum mean held-out RMSE
cv.one_se_index             # largest lambda within one SE of that minimum
cv.fold_scores              # lambda ratio × fold
cv.fold_converged            # numerical convergence of each training fit
```

Alternatively, pass your own positive, consecutive `fold_ids` directly.
`assignment=:contiguous` holds out adjacent groups instead of round-robin
groups. Neither choice removes possible temporal dependence across fold
boundaries; add an explicit gap in your data preparation if needed.

Each fold computes its own training-data `lambda_max` and evaluates the same
descending `lambda/lambda_max` ratios. Thus a ratio corresponds to a comparable
position on each fold's regularization path, **not** the same absolute lambda.
`fold_lambdas` and `fold_lambda_max` record the actual values. Predictions
are scored on the held-out rows, in the units of `b`. For a whitened fit,
provide a custom `score(prediction, observed, validation_rows)` callback to
report raw-unit error if desired.

`best_index` and `one_se_index` consider only ratios that converged in
every fold. They are `nothing` if no ratio qualifies; scores for unfinished
fits remain available for diagnosis. SE is the sample standard deviation of
the fold scores divided by the square root of the fold count. CV is a
diagnostic for predictive error, not proof that the recovered emission field
is physically identifiable.
