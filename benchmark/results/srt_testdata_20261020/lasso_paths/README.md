# SRT SNR30 nonnegative LASSO paths with Spasmi

[Path script](../../../srt_testdata_lasso_path.jl). These archived results use the same SRT test matrix and observations, Spasmi FISTA, the same power-estimated step, the then-default gradient-mapping stop, and warm starts along an identical decreasing lambda grid. The linked script now inherits Spasmi's dual-gap default, so rerunning it unmodified does not reproduce these old stopping results. CUDA device 1 is the Blackwell GPU. The first path point is **the exact zero solution**: for nonnegative LASSO, `lambda_max = max(A' * b) = 7.258878902438494e-5`; all 6,000 coefficients are zero there. This is not the earlier fixed benchmark lambda.

The path matches the SRE inversion workflow: 30 log-spaced points from `lambda/lambda_max = 1` to `1e-4`, followed by continuation at the same log spacing. The continuation stops after three consecutive relative RMSE gains below 1%, or at the safety ratio `1e-8`. Unlike the real observations, this synthetic benchmark has no defensible measurement-error RMSE target, so that target is omitted. The CPU path determines the grid; GPU dense and CSR replay it. Every point saves a checkpoint in `coefficients.jld2` plus per-point diagnostics in `path_diagnostics.csv`.

| Variant | Points | Sum of solve time | Final lambda ratio | Final RMSE | Final KKT/lambda |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU dense | 31 | 30.88 s | 7.27895e-5 | 2.917149e-5 | 0.611 |
| Blackwell GPU dense | 31 | 5.55 s | 7.27895e-5 | 2.917149e-5 | 0.611 |
| Blackwell GPU CSR | 31 | 2.15 s | 7.27895e-5 | 2.917149e-5 | 0.611 |

All 31 solves in all three variants met the historical gradient-mapping stop. Their predictions agree with CPU to relative error below `5e-15` (normalized by `norm(b)`). The timings sum each `solve!` call and include compilation in the first GPU call, but exclude data transfer and initialization.

**Interpretation caution:** the RMSE plateau is *not* a coefficient plateau. At the last step the relative coefficient change is about 1.24%, and KKT/lambda is 0.611; Spasmi warns about this returned-point optimality residual. Thus the path has reached the SRE-style *fit* stopping rule, not a demonstrated stable or high-accuracy solution at tiny lambda. A stricter low-lambda optimality study would need a different stop or further solves.

The [returned-point KKT tail test](../lasso_path_kkt_tail/README.md) confirms substantial field changes despite tiny prediction changes and uses the benchmark's known truth to show worsening reconstruction at smaller lambda. The short development smoke run is in sibling `lasso_path_smoke/`, separate from these full results.
