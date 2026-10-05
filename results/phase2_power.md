# Phase 2 sample size, simulated from Phase 1 residuals
Paired Wilcoxon signed-rank, alpha=0.05, target power=80%, 4000 simulations per point.
`n` is paired trials per arm. `--` means 480 trials were not enough.

## detector = `energy`, viewpoint = `local_user`

| condition | IQR | n for 10 ms | n for 25 ms | n for 50 ms | n for 100 ms |
|---|--:|--:|--:|--:|--:|
| nb8_ulaw | 3.6 | 10 | 10 | 10 | 10 |
| pstn_typical | 4.3 | 15 | 10 | 10 | 10 |
| pstn_poor | 5.0 | 30 | 20 | 10 | 10 |

## detector = `energy`, viewpoint = `wire`

| condition | IQR | n for 10 ms | n for 25 ms | n for 50 ms | n for 100 ms |
|---|--:|--:|--:|--:|--:|
| ref_16k | 3.6 | 10 | 10 | 10 | 10 |
| nb8_ulaw | 3.7 | 15 | 15 | 10 | 10 |
| pstn_typical | 6.4 | 30 | 15 | 10 | 10 |
| pstn_poor | 22.3 | 80 | 40 | 15 | 10 |

## detector = `silero`, viewpoint = `local_user`

| condition | IQR | n for 10 ms | n for 25 ms | n for 50 ms | n for 100 ms |
|---|--:|--:|--:|--:|--:|
| nb8_ulaw | 38.3 | 480 | 80 | 20 | 15 |
| pstn_typical | 90.4 | -- | 160 | 40 | 15 |
| pstn_poor | 88.4 | -- | 120 | 40 | 15 |

## detector = `silero`, viewpoint = `wire`

| condition | IQR | n for 10 ms | n for 25 ms | n for 50 ms | n for 100 ms |
|---|--:|--:|--:|--:|--:|
| ref_16k | 31.3 | 240 | 40 | 15 | 15 |
| nb8_ulaw | 63.5 | -- | 120 | 40 | 15 |
| pstn_typical | 139.5 | -- | 320 | 120 | 30 |
| pstn_poor | 113.4 | -- | 320 | 80 | 30 |
