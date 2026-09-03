## Step 04 — the two dependent variables

Denominator: **8** behaviour-changing mutations the suite detects.

| Condition | arm | selector | violations | violation rate | mean occupancy | saving |
|---|---|---|---|---|---|---|
| **C** | `extern` | testmon ∪ kernel-aware | 1/8 | **12.5 %** | 174.0 s | **24.5 %** (ref 230.3 s) |

### Per-mutation selection sizes

| Mutation | |D(m)| | C selected |
|---|---|---|
| `absdiff-drop-abs` | 17 | 59 |
| `conv1d-h-border-wrap` | 17 | 80 |
| `conv1d-h-drop-last-tap` | 17 | 80 |
| `conv1d-v-border-wrap` | 17 | 80 |
| `geometry-halve-max-radius` | 1 | 0 ⚠ |
| `grayscale-perturb-weight` | 5 | 44 |
| `grayscale-swap-bgr-weights` | 5 | 44 |
| `threshold-boundary` | 10 | 71 |

**Condition C violated on:**
- `geometry-halve-max-radius`
