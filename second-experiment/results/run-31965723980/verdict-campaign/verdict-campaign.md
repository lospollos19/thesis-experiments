## Step 04 — the two dependent variables

Denominator: **8** behaviour-changing mutations the suite detects.

| Condition | arm | selector | violations | violation rate | mean occupancy | saving |
|---|---|---|---|---|---|---|
| **C** | `extern` | testmon ∪ kernel-aware | 0/5 (of 8; 3 not measured) | **0.0 %** | 205.4 s | **nan %** (ref nan s) |

### Per-mutation selection sizes

| Mutation | |D(m)| | C selected |
|---|---|---|
| `absdiff-drop-abs` | 17 | 59 |
| `conv1d-h-border-wrap` | 17 | 80 |
| `conv1d-h-drop-last-tap` | 17 | — |
| `conv1d-v-border-wrap` | 17 | — |
| `geometry-halve-max-radius` | 1 | 188 |
| `grayscale-perturb-weight` | 5 | — |
| `grayscale-swap-bgr-weights` | 5 | 44 |
| `threshold-boundary` | 10 | 71 |

**Condition C has no measurement for:**
- `conv1d-h-drop-last-tap`
- `conv1d-v-border-wrap`
- `grayscale-perturb-weight`
