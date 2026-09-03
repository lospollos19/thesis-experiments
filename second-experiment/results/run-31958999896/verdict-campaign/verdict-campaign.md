## Step 04 — the two dependent variables

Denominator: **8** behaviour-changing mutations the suite detects.

| Condition | arm | selector | violations | violation rate | mean occupancy | saving |
|---|---|---|---|---|---|---|
| **C** | `extern` | testmon ∪ kernel-aware | 0/1 (of 8; 7 not measured) | **0.0 %** | 1.5 s | **99.4 %** (ref 230.0 s) |

### Per-mutation selection sizes

| Mutation | |D(m)| | C selected |
|---|---|---|
| `absdiff-drop-abs` | 17 | — |
| `conv1d-h-border-wrap` | 17 | — |
| `conv1d-h-drop-last-tap` | 17 | — |
| `conv1d-v-border-wrap` | 17 | — |
| `geometry-halve-max-radius` | 1 | 188 |
| `grayscale-perturb-weight` | 5 | — |
| `grayscale-swap-bgr-weights` | 5 | — |
| `threshold-boundary` | 10 | — |

**Condition C has no measurement for:**
- `absdiff-drop-abs`
- `conv1d-h-border-wrap`
- `conv1d-h-drop-last-tap`
- `conv1d-v-border-wrap`
- `grayscale-perturb-weight`
- `grayscale-swap-bgr-weights`
- `threshold-boundary`
