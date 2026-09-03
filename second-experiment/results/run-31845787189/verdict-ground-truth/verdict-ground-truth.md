## Ground truth — D(m)

- inline: baseline green at `3c991a6`
- extern: baseline green at `ee589c2`

| Mutation | changes behaviour | tests broken, per arm | in denominator |
|---|---|---|---|
| `absdiff-drop-abs` | True | inline: 17 / extern: 17 | yes |
| `conv1d-h-border-wrap` | True | inline: 17 / extern: 17 | yes |
| `conv1d-h-drop-last-tap` | True | inline: 17 / extern: 17 | yes |
| `conv1d-v-border-wrap` | True | inline: 17 / extern: 17 | yes |
| `geometry-halve-max-radius` | True | inline: 1 / extern: 1 | yes |
| `geometry-narrow-tile-width` | False | inline: 0 / extern: 0 | no (equivalent control) |
| `grayscale-perturb-weight` | True | inline: 5 / extern: 5 | yes |
| `grayscale-swap-bgr-weights` | True | inline: 5 / extern: 5 | yes |
| `threshold-boundary` | True | inline: 10 / extern: 10 | yes |

- mutations in the violation-rate denominator: **8**
- behaviour-changing but undetectable by the suite: **0**

- verdict: **GREEN**
