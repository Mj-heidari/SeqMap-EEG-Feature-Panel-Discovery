# Study results

Release-blocked leave-one-release-out on 514 participants at the 40 s budget.
Three targets are decoded from the identical cohort, representation and
protocol; only the label changes. Every figure and number below is reproduced
by the commands in the README.

## Cohort

| Release | Low attention | High attention | Mean age | Female |
|---|---:|---:|---:|---:|
| R4 | 61 | 62 | 10.9 | 42 |
| R5 | 70 | 63 | 10.5 | 51 |
| R7 | 60 | 70 | 10.4 | 51 |
| R9 | 66 | 62 | 10.1 | 48 |
| **Total** | **257** | **257** | **10.5** | **192** |

Cohort membership is fixed by the attention extreme-group frame (top vs bottom
tertile of the CBCL attention bifactor, cut-points fixed before modelling).
Groups are matched on age (p = 0.14) and differ modestly in sex composition
(p = 0.023). Of 771 eligible participants, 91 were excluded — 67 for
insufficient clean eyes-open data, 13 for channel-screen failure, 4 for
structurally corrupt amplitude, 3 for missing event markers, 2 for short
recordings, 2 other.

## Untouched-release performance of the selected panel

| Target | Pooled BA | AUC | Min | Max |
|---|---:|---:|---:|---:|
| Age band | 0.724 | 0.797 | 0.675 | 0.777 |
| Sex | 0.686 | 0.750 | 0.595 | 0.744 |
| Attention group | 0.498 | 0.482 | 0.470 | 0.569 |

Maturational and sex-related structure survives the acquisition-batch shift;
the attention contrast does not, and with 257 participants per class the
standard error of 0.044 excludes an effect larger than about 0.09 in either
direction.

## Parsimony

The one-standard-error panel is statistically indistinguishable from the full
315-feature representation for every target. For age band it uses 21–42 of the
315 candidates:

| Contrast (age band) | Δ BA | 95% CI |
|---|---:|---|
| Selected vs all 315 candidates | −0.004 | [−0.034, +0.027] |
| Selected vs first search step | +0.010 | [−0.012, +0.029] |

The corresponding intervals for sex (+0.005 and +0.010) and for attention
(−0.002 and +0.016) also contain zero.

## Single-block held-out balanced accuracy

| Block | Attention | Sex | Age band |
|---|---:|---:|---:|
| spectral LP | 0.503 | 0.662 | 0.731 |
| spectral LF | 0.468 | 0.570 | 0.715 |
| spectral RC | 0.475 | 0.634 | 0.709 |
| spectral LC | 0.488 | 0.638 | 0.706 |
| spectral RP | 0.540 | 0.696 | 0.702 |
| spectral RF | 0.456 | 0.581 | 0.700 |
| strength beta | 0.521 | 0.554 | 0.551 |
| connectivity alpha | 0.476 | 0.560 | 0.547 |
| strength theta | 0.519 | 0.544 | 0.543 |
| connectivity theta | 0.515 | 0.551 | 0.538 |
| strength alpha | 0.507 | 0.523 | 0.528 |
| connectivity beta | 0.553 | 0.576 | 0.504 |

The maturational signal lives in regional spectra and is duplicated across all
six regions; connectivity and strength blocks carry little. For attention no
block departs from chance.

## Where the maturational signal lives, by feature family

| Family (age band) | Features | BA | AUC |
|---|---:|---:|---:|
| Spectral-model parameters only | 72 | 0.744 | 0.816 |
| All candidates | 315 | 0.732 | 0.801 |
| Band powers only | 54 | 0.702 | 0.787 |
| Connectivity and strength only | 189 | 0.592 | 0.624 |

What these recordings encode about maturation is the shape of the power
spectrum — peak frequency and aperiodic slope. Phase coupling adds nothing.

## Why permutation importance hides it

Block permutation importance is near zero for the decodable targets: no block
exceeds 1.2 percentage points for age band despite its 0.724 accuracy, and
none exceeds 1.3 points for attention. The single exception anywhere in the
study is right-posterior spectra for sex, at 4.2 points, which is also the
target whose signal is least duplicated. Read alone, the permutation view
would make a strongly decodable target look uninformative; pairing it with
single-block refits is what separates "no signal anywhere" from "signal
duplicated everywhere".

## Panel stability

Mean pairwise Jaccard similarity of selected feature sets across the four outer
folds: age band 0.167, sex 0.222, attention 0.230. Instability is therefore not
a symptom of weak signal — it is most pronounced for the best-decoded target.
Every block entering an age-band panel is a regional spectral block, but no
region is chosen in more than two of four folds. A selected panel is a
sufficient measurement set, not the set of informative coordinates.

## Validity checks

Within the same cohort and from the same fitted spectral model, alpha centre
frequency correlates with age at r = +0.33 (p = 2×10⁻¹²) and the aperiodic
exponent at r = −0.27 (p = 1×10⁻⁹), recovering two established developmental
effects without being targeted.
