# SeqMap

Sequential EEG feature-panel discovery with random forests, release-blocked
evaluation, and explanation views that stay interpretable when the result is
negative.

**Headline result.** On 514 children and adolescents from HBN-EEG releases R4,
R5, R7 and R9, no reproducible resting-state panel was found for CBCL
attention-factor extreme groups (pooled balanced accuracy 0.498). The identical
features, classifier and protocol recover age band at 0.724 and sex at 0.686,
so the pipeline demonstrably works and the limit is the label. See
`paper/main.pdf`.

## Quick start

Install 64-bit Python 3.12 and Git, then in PowerShell:

```powershell
git clone https://github.com/Mj-heidari/Repo4.git
cd Repo4
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[bdf]"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Reproducing the study

One command builds the feature table directly from the 100 Hz BDF releases:

```powershell
.\.venv\Scripts\python.exe scripts\extract_features.py --root "D:\EEG_Dataset\HBN_100Hz" --output local\features.csv
```

It writes `features.csv`, a per-participant exclusion log, and an audit file
recording every preprocessing setting. Then:

```powershell
.\.venv\Scripts\python.exe -m hbn_r4.cli search  --features local\features.csv --config configs\attention_blocks.json --output outputs\attention
.\.venv\Scripts\python.exe -m hbn_r4.cli plot    --results outputs\attention
.\.venv\Scripts\python.exe -m hbn_r4.cli explain --features local\features.csv --config configs\attention_blocks.json --output outputs\explain
```

`explain` computes both explanation views for the primary target and the two
positive controls, and writes the manuscript figures.

## Study definition

| | |
|---|---|
| Cohort | HBN-EEG R4, R5, R7, R9; ages 6–17; resting state available |
| Target | CBCL attention factor, top vs bottom tertile — **not** an ADHD diagnosis |
| Controls | Age band and sex, from the identical representation |
| Montage | Six symmetric scalp regions, 13 electrodes each (78 of 129) |
| Filter | 4th-order Butterworth IIR band-pass 1–40 Hz, average reference |
| Rejection | 250 µV peak-to-peak after a per-recording bad-channel screen |
| Features | 315: regional spectra, spectral-model parameters, debiased weighted phase-lag connectivity, regional strength, and EO/EC contrasts |
| Budgets | 20 s and 40 s per condition, nested |
| Evaluation | Leave-one-release-out, all fitting inside outer-training releases |

The distribution carries no clinician-assigned ADHD label, so the target is
named by what it is: an attention-problem extreme-group contrast derived from a
questionnaire bifactor.

## Two explanation views, and why both are needed

`explain` reports block permutation importance and single-block performance
together. Permutation importance answers what a block adds that no other block
supplies; in a redundant representation that is near zero even for a strongly
decodable target. Fitting each block alone answers what information the block
contains. Only the pair separates *no signal anywhere* from *signal duplicated
across many blocks* — which is exactly the distinction this dataset turns on.

## Interpreting selected panels

Selected panels are unstable across outer folds in this cohort, and that
instability is **not** a symptom of the null: it is just as pronounced for the
age control, where accuracy is high and reproduces in every release. When
several blocks carry the same information, which one a greedy search reaches
first is close to arbitrary. Report a panel as one adequate subset, never as
the biomarker set.

## Preprocessing notes

Two label-blind quality gates precede epoching, both consequential on this
release. Recordings whose median absolute amplitude exceeds 50 µV are rejected
as structurally corrupt. Within the rest, a channel whose median four-second
peak-to-peak amplitude departs from the montage median by more than a factor
of three is dropped; without this screen a single faulty electrode discards
every epoch of an otherwise usable recording. The 250 µV epoch threshold was
set in a diagnosis-blinded pilot: the conventional 150 µV discards most
eyes-open epochs in this pediatric cohort, where blinks dominate.

See [EXPERIMENTS.md](docs/EXPERIMENTS.md), [VALIDATION.md](docs/VALIDATION.md)
and [DATA_CONTRACT.md](docs/DATA_CONTRACT.md). Keep recordings, local manifests
and outputs outside git.
