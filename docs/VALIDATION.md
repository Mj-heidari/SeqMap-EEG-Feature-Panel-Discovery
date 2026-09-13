# Validation record

Four focused SeqMap tests passed locally on Python 3.12.14/Linux:

- Candidate parsing preserves distinct region names, including underscores.
- Forward selection identifies a known synthetic predictive coordinate and the compact-panel heuristic chooses one coordinate when adding noise does not help.
- Prediction on hostile/missing held-out values does not refit imputation.
- Family identifiers block relatives across outer releases.

A complete four-release synthetic block search ran for 128 fixture participants with three forward steps. It generated selected, first-step and all-candidate predictions and six PDF/PNG figure families. Those fixtures are intentionally separable and are not HBN performance evidence. Shared spectral/epoch/manifest code comes from the tested Repo3 extraction path. Actual BDF file reading and Windows execution were not available for local validation; the optional MNE path requires representative-file checks on the user's device.

Exact numerical package versions are in requirements-tested.txt. The repository CI is configured for Linux and Windows and is distinct from the local checks above.
