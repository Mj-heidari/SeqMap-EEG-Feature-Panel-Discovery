"""HBN-EEG resting-state feature extraction for the SeqMap / R3-Budget studies.

Builds the audited feature table used by both manuscripts directly from the
100 Hz BDF releases. Run once; the output feeds every downstream experiment.

    py -3 extract_features.py --root "D:\\EEG_Dataset\\HBN_100Hz" --output features.csv

Requires: mne, numpy, scipy, pandas.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import periodogram, windows

warnings.filterwarnings("ignore")

RELEASES = ("R4_L100", "R5_L100", "R7_L100", "R9_L100")
BANDS = {"theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
EO_CODE, EC_CODE = "instructed_toOpenEyes", "instructed_toCloseEyes"

# Prespecified analysis montage: six symmetric scalp regions of thirteen
# electrodes each, derived from the GSN-HydroCel-128 coordinates. Peripheral
# face, neck and outer-rim sensors and the midline strip are excluded by
# construction, so they never enter the average reference or epoch rejection.
REGIONS = {
    "LF": ["E13", "E18", "E19", "E20", "E22", "E23", "E24", "E25", "E26", "E27", "E28", "E33", "E34"],
    "RF": ["E2", "E3", "E4", "E8", "E9", "E10", "E112", "E116", "E117", "E118", "E122", "E123", "E124"],
    "LC": ["E29", "E30", "E31", "E35", "E36", "E37", "E39", "E40", "E41", "E42", "E45", "E46", "E47"],
    "RC": ["E80", "E87", "E93", "E98", "E102", "E103", "E104", "E105", "E108", "E109", "E110", "E111", "E115"],
    "LP": ["E50", "E51", "E52", "E53", "E54", "E58", "E59", "E60", "E61", "E65", "E66", "E67", "E70"],
    "RP": ["E77", "E78", "E79", "E83", "E84", "E85", "E86", "E90", "E91", "E92", "E96", "E97", "E101"],
}
ANALYSIS_CHANNELS = [c for v in REGIONS.values() for c in v]

REJECT_UV = 250.0        # blinded-pilot value for this pediatric cohort
FLAT_UV = 0.1
TRIM_SECONDS = 2.0
MAX_MEDIAN_UV = 50.0     # recording-level amplitude gate
MIN_REGION_CHANNELS = 8
DURATIONS = (20, 40)


# --------------------------------------------------------------------------
# Spectral and connectivity features
# --------------------------------------------------------------------------
def spectral_model(f, p):
    offset, exponent, height, center, width = p
    return offset - exponent * np.log10(f) + height * np.exp(-0.5 * ((f - center) / width) ** 2)


def spectral_parameters(freq, power):
    """Bounded single-alpha-peak fit of the 2-30 Hz log spectrum.

    A transparent approximation inspired by spectral parameterization; it is
    not the specparam implementation. Poor fits and boundary peaks are missing.
    """
    keep = (freq >= 2) & (freq <= 30)
    f, y = freq[keep], np.log10(np.maximum(power[keep], np.finfo(float).tiny))
    if len(f) < 30 or not np.isfinite(y).all():
        return np.full(4, np.nan), np.nan
    base = (f < 7) | (f > 14)
    slope, offset = np.polyfit(np.log10(f[base]), y[base], 1)
    residual = y - (offset + slope * np.log10(f))
    alpha = (f >= 7) & (f <= 13)
    start = [offset, np.clip(-slope, 0.01, 3.99), np.clip(np.max(residual[alpha]), 0.05, 2.9),
             f[alpha][np.argmax(residual[alpha])], 1.0]
    fit = least_squares(lambda p: spectral_model(f, p) - y, start,
                        bounds=([-40, 0, 0, 7, 0.5], [20, 4, 3, 13, 3]),
                        loss="soft_l1", f_scale=0.1, max_nfev=150)
    yhat = spectral_model(f, fit.x)
    r2 = 1 - np.sum((y - yhat) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
    b, chi, amp, cf, width = fit.x
    if not fit.success or r2 < 0.8:
        return np.full(4, np.nan), float(r2)
    if amp < 0.1 or width >= 2.99 or cf <= 7.01 or cf >= 12.99:
        amp, cf = np.nan, np.nan
    return np.array([chi, b, cf, amp]), float(r2)


def debiased_wpli(imaginary):
    """Debiased squared weighted phase-lag index; finite-sample negatives kept."""
    s = np.sum(imaginary, axis=0)
    a = np.sum(np.abs(imaginary), axis=0)
    q = np.sum(imaginary ** 2, axis=0)
    denominator = a * a - q
    return np.divide(s * s - q, denominator, out=np.full_like(s, np.nan, dtype=float),
                     where=denominator > np.finfo(float).tiny)


def condition_features(epochs, fs, regions):
    freq, p = periodogram(epochs, fs=fs, window="hann", detrend="constant", axis=-1, scaling="density")
    psd = np.mean(p, axis=0)
    values, names, fits = [], [], []
    for region, channels in regions.items():
        spectrum = np.mean(psd[channels], axis=0)
        for band, (lo, hi) in BANDS.items():
            mask = (freq >= lo) & (freq < hi)
            power = np.sum(spectrum[mask]) * (freq[1] - freq[0])
            values.append(np.log10(max(power, np.finfo(float).tiny)))
            names.append(f"sp_{region}_{band}")
        params, r2 = spectral_parameters(freq, spectrum)
        for name, val in zip(("exponent", "offset", "alpha_cf", "alpha_height"), params):
            names.append(f"sp_{region}_{name}")
            values.append(val)
        fits.append(r2)
    centered = epochs - epochs.mean(axis=-1, keepdims=True)
    ft = np.fft.rfft(centered * windows.hann(epochs.shape[-1], sym=False), axis=-1)
    region_names = list(regions)
    matrices = {band: np.zeros((6, 6)) for band in BANDS}
    for i, j in combinations(range(6), 2):
        pairs = []
        for ci in regions[region_names[i]]:
            for cj in regions[region_names[j]]:
                pairs.append(debiased_wpli(np.imag(ft[:, ci, :] * ft[:, cj, :].conj())))
        edge = np.nanmean(np.array(pairs), axis=0)
        for band, (lo, hi) in BANDS.items():
            val = float(np.nanmean(edge[(freq >= lo) & (freq < hi)]))
            names.append(f"co_{region_names[i]}_{region_names[j]}_{band}")
            values.append(val)
            matrices[band][i, j] = matrices[band][j, i] = val
    for band, mat in matrices.items():
        positive = np.clip(mat, 0, 1)
        for i, region in enumerate(region_names):
            names.append(f"gr_{region}_{band}_strength")
            values.append(float(np.sum(positive[i]) / 5))
    return np.asarray(values), names, float(np.nanmean(fits))


def feature_vector(eo, ec, fs, regions):
    a, names, qa = condition_features(eo, fs, regions)
    b, names2, qb = condition_features(ec, fs, regions)
    assert names == names2
    return np.r_[a, b, b - a], [f"{c}_{n}" for c in ("eo", "ec", "delta") for n in names], (qa, qb)


def progressive_indices(total, n, anchor=15):
    """Nested subsets of a fixed distributed anchor; not acquisition prefixes."""
    count = min(total, max(anchor, n))
    if n > total:
        raise ValueError("Not enough epochs")
    pool = np.linspace(0, total - 1, count, dtype=int)
    order = [0]
    while len(order) < n:
        remaining = [i for i in range(count) if i not in order]
        order.append(max(remaining, key=lambda i: min(abs(i - j) for j in order)))
    return np.sort(pool[order])


def extract_participant(data, fs, regions, duration):
    n = int(duration / 4)
    if min(len(data["eo"]), len(data["ec"])) < n:
        raise ValueError(f"Need {duration} clean seconds in EACH condition")
    selected = {c: x[progressive_indices(len(x), n)] for c, x in data.items()}
    full, names, qa = feature_vector(selected["eo"], selected["ec"], fs, regions)
    even = 2 * (n // 2)
    first, _, _ = feature_vector(selected["eo"][:even:2], selected["ec"][:even:2], fs, regions)
    second, _, _ = feature_vector(selected["eo"][1:even:2], selected["ec"][1:even:2], fs, regions)
    row = {f"f__{name}": float(v) for name, v in zip(names, full)}
    row.update({f"a__{name}": float(v) for name, v in zip(names, first)})
    row.update({f"b__{name}": float(v) for name, v in zip(names, second)})
    row.update({"qc_eo_r2": qa[0], "qc_ec_r2": qa[1], "duration": duration})
    return row


# --------------------------------------------------------------------------
# Signal preparation
# --------------------------------------------------------------------------
def epoch_arrays(signal, fs, intervals, reject_uv, flat_uv, trim_seconds):
    intervals = intervals.sort_values("onset_seconds")
    out = {"eo": [], "ec": []}
    rejected = 0
    for row in intervals.itertuples():
        start, duration = float(row.onset_seconds), float(row.duration_seconds)
        condition = str(row.condition)
        lo = int(np.ceil((start + trim_seconds) * fs))
        hi = int(np.floor((start + duration - trim_seconds) * fs))
        for first in range(lo, hi - 399, 400):
            x = signal[:, first:first + 400]
            if x.shape[1] < 400:
                continue
            ptp = np.ptp(x, axis=1) * 1e6
            if np.any(ptp > reject_uv) or np.any(ptp < flat_uv):
                rejected += 1
                continue
            out[condition].append(x)
    for condition in out:
        out[condition] = np.stack(out[condition]) if out[condition] else np.empty((0, signal.shape[0], 400))
    return out, rejected


def flag_bad_channels(signal, names, ratio=3.0, floor_ratio=0.2):
    """Label-blind channel screen on the robust within-channel amplitude level."""
    n = signal.shape[1] // 400
    if n < 10:
        raise ValueError("Recording too short to screen channels")
    level = np.median(np.ptp(signal[:, :n * 400].reshape(len(names), n, 400), axis=2), axis=1)
    reference = np.median(level)
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError("Degenerate amplitude reference")
    keep = (level <= ratio * reference) & (level >= floor_ratio * reference)
    return [c for c, k in zip(names, keep) if k]


def build_intervals(events_path, recording_seconds):
    """Verified EO/EC spans from the BIDS event markers; nothing is guessed."""
    ev = pd.read_csv(events_path, sep="\t")
    ev = ev[ev.value.isin([EO_CODE, EC_CODE])].sort_values("onset").reset_index(drop=True)
    if ev.empty:
        raise ValueError("No resting-state eye-condition markers in events.tsv")
    stops = list(ev.onset[1:]) + [recording_seconds]
    rows = []
    for (_, row), stop in zip(ev.iterrows(), stops):
        start = float(row.onset)
        duration = min(float(stop), recording_seconds) - start
        if duration > 0:
            rows.append({"onset_seconds": start, "duration_seconds": duration,
                         "condition": "eo" if row.value == EO_CODE else "ec"})
    return pd.DataFrame(rows)


def process_one(task):
    subject, release, meta, root = task
    import mne
    stem = Path(root) / f"{release}_bdf" / f"{release}_bdf" / subject / "eeg" / f"{subject}_task-RestingState"
    try:
        raw = mne.io.read_raw_bdf(str(stem) + "_eeg.bdf", preload=True, verbose="ERROR")
        present = [c for c in ANALYSIS_CHANNELS if c in raw.ch_names]
        if len(present) < len(ANALYSIS_CHANNELS) - 4:
            raise ValueError(f"Only {len(present)} of {len(ANALYSIS_CHANNELS)} analysis channels present")
        raw.pick(present)
        raw.set_channel_types({c: "eeg" for c in present}, verbose="ERROR")
        raw.filter(1.0, 40.0, method="iir", iir_params={"order": 4, "ftype": "butter"}, verbose="ERROR")
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")
        fs = float(raw.info["sfreq"])
        if not np.isclose(fs, 100.0):
            raw.resample(100.0, verbose="ERROR")
            fs = 100.0
        signal, names = raw.get_data(), list(raw.ch_names)
        raw.close()

        level = float(np.median(np.abs(signal))) * 1e6
        if not np.isfinite(level) or level > MAX_MEDIAN_UV:
            raise ValueError(f"Recording amplitude {level:.0f} uV exceeds the {MAX_MEDIAN_UV:.0f} uV quality gate")
        good = flag_bad_channels(signal, names)
        order = [names.index(c) for c in good]
        signal, names = signal[order], good
        signal = signal - signal.mean(axis=0, keepdims=True)

        intervals = build_intervals(str(stem) + "_events.tsv", signal.shape[1] / fs)
        arrays, rejected = epoch_arrays(signal, fs, intervals, REJECT_UV, FLAT_UV, TRIM_SECONDS)
        index = {c: [names.index(x) for x in v if x in names] for c, v in REGIONS.items()}
        if any(len(v) < MIN_REGION_CHANNELS for v in index.values()):
            raise ValueError("A scalp region retained too few channels after screening")

        counts = {"n_eo_epochs": len(arrays["eo"]), "n_ec_epochs": len(arrays["ec"]),
                  "n_epochs_rejected": rejected, "n_channels_dropped": len(ANALYSIS_CHANNELS) - len(names),
                  "median_uv": round(level, 2)}
        rows, failures = [], []
        for d in DURATIONS:
            row = dict(meta)
            row.update(counts)
            try:
                row.update(extract_participant(arrays, fs, index, d))
            except ValueError as exc:
                failures.append(f"{d}s: {exc}")
                continue
            rows.append(row)
        if not rows:
            raise ValueError("; ".join(failures))
        return subject, rows, None
    except Exception as exc:
        return subject, [], f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------
def build_cohort(root):
    frames = []
    for release in RELEASES:
        path = Path(root) / f"{release}_bdf" / f"{release}_bdf" / "participants.tsv"
        d = pd.read_csv(path, sep="\t")
        d["release"] = release
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    eligible = df[(df.RestingState == "available") & df.attention.notna() & df.age.between(6, 17)].copy()
    low, high = eligible.attention.quantile([1 / 3, 2 / 3])
    eligible["label"] = np.where(eligible.attention >= high, 1,
                                 np.where(eligible.attention <= low, 0, -1))
    cohort = eligible[eligible.label >= 0].copy()
    cohort["group_id"] = cohort.participant_id
    return cohort, float(low), float(high)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=r"D:\EEG_Dataset\HBN_100Hz")
    parser.add_argument("--output", default="features.csv")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    args = parser.parse_args()

    cohort, low, high = build_cohort(args.root)
    print(f"Cohort: {len(cohort)} participants "
          f"({(cohort.label == 1).sum()} high / {(cohort.label == 0).sum()} low attention)")
    print(f"Attention tertile cut-points: low <= {low:.4f}, high >= {high:.4f}")
    print(f"Extracting with {args.workers} worker processes...\n", flush=True)

    tasks = []
    for r in cohort.itertuples():
        meta = {"participant_id": r.participant_id, "release": r.release, "label": int(r.label),
                "age": float(r.age), "sex": r.sex, "group_id": r.participant_id,
                "attention": float(r.attention), "p_factor": float(r.p_factor)}
        tasks.append((r.participant_id, r.release, meta, args.root))

    records, excluded, done = [], [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, t): t[0] for t in tasks}
        for future in as_completed(futures):
            subject, rows, error = future.result()
            done += 1
            if error:
                excluded.append({"participant_id": subject, "reason": error})
            else:
                records.extend(rows)
            if done % 25 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} processed, {len(excluded)} excluded", flush=True)

    if not records:
        raise SystemExit("No participants extracted; inspect the exclusion log.")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(out, index=False)
    pd.DataFrame(excluded, columns=["participant_id", "reason"]).to_csv(out.with_suffix(".excluded.csv"), index=False)
    audit = {
        "cohort_participants": len(cohort),
        "extracted_participants": int(frame.participant_id.nunique()),
        "excluded_participants": len(excluded),
        "durations": list(DURATIONS),
        "reject_uv": REJECT_UV, "flat_uv": FLAT_UV, "trim_seconds": TRIM_SECONDS,
        "max_median_uv": MAX_MEDIAN_UV, "min_region_channels": MIN_REGION_CHANNELS,
        "attention_tertile_low": low, "attention_tertile_high": high,
        "regions": REGIONS, "sampling_hz": 100, "epoch_seconds": 4,
        "reference": "average over the analysis montage",
        "filter": "4th-order Butterworth IIR band-pass, 1-40 Hz",
        "synthetic": False,
        "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__,
    }
    out.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2))

    n = frame[frame.duration == max(DURATIONS)]
    print(f"\nWrote {out} ({frame.participant_id.nunique()} participants, {len(frame)} rows)")
    print(f"At the {max(DURATIONS)} s budget: {len(n)} participants")
    print(n.groupby(["release", "label"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
