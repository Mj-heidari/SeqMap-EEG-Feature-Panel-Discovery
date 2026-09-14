"""Checks for the block-level explanation views."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hbn_r3.cli import RELEASES  # noqa: E402
from hbn_r4 import explain  # noqa: E402
from hbn_r4.selection import candidate_groups  # noqa: E402

CONFIG = {"seed": 7, "duration": 60, "inner_folds": 3, "n_estimators": 40,
          "min_samples_leaf": 3, "max_depth": 6}


def fixture(n_per_release=24, informative=True, seed=7):
    """Synthetic table with signal confined to one spectral block, or none."""
    rng = np.random.default_rng(seed)
    base = [f"sp_R{r}_{n}" for r in range(6)
            for n in ("theta", "alpha", "beta", "exponent", "offset", "alpha_cf", "alpha_height")]
    base += [f"co_R{i}_R{j}_{b}" for i in range(6) for j in range(i + 1, 6)
             for b in ("theta", "alpha", "beta")]
    base += [f"gr_R{r}_{b}_strength" for b in ("theta", "alpha", "beta") for r in range(6)]
    names = [f"{c}_{n}" for c in ("eo", "ec", "delta") for n in base]
    rows = []
    for release in RELEASES:
        for i in range(n_per_release):
            label = i % 2
            values = rng.normal(size=len(names))
            if informative:
                for k, name in enumerate(names):
                    if name.startswith("eo_sp_R0_"):
                        values[k] += 2.5 * label
            row = {"participant_id": f"{release}_{i:03d}", "group_id": f"{release}_{i:03d}",
                   "release": release, "label": label, "duration": CONFIG["duration"],
                   "age": float(rng.uniform(6, 17)), "sex": "F" if i % 3 else "M"}
            row.update({f"f__{n}": float(v) for n, v in zip(names, values)})
            rows.append(row)
    return pd.DataFrame(rows)


class ExplanationTests(unittest.TestCase):
    def setUp(self):
        self.table = fixture()
        self.groups = candidate_groups(self.table.columns, "blocks")

    def test_solo_finds_the_informative_block(self):
        solo = explain.block_solo_performance(self.table, self.groups, CONFIG)
        mean = solo.groupby("block").balanced_accuracy.mean()
        self.assertEqual(mean.idxmax(), "spectral_R0")
        self.assertGreater(mean.max(), 0.9)
        others = mean.drop("spectral_R0")
        self.assertLess(others.max(), 0.7)

    def test_permutation_credits_the_informative_block(self):
        importance = explain.block_importance(self.table, self.groups, CONFIG, n_repeats=4)
        mean = importance.groupby("block").importance.mean()
        self.assertEqual(mean.idxmax(), "spectral_R0")
        self.assertGreater(mean.max(), 0.1)

    def test_null_fixture_stays_at_chance(self):
        table = fixture(informative=False, seed=11)
        solo = explain.block_solo_performance(table, self.groups, CONFIG)
        mean = solo.groupby("block").balanced_accuracy.mean()
        self.assertLess(mean.max(), 0.65,
                        "a null fixture must not produce a decodable block")

    def test_every_block_is_scored_in_every_fold(self):
        solo = explain.block_solo_performance(self.table, self.groups, CONFIG)
        self.assertEqual(len(solo), len(self.groups) * len(RELEASES))
        self.assertTrue(solo.balanced_accuracy.between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
