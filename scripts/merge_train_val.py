"""Merge train (2010-2019) + val (2020-2021) into extended train CSVs.

Output:
    data/prices_train_extended.csv  (2010-01-04 -> 2021-12-31)
    data/macro_train_extended.csv   (2010-01-04 -> 2021-12-31)

Week 6 diagnostic showed train lacks any persistent-crisis regime (no GFC,
no COVID). Adding 2020-2021 brings the COVID crash + recovery into training,
the closest analogue to the 2022-2024 test regime. Test files untouched.

Stdlib only — no pandas required.
"""

import csv
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def read_csv(p):
    with open(p, newline="") as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


for name in ("prices", "macro"):
    th, tr = read_csv(DATA / f"{name}_train.csv")
    vh, vr = read_csv(DATA / f"{name}_val.csv")

    assert th == vh, f"{name}: column mismatch between train and val"
    assert vr[0][0] > tr[-1][0], \
        f"{name}: val starts at {vr[0][0]}, train ends at {tr[-1][0]} — overlap risk"

    out = DATA / f"{name}_train_extended.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(th)
        w.writerows(tr)
        w.writerows(vr)

    print(f"{name:6s}: {len(tr):4d} + {len(vr):4d} -> {len(tr)+len(vr):4d} rows  "
          f"({tr[0][0]} -> {vr[-1][0]})  ->  {out.name}")
