# =========================================================
# soccer_total_calib_fit.py (2026-10-06 user audit -- "Soccer totals are still too low ... Riestra 1.64 /
# Velez 1.92 vs 2.75 actual"). Out-of-sample check on the Dixon-Coles walk-forward
# (tracking/soccer_dc_walkforward.csv: every game's prior-only lambdas) found the opposite of a
# league-constant problem: per league the DC projection is ~unbiased overall but TOO SPREAD -- Argentina
# 2025-07+: projections < 1.8 averaged 1.62 vs 1.91 actual, > 2.4 averaged 2.59 vs 2.00 -- plus a
# current-season scoring drift (ARG 2026-07+: 2.02 projected vs 2.22 actual).
# Calibration fitted here:   total' = L_recent + b * (total_DC - L_dc_recent)
#   L_recent    = the league's actual mean total over the prior 120 days (prior-only, min 30 games;
#                 else the DC league mean)  -> tracks scoring drift
#   L_dc_recent = the DC projections' mean over the same window (so b only rescales the spread)
#   b           = one slope for all leagues, fitted 2024-07..2025-06, tested 2025-07+
# Writes sports/SOCCER/tracking/soccer_total_calib.json for SOCCERMODEL Cell 22.
#   python scripts/soccer_total_calib_fit.py [--write]
# =========================================================
import json, sys
import os
from pathlib import Path

ROOT = Path(os.environ.get("EKOBETS_ROOT", Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from scipy.stats import poisson

T = ROOT / "sports" / "SOCCER" / "tracking"


def main():
    w = pd.read_csv(T / "soccer_dc_walkforward.csv", parse_dates=["date"]).sort_values("date")
    w["pt"] = w.lh + w.la; w["at"] = w.hg + w.ag
    rows = []
    for lg, g in w.groupby("competition_key"):
        g = g.sort_values("date")
        for r in g.itertuples(index=False):
            win = g[(g.date < r.date) & (g.date >= r.date - pd.Timedelta(days=120))]
            if len(win) >= 30:
                rows.append((r.date, lg, r.pt, r.at, win["at"].mean(), win["pt"].mean()))
            else:
                rows.append((r.date, lg, r.pt, r.at, np.nan, np.nan))
    d = pd.DataFrame(rows, columns=["date", "lg", "pt", "at", "L", "Ldc"])
    d = d.dropna()
    tr, te = d[d.date < "2025-07-01"], d[d.date >= "2025-07-01"]
    mse = lambda p, y: float(np.mean((y - p) ** 2))
    pll = lambda p, y: float(np.mean(poisson.logpmf(y.astype(int), np.clip(p, 0.3, 8))))
    best = min(np.linspace(0.0, 1.2, 61), key=lambda b: mse(tr.L + b * (tr.pt - tr.Ldc), tr["at"]))
    for nm, f in (("DC raw", lambda x: x.pt), ("level only (b=1)", lambda x: x.L + (x.pt - x.Ldc)),
                  (f"level + slope b={best:.2f}", lambda x: x.L + best * (x.pt - x.Ldc))):
        print(f"{nm:24s} train MSE {mse(f(tr), tr['at']):.4f} | test MSE {mse(f(te), te['at']):.4f} | test Poisson LL {pll(f(te), te['at']):.4f}")
    a = te[te.lg == "argentina"]
    for lo, hi in ((0, 1.8), (1.8, 2.1), (2.1, 2.4), (2.4, 9)):
        s = a[(a.pt >= lo) & (a.pt < hi)]
        c = s.L + best * (s.pt - s.Ldc)
        print(f"  ARG test DC {lo}-{hi}: n={len(s)} raw {s.pt.mean():.2f} -> calibrated {c.mean():.2f} | actual {s['at'].mean():.2f}")
    if "--write" in sys.argv:
        (T / "soccer_total_calib.json").write_text(json.dumps({"slope_b": round(float(best), 3), "window_days": 120, "min_games": 30,
                                                                 "fitted": "2026-10-06", "train": "2024-07..2025-06", "test": "2025-07+"}, indent=1))
        print("wrote", T / "soccer_total_calib.json")


if __name__ == "__main__":
    main()
