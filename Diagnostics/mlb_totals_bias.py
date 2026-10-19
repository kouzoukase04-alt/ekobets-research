# =========================================================
# mlb_totals_bias.py (2026-10-02, backlog item #6 -- "total-projection bias
# correction"). Blocked on 2026-09-26 because no graded totals history
# existed. There still isn't a totals ledger, but every saved MLB digest
# (daily_cards/MLB.txt + backups) and every saved run notebook prints, per
# snapshot game:
#     <Away> @ <Home>
#       TOT  line 7.5  model proj 7.9  (calibrated 8.2)
# This harvests every one of those (deduped by card date + matchup, keeping
# the LAST print of the day), grades them against MLB Stats API final scores,
# and reports mean(projected - actual) for the raw and calibrated totals, plus
# the market line's own bias as the yardstick.
# Caveat: the snapshot only prints games that had SOME edge on some market,
# not the full slate -- a selected, not random, sample.
# =========================================================
import json
import re
import os
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(os.environ.get("EKOBETS_ROOT", Path(__file__).resolve().parent.parent))
DATE_RE = re.compile(r"Card date: (\d{4}-\d{2}-\d{2})")
GAME_RE = re.compile(r"^(\S.*?) @ (\S.*?)\s*$")
TOT_RE = re.compile(r"TOT\s+line\s+([\d.]+)\s+model proj\s+([\d.]+)\s+\(calibrated\s+([\d.]+)\)")


def harvest_text(text, fallback_date=None, mtime=0.0):
    rows = []
    m = DATE_RE.search(text)
    d = m.group(1) if m else fallback_date
    if not d:
        return rows
    cur = None
    for line in text.splitlines():
        g = GAME_RE.match(line)
        if g and "TOT" not in line:
            cur = (g.group(1).strip(), g.group(2).strip())
            continue
        t = TOT_RE.search(line)
        if t and cur:
            rows.append({"date": d, "away": cur[0], "home": cur[1], "line": float(t.group(1)),
                         "proj": float(t.group(2)), "proj_cal": float(t.group(3)), "_mtime": mtime})
    return rows


def main():
    rows = []
    for f in list((ROOT / "daily_cards").glob("MLB*.txt")) + list((ROOT / "daily_cards" / "backups").rglob("MLB*")):
        try:
            rows += harvest_text(f.read_text(encoding="utf-8", errors="ignore"), mtime=f.stat().st_mtime)
        except Exception:
            pass
    for f in (ROOT / "sports" / "MLB").glob("NEWMLBMODEL2_out*.ipynb*"):
        try:
            nb = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        txt = "\n".join("".join(o.get("text", "")) for c in nb.get("cells", []) for o in c.get("outputs", [])
                        if o.get("output_type") == "stream")
        rows += harvest_text(txt, mtime=f.stat().st_mtime)
    df = pd.DataFrame(rows)
    if df.empty:
        print("No TOT snapshot lines found.")
        return
    df = df.sort_values("_mtime").drop_duplicates(["date", "away", "home"], keep="last")
    print(f"Harvested {len(df)} distinct (date, game) total projections, {df['date'].min()} -> {df['date'].max()}")

    finals = {}
    for d in sorted(df["date"].unique()):
        sch = requests.get(f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={d}", timeout=20).json()
        for dd in sch.get("dates", []):
            for g in dd["games"]:
                if g["status"]["detailedState"] != "Final":
                    continue
                t = g["teams"]
                # doubleheaders: two finals for the same pair on one date -> ambiguous, skipped
                k = (d, t["away"]["team"]["name"], t["home"]["team"]["name"])
                finals[k] = None if k in finals else t["away"]["score"] + t["home"]["score"]
    df["actual"] = [finals.get((r.date, r.away, r.home)) for r in df.itertuples()]
    g = df.dropna(subset=["actual"]).copy()
    print(f"Graded against final scores: {len(g)} (doubleheader/unmatched dropped: {len(df) - len(g)})")
    if g.empty:
        return
    for col, lab in [("proj", "raw model projection"), ("proj_cal", "calibrated projection"), ("line", "market line")]:
        b = (g[col] - g["actual"])
        print(f"  {lab:<24} mean(proj - actual) {b.mean():+.2f} runs  (SE {b.std(ddof=1) / len(b) ** 0.5:.2f})  "
              f"MAE {b.abs().mean():.2f}")
    g["cal_minus_raw"] = g["proj_cal"] - g["proj"]
    print(f"  calibration step adds {g['cal_minus_raw'].mean():+.2f} runs on average")
    over_side = g["proj_cal"] > g["line"]
    for lab, sub in [("model said OVER", g[over_side]), ("model said UNDER", g[~over_side])]:
        if len(sub):
            hit = (sub["actual"] > sub["line"]) if lab.endswith("OVER") else (sub["actual"] < sub["line"])
            push = (sub["actual"] == sub["line"]).sum()
            print(f"  {lab:<18} n={len(sub):<4} hit {hit.mean():.1%} (pushes {push})")
    g.to_csv(ROOT / "sports" / "MLB" / "tracking" / "mlb_totals_projection_graded.csv", index=False)
    print("  -> sports/MLB/tracking/mlb_totals_projection_graded.csv")


if __name__ == "__main__":
    main()
