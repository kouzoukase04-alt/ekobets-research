# =========================================================
# market_baseline_diagnostic.py (2026-10-04, user request -- "Score every model against a
# market-only baseline ... any sport where the model's Brier is worse than the market's gets
# its blend weight cut toward 0 until fixed. One diagnostic, run four times").
#
# For every GRADED bet each sport logged: the model's PRE-BLEND probability for the picked
# side vs the market's NO-VIG probability for the same side, scored on the same outcome.
#   - Brier(model) vs Brier(market), with a paired bootstrap CI on the difference
#   - blend sweep p = w*model + (1-w)*market, w in 0..1 -> evidence-based w (min log loss)
#   - by model-vs-market gap band: does a bigger disagreement lose more?
# Pushes / unresolved rows are excluded. UFC has no live graded ledger; it is scored from
# the notebook's own point-in-time backtest (scripts/ufc_backtest_dump.py writes it).
# Selection caveat: these are the bets each model chose to make (it disagreed with the
# market), which is exactly the population the blend weight acts on.
# =========================================================
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("EKOBETS_ROOT", Path(__file__).resolve().parent.parent))
rng = np.random.default_rng(0)


def _ll(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def score(name, model, market, y):
    model, market, y = map(lambda a: np.asarray(a, float), (model, market, y))
    ok = ~(np.isnan(model) | np.isnan(market) | np.isnan(y))
    model, market, y = model[ok], market[ok], y[ok]
    n = len(y)
    out = {"sport": name, "n": n}
    if n < 20:
        print(f"\n{name}: n={n} -- too few graded bets to score"); out["verdict"] = "insufficient"; return out
    bm, bk = ((model - y) ** 2).mean(), ((market - y) ** 2).mean()
    d = (model - y) ** 2 - (market - y) ** 2
    boots = [d[rng.integers(0, n, n)].mean() for _ in range(4000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    ws = np.round(np.arange(0, 1.0001, 0.05), 2)
    lls = [_ll(w * model + (1 - w) * market, y) for w in ws]
    w_best = float(ws[int(np.argmin(lls))])
    print(f"\n=== {name}  (n={n} graded bets, win rate {y.mean():.1%}) ===")
    print(f"  Brier  model {bm:.4f}   market {bk:.4f}   diff {bm - bk:+.4f}  (95% CI {lo:+.4f} .. {hi:+.4f})"
          f"   -> {'MODEL WORSE' if bm > bk else 'model better'}{' (significant)' if (lo > 0 or hi < 0) else ' (not significant)'}")
    print(f"  log-loss by blend weight w (p = w*model + (1-w)*market):")
    print("   " + "  ".join(f"w={w:.2f}:{l:.4f}" for w, l in zip(ws, lls) if w in (0, .1, .2, .3, .4, .5, .7, 1.0)))
    print(f"  evidence-based w (min log loss) = {w_best:.2f}")
    gap = np.abs(model - market)
    print("  by |model - market| gap:   n    real win   model avg   market avg   Brier model / market")
    for a, b in ((0, .05), (.05, .10), (.10, .15), (.15, .25), (.25, 1.01)):
        m = (gap >= a) & (gap < b)
        if m.sum() < 5:
            continue
        print(f"    {a:.2f}-{b:.2f}            {m.sum():<4} {y[m].mean():8.1%}   {model[m].mean():9.1%}   {market[m].mean():9.1%}"
              f"      {((model[m]-y[m])**2).mean():.4f} / {((market[m]-y[m])**2).mean():.4f}")
    out.update(brier_model=bm, brier_market=bk, ci=(lo, hi), w_best=w_best, verdict="MODEL WORSE" if bm > bk else "model better")
    return out


def soccer():
    df = pd.read_csv(ROOT / "sports/SOCCER/tracking/soccer_official_plays_ledger.csv")
    df = df[df["result"].isin(["WIN", "LOSS"])]
    return score("SOCCER", df["model"], df["market"], (df["result"] == "WIN").astype(float))


def cfb():
    df = pd.read_csv(ROOT / "sports/CFB/tracking/cfb_card_history.csv")
    df = df[df["result_label"].isin(["WIN", "LOSS"])].copy()
    pre = df["model_prob_preblend"] if "model_prob_preblend" in df else pd.Series(np.nan, index=df.index)
    # rows before the 10-03 blend stored model_prob on the PROB_SHRINK(0.80) scale -> undo it
    raw_old = 0.5 + (df["model_prob"] - 0.5) / 0.8
    model = pre.where(pre.notna(), raw_old)
    return score("CFB", model, df["market_prob"], (df["result_label"] == "WIN").astype(float))


def nfl():
    df = pd.read_csv(ROOT / "sports/NFL/tracking/nfl_card_history.csv")
    df = df[df["result_label"].isin(["WIN", "LOSS"])].copy()
    mkt = df["model_prob"] - df["edge"]
    # NFL-EDGE-BLEND-1 (2026-09-28): rows from then on store the BLENDED prob (w=0.40) ->
    # edge = w*(raw - mkt), so raw = mkt + edge/w
    blended = pd.to_datetime(df["card_date"], errors="coerce") >= "2026-09-28"
    raw = df["model_prob"].where(~blended, mkt + df["edge"] / 0.40)
    return score("NFL", raw, mkt, (df["result_label"] == "WIN").astype(float))


def ufc():
    p = ROOT / "sports/UFC/backtest_all_fights.csv"
    if not p.exists():
        print("\nUFC: run scripts/ufc_backtest_dump.py first"); return {"sport": "UFC", "verdict": "missing"}
    df = pd.read_csv(p)
    res = [score("UFC (all fights, both sides, point-in-time backtest)", df["model"], df["market"], df["outcome"])]
    q = df[df["qualified"] == 1]
    res.append(score("UFC (qualifying plays only)", q["model"], q["market"], q["outcome"]))
    return res[0]


WEIGHTS = ROOT / "sports" / "model_blend_weights.json"
MIN_N_TO_SET = 50   # below this many graded bets a sport keeps its current weight


def write_weights(results):
    """--write: the evidence-based w becomes the live blend weight each notebook reads.
    Rule (user, 2026-10-04): model Brier worse than market -> w cut to the fitted value
    (toward 0) until fixed; a model that beats the market keeps its fitted w."""
    import json, datetime
    _full = json.loads(WEIGHTS.read_text()) if WEIGHTS.exists() else {}
    cur = dict(_full.get("live_weights", {}))   # SHADOW-1 (2026-10-05): live_weights only
    for r in results:
        key = r["sport"].split()[0]
        if "w_best" not in r or r["n"] < MIN_N_TO_SET:
            print(f"  {key}: not updated (n={r.get('n', 0)} < {MIN_N_TO_SET} or no score)"); continue
        cur[key] = {"w": r["w_best"], "brier_model": round(r["brier_model"], 4), "brier_market": round(r["brier_market"], 4),
                    "n": r["n"], "verdict": r["verdict"], "set_on": datetime.date.today().isoformat(),
                    "source": "scripts/market_baseline_diagnostic.py"}
    _full["live_weights"] = cur
    WEIGHTS.write_text(json.dumps(_full, indent=1))
    print(f"  wrote {WEIGHTS}")


if __name__ == "__main__":
    results = [soccer(), cfb(), nfl(), ufc()]
    print("\n" + "=" * 70 + "\nSUMMARY -- evidence-based blend weight per sport\n" + "=" * 70)
    for r in results:
        if "w_best" in r:
            print(f"  {r['sport']:<8} n={r['n']:<5} Brier model {r['brier_model']:.4f} vs market {r['brier_market']:.4f}  "
                  f"-> {r['verdict']:<13} evidence w = {r['w_best']:.2f}")
        else:
            print(f"  {r['sport']:<8} {r.get('verdict')}")
    if "--write" in sys.argv:
        write_weights(results)
