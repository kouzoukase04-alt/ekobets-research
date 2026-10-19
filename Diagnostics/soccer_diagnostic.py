# =========================================================
# soccer_diagnostic.py (2026-09-19, user request)
# Mirrors UFC's backtest diagnostics (win rate by edge band, win
# rate by grade/tier, Brier score) but reads the live
# soccer_official_plays_ledger.csv instead of a historical backtest --
# soccer has no equivalent to UFC's 2-years-of-known-fights backtest,
# only this manually-graded ledger (soccer_update_result(), same
# manual-grading pattern as UFC's ufc_update_result()).
#
# Run against the EXISTING ledger, before touching the projection
# model, per the user's own stated priority.
# =========================================================
import sys
import os
from pathlib import Path

ROOT = Path(os.environ.get("EKOBETS_ROOT", Path(__file__).resolve().parent.parent))

import pandas as pd

LEDGER = ROOT / "sports" / "SOCCER" / "tracking" / "soccer_official_plays_ledger.csv"
RAW_MATCHES = ROOT / "sports" / "SOCCER" / "data" / "raw" / "soccer" / "soccer_epl_ucl_raw_matches.csv"
CUP_KEYS = {"libertadores", "ucl", "leagues_cup", "sudamericana", "uel", "uecl", "efl_cup"}
# ledger competition_key -> raw_matches competition_key (the two files name some leagues differently)
LEDGER_TO_RAW = {"serie_a": "seriea", "efl_champ": "championship"}


def load_graded():
    df = pd.read_csv(LEDGER)
    graded = df[df["result"].isin(["WIN", "LOSS", "PUSH"])].copy()
    return df, graded


def compute_pl(row):
    """profit_dollars is populated for only 7/595 rows -- compute it fresh
    from size_dollars/dec_odds/result instead of relying on that mostly-
    empty column (same approach UFC's backtest already uses)."""
    if row["result"] == "PUSH":
        return 0.0
    size = row.get("size_dollars")
    dec = row.get("dec_odds")
    if pd.isna(size):
        return None
    if row["result"] == "WIN":
        if pd.isna(dec):
            return None
        return round(size * (dec - 1), 2)
    return round(-size, 2)


def band_report(df, col, bands, label_fmt, title):
    print(title)
    scored = df[df["result"].isin(["WIN", "LOSS"])]
    for lo, hi in bands:
        bp = scored[(scored[col] >= lo) & (scored[col] < hi)]
        if bp.empty:
            continue
        w = int((bp["result"] == "WIN").sum())
        l = int((bp["result"] == "LOSS").sum())
        pl = bp["pl"].dropna().sum()
        print(f"  {label_fmt(lo, hi)}  {w}-{l} ({100*w/(w+l):.1f}%)  ${pl:+.2f}  ({len(bp)} plays)")
    print()


def main():
    raw, graded = load_graded()
    print("=" * 70)
    print("SOCCER LEDGER DIAGNOSTIC")
    print("=" * 70)
    print(f"Ledger rows total: {len(raw)}  |  Graded (WIN/LOSS/PUSH): {len(graded)}")
    ungraded = raw[raw["result"].isna()]
    if len(ungraded):
        by_date = ungraded.groupby("card_date").size()
        print(f"UNGRADED: {len(ungraded)} rows across {by_date.shape[0]} card date(s) "
              f"({by_date.index.min()} -> {by_date.index.max()}) -- no automated grading exists "
              f"for this sport (soccer_update_result() is manual-only, same pattern as UFC's "
              f"ufc_update_result()); most of these are for games that have already been played "
              f"and could be backfilled from real final scores.")
    print()

    if graded.empty:
        print("No graded rows -- nothing to diagnose. Grade the ledger first.")
        return

    graded["pl"] = graded.apply(compute_pl, axis=1)
    scored = graded[graded["result"].isin(["WIN", "LOSS"])]
    wins = int((scored["result"] == "WIN").sum())
    losses = int((scored["result"] == "LOSS").sum())
    pushes = int((graded["result"] == "PUSH").sum())
    net_pl = scored["pl"].dropna().sum()
    print(f"Overall (excl. push): {wins}-{losses} ({100*wins/(wins+losses):.1f}%)  "
          f"|  Net P/L: ${net_pl:+.2f}  |  Pushes: {pushes}")
    if len(scored) < 40:
        print(f"  ** SAMPLE SIZE WARNING: n={len(scored)} is thin -- UFC's own backtest runs on "
              f"78-82 plays and its bands still carry real uncertainty at that size. Every "
              f"breakdown below is directional, not conclusive, until this grows.")
    print()

    # Win rate by edge band -- soccer edges run smaller than UFC's typically;
    # bands chosen to match this sport's own real distribution instead of
    # reusing UFC's bands verbatim.
    band_report(scored, "edge", [(0.04, 0.08), (0.08, 0.12), (0.12, 0.18), (0.18, 1.00)],
                lambda lo, hi: f"{lo:.0%}-{hi:.0%} edge",
                "Win rate by edge band:")

    # Win rate by model-probability band (mirrors the UFC grade-inversion
    # diagnostic -- does a higher model win-prob actually track outcomes,
    # independent of edge size?)
    band_report(scored, "model", [(0.50, 0.55), (0.55, 0.60), (0.60, 0.70), (0.70, 1.00)],
                lambda lo, hi: f"{lo:.0%}-{hi:.0%} model prob",
                "Win rate by model-probability band:")

    # Win rate by play tier (OFFICIAL PLAY vs LEAN vs everything else,
    # collapsing the many distinct INFORMATIONAL/CALIBRATION-WATCH variants
    # into one "other" bucket -- there are too few graded rows per exact
    # variant to break them out individually yet)
    def _tier(cat):
        c = str(cat)
        if c == "OFFICIAL PLAY":
            return "OFFICIAL PLAY"
        if c == "LEAN":
            return "LEAN"
        return "OTHER (informational/calibration-watch/new-league)"
    scored = scored.copy()
    scored["tier"] = scored["play_category"].apply(_tier)
    print("Win rate by play tier:")
    for tier in ["OFFICIAL PLAY", "LEAN", "OTHER (informational/calibration-watch/new-league)"]:
        bp = scored[scored["tier"] == tier]
        if bp.empty:
            continue
        w = int((bp["result"] == "WIN").sum())
        l = int((bp["result"] == "LOSS").sum())
        pl = bp["pl"].dropna().sum()
        print(f"  {tier:<48} {w}-{l} ({100*w/(w+l):.1f}%)  ${pl:+.2f}  ({len(bp)} plays)")
    print()

    # Win rate by no_backtest_league (True = one of the leagues added without
    # backtested history -- the exact split the "new league" self-flag was
    # meant to isolate)
    print("Win rate by no_backtest_league flag:")
    for flag, label in [(True, "new/no-backtest league"), (False, "established/backtested league")]:
        bp = scored[scored["no_backtest_league"] == flag]
        if bp.empty:
            continue
        w = int((bp["result"] == "WIN").sum())
        l = int((bp["result"] == "LOSS").sum())
        pl = bp["pl"].dropna().sum()
        print(f"  {label:<32} {w}-{l} ({100*w/(w+l):.1f}%)  ${pl:+.2f}  ({len(bp)} plays)")
    print()

    # DATA COVERAGE split (2026-10-03 user review -- "the soccer diagnostic's numbers for
    # Brazilian leagues are contaminated"). raw_matches only carries domestic-league games for
    # the competitions listed in it (EPL/LaLiga/Serie A/Bundesliga/Ligue1/Championship/MLS);
    # Brazil/Argentina/Liga MX/Eredivisie/Portugal/Belgium/Turkey clubs were priced off
    # continental-cup games only (Atletico-MG's "last 5" on 10-03 were 2024 Libertadores).
    # Those rows are reported separately so they don't pass as evidence about the model.
    try:
        _rm = pd.read_csv(RAW_MATCHES, usecols=["competition_key"])
        _domestic = set(_rm["competition_key"].astype(str).str.lower()) - CUP_KEYS
    except Exception as _e:
        _domestic = None
        print(f"Data-coverage split skipped (raw_matches unreadable: {_e})\n")
    if _domestic is not None and "competition_key" in scored.columns:
        _ck = scored["competition_key"].astype(str).str.lower().map(lambda k: LEDGER_TO_RAW.get(k, k))
        scored["_low_data"] = ~_ck.isin(_domestic | CUP_KEYS)
        print("By data coverage (LOW DATA = league has NO domestic match data in raw_matches -- model inputs stale):")
        for flag, label in [(False, "covered league"), (True, "LOW DATA league (contaminated)")]:
            bp = scored[scored["_low_data"] == flag]
            if bp.empty:
                continue
            w = int((bp["result"] == "WIN").sum()); l = int((bp["result"] == "LOSS").sum())
            br = bp.dropna(subset=["model"])
            b = (((br["model"] - (br["result"] == "WIN").astype(float)) ** 2).mean()) if len(br) else float("nan")
            print(f"  {label:<34} {w}-{l} ({100*w/max(1,w+l):.1f}%)  ${bp['pl'].dropna().sum():+.2f}  "
                  f"Brier {b:.4f}  ({len(bp)} plays; leagues: {', '.join(sorted(bp['competition_key'].astype(str).str.lower().unique()))})")
        print("  -> every number below this point still pools both groups; read the LOW DATA rows as data-gap noise.\n")

    # Brier score: model's own predicted probability for the PICKED side vs
    # whether that side actually won -- same metric UFC's backtest reports.
    brier_rows = scored.dropna(subset=["model"])
    outcome = (brier_rows["result"] == "WIN").astype(float)
    brier = ((brier_rows["model"] - outcome) ** 2).mean()
    print(f"Brier score (picked-side model prob vs outcome): {brier:.4f}  "
          f"(lower=better, ref=0.25 coin-flip)  n={len(brier_rows)}")
    print()

    brier_by_edge_band(scored)
    systematic_vs_noise_by_side(scored)
    market_blend_sweep(scored)
    margin_stage_slopes()
    projected_total_bias_by_league(raw)


def margin_stage_slopes():
    """(2026-10-02) Where does the soccer strength gap get squashed? Reads
    tracking/soccer_margin_stages.csv (logged every run by SOCCERMODEL Cell 24)
    and regresses Pinnacle's AH-implied margin on the model's margin at each
    stage. A stage whose slope jumps well above 1 relative to the stage before
    it is where compression enters. sigma = sqrt(2.7) converts the 1X2-stage
    probits to goals so all three stages read on one scale."""
    import numpy as np
    from scipy.stats import norm
    path = LEDGER.parent / "soccer_margin_stages.csv"
    print("=" * 70)
    print("MARGIN BY PIPELINE STAGE vs PINNACLE AH (slope > 1 = model squashes gaps)")
    print("=" * 70)
    if not path.exists():
        print("  soccer_margin_stages.csv not written yet -- logged from 2026-10-02 forward.\n")
        return
    d = pd.read_csv(path)
    d["m_core"] = norm.ppf(d["q_core"].clip(0.02, 0.98)) * np.sqrt(2.7)
    d["m_final1x2"] = norm.ppf(d["q_final"].clip(0.02, 0.98)) * np.sqrt(2.7)
    d["m_lambda"] = pd.to_numeric(d["lam_home"], errors="coerce") - pd.to_numeric(d["lam_away"], errors="coerce")
    for col, lab in [("m_core", "stage 1  ClubElo core 1X2"), ("m_final1x2", "stage 2  final 1X2 (override+adj)"),
                     ("m_lambda", "stage 3  AH-grid lambdas")]:
        s = d.dropna(subset=[col, "mkt_margin"])
        if len(s) < 30:
            print(f"  {lab:<36} n={len(s)} -- need 30+ games before this slope means anything")
            continue
        b = np.polyfit(s[col], s["mkt_margin"], 1)
        print(f"  {lab:<36} n={len(s)}  slope {b[0]:.2f}  intercept {b[1]:+.2f}")
    print()


# ---------------------------------------------------------------------------
# (2026-10-01, user request -- SKC/SEA: model had Seattle 52.7% vs the
# market's 62% against the league's worst defense, second straight SKC
# Under.) "Is the model systematically low on mismatches, or just noisy?"
# needs more than overall win rate: Brier per edge band (model vs market on
# the SAME rows), and a bias test per side-of-the-bet segment. A segment is
# called SYSTEMATIC when mean(outcome - model_prob) sits more than 2 standard
# errors from zero -- the model's miss has a consistent direction there, not
# just scatter around the right answer.
# ---------------------------------------------------------------------------
_AH_LINE_RE = r"([+-]\d+(?:\.\d+)?)\s*$"


def _bet_side(row):
    pick = str(row["pick"])
    if row["play_type"] == "O/U":
        return pick.split()[0].upper()
    if row["play_type"] == "AH":
        m = pd.Series([pick]).str.extract(_AH_LINE_RE)[0].iloc[0]
        if pd.isna(m):
            return "AH ?"
        v = float(m)
        return "AH underdog (+line)" if v > 0 else ("AH favorite (-line)" if v < 0 else "AH level (0)")
    return str(row["play_type"])


def _brier(p, y):
    return float(((p - y) ** 2).mean())


def brier_by_edge_band(scored):
    print("Brier by edge band (model vs market on the SAME plays -- if the market's Brier")
    print("is lower in a band, the model's disagreement in that band is costing accuracy):")
    rows = scored.dropna(subset=["model", "market"])
    y = (rows["result"] == "WIN").astype(float)
    for lo, hi in [(0.04, 0.08), (0.08, 0.12), (0.12, 0.18), (0.18, 1.00)]:
        m = (rows["edge"] >= lo) & (rows["edge"] < hi)
        if not m.any():
            continue
        bm, bk = _brier(rows.loc[m, "model"], y[m]), _brier(rows.loc[m, "market"], y[m])
        print(f"  {lo:.0%}-{hi:.0%} edge  model {bm:.4f}  vs  market {bk:.4f}  "
              f"({'MARKET better' if bk < bm else 'model better'}, n={int(m.sum())})")
    print()


def systematic_vs_noise_by_side(scored):
    print("Systematic vs noise, by side of the bet (bias = real win rate - model avg prob;")
    print("SYSTEMATIC = |bias| > 2 SE, i.e. the model misses in a consistent direction):")
    rows = scored.dropna(subset=["model", "market"]).copy()
    rows["y"] = (rows["result"] == "WIN").astype(float)
    rows["side"] = rows.apply(_bet_side, axis=1)
    for side, sub in rows.groupby("side"):
        n = len(sub)
        bias = sub["y"].mean() - sub["model"].mean()
        se = (sub["y"].std(ddof=1) / n ** 0.5) if n > 1 else float("nan")
        verdict = ("SYSTEMATIC" if se == se and abs(bias) > 2 * se else "noise-compatible")
        print(f"  {side:<22} n={n:<4} real {sub['y'].mean():.1%}  model {sub['model'].mean():.1%}  "
              f"mkt {sub['market'].mean():.1%}  bias {bias:+.1%} (SE {se:.1%})  -> {verdict}"
              + ("  ** n<20, directional only" if n < 20 else ""))
    # Mismatch magnitude: bigger AH lines = bigger market-perceived mismatch.
    ah = rows[rows["play_type"] == "AH"].copy()
    ah["line"] = pd.to_numeric(ah["pick"].str.extract(_AH_LINE_RE)[0], errors="coerce")
    ah = ah.dropna(subset=["line"])
    if not ah.empty:
        print("  AH by mismatch size (|handicap line|):")
        for lo, hi in [(0.0, 0.75), (0.75, 1.25), (1.25, 10.0)]:
            sub = ah[(ah["line"].abs() >= lo) & (ah["line"].abs() < hi)]
            if sub.empty:
                continue
            print(f"    |line| {lo:g}-{hi:g}: n={len(sub):<3} real {sub['y'].mean():.1%}  "
                  f"model {sub['model'].mean():.1%}  mkt {sub['market'].mean():.1%}  "
                  f"underdog-side share {(sub['line'] > 0).mean():.0%}")
    print()


def market_blend_sweep(scored):
    """Same log-loss grid search MLB used to pick its w=0.15 (2026-09-26) --
    p = w*model + (1-w)*market on the graded picked-side probabilities.
    The live constant is _SOCCER_MODEL_MARKET_BLEND_W in SOCCERMODEL.ipynb
    (shared-helpers cell); this sweep is what it should be checked against."""
    import numpy as np
    print("Market-blend sweep (p = w*model + (1-w)*no-vig market, graded plays):")
    rows = scored.dropna(subset=["model", "market"]).copy()
    rows["y"] = (rows["result"] == "WIN").astype(float)

    def _ll(p, y):
        p = p.clip(1e-4, 1 - 1e-4)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    for label, sub in [("ALL", rows)] + list(rows.groupby("play_type")):
        best = min((_ll(w * sub["model"] + (1 - w) * sub["market"], sub["y"]), w)
                   for w in np.arange(0, 1.0001, 0.05))
        print(f"  {label:<4} n={len(sub):<4} best w={best[1]:.2f} (log-loss {best[0]:.4f})  |  "
              f"w=0.15 {_ll(0.15*sub['model']+0.85*sub['market'], sub['y']):.4f}  |  "
              f"w=1.00 (raw model) {_ll(sub['model'], sub['y']):.4f}")
    print("  Caveat: these are plays the model CHOSE because it disagreed with the market,")
    print("  so this measures whether that disagreement carried information -- not how the")
    print("  model does on games it never flagged.")
    print()


def projected_total_bias_by_league(raw):
    """(2026-09-27 user audit, priority item 2) mean(projected - actual) total
    goals, grouped by league -- "before any projection changes." Real cases
    that prompted this: Liga MX projected 3.33 goals for a game that had 1;
    SKC/HOU projected 2.39 and was closer -- the same non-EPL projection gap
    behind the 9/19 losses, never previously measured directly.

    proj_total_goals is a NEW ledger column (added 2026-09-27) -- every row
    logged before that date has it blank, so this can only report on rows
    logged since. Runs and prints an honest "not enough data yet" rather
    than silently producing nothing when that's all there is, same
    discipline as this script's own sample-size warning above.
    """
    print("=" * 70)
    print("PROJECTED VS ACTUAL TOTAL GOALS, BY LEAGUE")
    print("=" * 70)
    if "proj_total_goals" not in raw.columns:
        print("  proj_total_goals column not found -- ledger predates this diagnostic; "
              "re-run after SOCCERMODEL.ipynb has logged at least one play with it.")
        return
    have_proj = raw.dropna(subset=["proj_total_goals", "actual_home_goals", "actual_away_goals"]).copy()
    have_proj = have_proj[have_proj["proj_total_goals"] != ""]
    if have_proj.empty:
        print("  0 graded rows have a real proj_total_goals value yet -- this column "
              "was only just added (2026-09-27). Nothing to report until new plays "
              "grade in; do NOT proceed to any projection-scale change without this.")
        return
    have_proj["proj_total_goals"] = pd.to_numeric(have_proj["proj_total_goals"], errors="coerce")
    have_proj["actual_total_goals"] = (
        pd.to_numeric(have_proj["actual_home_goals"], errors="coerce")
        + pd.to_numeric(have_proj["actual_away_goals"], errors="coerce")
    )
    have_proj["bias"] = have_proj["proj_total_goals"] - have_proj["actual_total_goals"]
    have_proj = have_proj.dropna(subset=["bias"])
    if have_proj.empty:
        print("  Rows had proj_total_goals but no real actual-goals data to compare against.")
        return
    print(f"  n={len(have_proj)} graded row(s) with a real projected total on record\n")
    by_league = have_proj.groupby("competition_key")["bias"].agg(["mean", "count"]).sort_values("count", ascending=False)
    for league, row in by_league.iterrows():
        n = int(row["count"])
        flag = "  ** n<5, directional only" if n < 5 else ""
        print(f"  {str(league):<14} mean(proj-actual)={row['mean']:+.2f} goals  (n={n}){flag}")
    overall_mean = have_proj["bias"].mean()
    print(f"\n  Overall mean(proj-actual): {overall_mean:+.2f} goals (n={len(have_proj)})")
    if len(have_proj) < 20:
        print("  ** Below 20 total graded rows -- this is a diagnostic tripwire, not yet "
              "a number to calibrate a projection-scale fix against.")


def push_line_recheck():
    """SOCCER-PUSH-1 check (2026-10-05 user request -- "rerun soccer_diagnostic.py to see how much of the
    Under overconfidence disappears"). The ledger stores the OLD model probability, which on a whole line
    counted a push as an Under WIN (Under = P(total <= L)) and as an Over LOSS, and scored quarter lines as
    expected half-wins. For every graded O/U row: invert the stored probability to the Poisson total-goals
    rate it implies under the old formula, recompute the push-excluded probability
    (whole L: P(<L)/(1-P(=L)) for Under, P(>L)/(1-P(=L)) for Over), and compare to the real hit rate with
    pushes excluded. Quarter lines are reported separately -- the card no longer prices them."""
    import numpy as np
    from scipy.stats import poisson
    from scipy.optimize import brentq
    df, g = load_graded()
    ou = g[g["play_type"].astype(str).str.contains("O/U")].copy()
    ou["line"] = pd.to_numeric(ou["pick"].astype(str).str.extract(r"([\d.]+)\s*$")[0], errors="coerce")
    ou["side"] = np.where(ou["pick"].astype(str).str.upper().str.startswith("UNDER"), "UNDER", "OVER")
    ou["model"] = pd.to_numeric(ou["model"], errors="coerce")
    ou = ou.dropna(subset=["line", "model"])

    def old_over(lam, L):
        n = int(np.floor(L)); fr = round(L - n, 2)
        gt = 1 - poisson.cdf(n, lam)
        if fr == 0.25:
            return gt + 0.5 * poisson.pmf(n, lam)
        if fr == 0.75:
            return (1 - poisson.cdf(n + 1, lam)) + 0.5 * poisson.pmf(n + 1, lam)
        return gt

    def new_p(lam, L, side):
        n = int(np.floor(L))
        if L == n:
            push = poisson.pmf(n, lam)
            over = (1 - poisson.cdf(n, lam)) / (1 - push)
        else:
            over = 1 - poisson.cdf(n, lam)
        return over if side == "OVER" else 1 - over

    rows = []
    for r in ou.itertuples(index=False):
        target_over = r.model if r.side == "OVER" else 1 - r.model
        try:
            lam = brentq(lambda x: old_over(x, r.line) - target_over, 0.05, 12)
        except ValueError:
            continue
        kind = "quarter" if abs((r.line * 4) % 2) == 1 else ("whole" if r.line == int(r.line) else "half")
        rows.append({"kind": kind, "side": r.side, "old": r.model,
                     "new": new_p(lam, r.line, r.side) if kind != "quarter" else np.nan, "result": r.result})
    t = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print("PUSH-EXCLUDED RE-CHECK (SOCCER-PUSH-1) -- graded O/U ledger rows, old vs push-excluded model prob")
    print("=" * 78)
    print("  hit rate excludes pushes; 'gap' = mean model prob - real hit rate (positive = overconfident)")
    for side in ("UNDER", "OVER"):
        for kind in ("whole", "half", "quarter"):
            x = t[(t.side == side) & (t.kind == kind)]
            xg = x[x.result.isin(["WIN", "LOSS"])]
            if not len(xg):
                continue
            hit = (xg.result == "WIN").mean()
            line = (f"  {side:<5} {kind:<7} n={len(xg):<4} pushes {int((x.result == 'PUSH').sum()):<3} real {hit:.1%} | "
                    f"old model {xg.old.mean():.1%} (gap {xg.old.mean() - hit:+.1%})")
            line += (f" | push-excluded {xg.new.mean():.1%} (gap {xg.new.mean() - hit:+.1%})" if kind != "quarter"
                     else " | (no longer priced)")
            print(line)
    for side in ("UNDER", "OVER"):
        keep = t[(t.side == side) & (t.kind != "quarter") & t.result.isin(["WIN", "LOSS"])]
        allg = t[(t.side == side) & t.result.isin(["WIN", "LOSS"])]
        if len(keep):
            print(f"  {side} all lines (old card, n={len(allg)}): gap {allg.old.mean() - (allg.result == 'WIN').mean():+.1%}  ->  "
                  f"half+whole only, push-excluded (n={len(keep)}): gap {keep.new.mean() - (keep.result == 'WIN').mean():+.1%}")


def full_stack_recheck(write=False):
    """SOCCER-DIAG-STACK-1 (2026-10-06 user request -- "rerun soccer_diagnostic.py on the full stack (push fix,
    shrink, moneyline fix) before reading any soccer shadow results. Everything graded before this week ran
    on the old math"). Every graded ledger row in a league the Dixon-Coles model covers is RE-PRICED with
    today's stack and graded from the actual score:
      lambdas : prior-only DC walk-forward (tracking/soccer_dc_walkforward.csv)
      totals  : SOCCER-TOT-CALIB-1 (L_recent 120d + 0.60 x (DC - Ldc_recent)), push-excluded whole lines,
                quarter lines dropped (not priced any more)
      AH      : raw DC grid (SOCCER-SIDEGRID-1), push-excluded whole handicaps, quarter dropped
      ML      : DC 1X2 at shadow w=1 (the SOCCER-BLEND-SERIES-1 fix)
    Reports, per market/side: n, mean OLD ledger prob, mean NEW prob, real hit rate (pushes excluded)."""
    import re as _re, json as _json
    import numpy as np
    sys.path.insert(0, str(ROOT / "scripts"))
    import soccer_dc_model as dcm
    import shadow_log as SL
    wf = pd.read_csv(ROOT / "sports" / "SOCCER" / "tracking" / "soccer_dc_walkforward.csv", parse_dates=["date"])
    cal = _json.loads((ROOT / "sports" / "SOCCER" / "tracking" / "soccer_total_calib.json").read_text())
    wf["pt"] = wf.lh + wf.la
    def nrm(x):
        return SL.norm_name(str(x).lower().replace("l.p.", " la plata ").replace("s.j.", " san juan "))
    wf["_h"], wf["_a"] = wf["home"].map(nrm), wf["away"].map(nrm)
    df, g = load_graded()
    g = g.copy()
    g["cdt"] = pd.to_datetime(g["card_date"], errors="coerce")
    lg_map = {"serie_a": "seriea", "efl_champ": "championship"}
    def find(r):
        lg = lg_map.get(str(r.competition_key), str(r.competition_key))
        c = wf[(wf.competition_key == lg) & ((wf.date - r.cdt).abs() <= pd.Timedelta(days=1))]
        if c.empty:
            return None
        h, a = nrm(r.team_a), nrm(r.team_b)
        def ok(x, y):
            return x == y or (len(x) >= 4 and (x in y or y in x)) or (set(x.split()) and (set(x.split()) <= set(y.split()) or set(y.split()) <= set(x.split())))
        m = c[[ok(h, x) and ok(a, y) for x, y in zip(c._h, c._a)]]
        return m.iloc[0] if len(m) == 1 else None
    def grid(lh, la):
        return dcm.score_grid(lh, la)
    def p_over(G, L):
        idx = np.add.outer(np.arange(G.shape[0]), np.arange(G.shape[1]))
        n = int(np.floor(L)); gt = G[idx > n].sum()
        if L == n:
            push = G[idx == n].sum(); return gt / (1 - push)
        return gt
    def p_ah(G, hcp, home_side):
        idx = np.subtract.outer(np.arange(G.shape[0]), np.arange(G.shape[1]))   # home - away
        mar = (idx + hcp) if home_side else (-idx + hcp)
        win, push = G[mar > 0].sum(), G[mar == 0].sum()
        return win / (1 - push)
    out, unmatched = [], 0
    for r in g.itertuples(index=False):
        pt = str(r.play_type); pick = str(r.pick)
        if pd.isna(r.actual_home_goals) or pd.isna(r.actual_away_goals):
            continue
        w = find(r)
        if w is None:
            unmatched += 1
            continue
        hg, ag = float(r.actual_home_goals), float(r.actual_away_goals)
        lm = _re.search(r"([+-]?\d+(?:\.\d+)?)\s*$", pick)
        if pt == "O/U" and lm:
            L = float(lm.group(1))
            if abs((L * 4) % 2) == 1:
                out.append(("O/U", "quarter (dropped)", r.model, np.nan, None, r.market)); continue
            win_ = wf[(wf.competition_key == w.competition_key) & (wf.date < w.date) & (wf.date >= w.date - pd.Timedelta(days=cal["window_days"]))]
            s = 1.0
            if len(win_) >= cal["min_games"]:
                L_r, L_dc = (win_.hg + win_.ag).mean(), win_.pt.mean()
                s = max(0.6, L_r + cal["slope_b"] * (w.pt - L_dc)) / w.pt
            G = grid(w.lh * s, w.la * s); po = p_over(G, L)
            side = "UNDER" if pick.upper().startswith("UNDER") else "OVER"
            pnew = po if side == "OVER" else 1 - po
            tot = hg + ag
            y = None if tot == L else float((tot > L) if side == "OVER" else (tot < L))
            out.append(("O/U", side + (" whole" if L == int(L) else " half"), r.model, pnew, y, r.market))
        elif pt.startswith("AH") and lm:
            hcp = float(lm.group(1))
            if abs((hcp * 4) % 2) == 1:
                out.append(("AH", "quarter (dropped)", r.model, np.nan, None, r.market)); continue
            team = pick[:lm.start()].strip()
            home_side = nrm(team) == nrm(r.team_a) or (len(nrm(team)) >= 4 and nrm(team) in nrm(r.team_a))
            G = grid(w.lh, w.la); pnew = p_ah(G, hcp, home_side)
            mar = (hg - ag) if home_side else (ag - hg)
            y = None if mar + hcp == 0 else float(mar + hcp > 0)
            out.append(("AH", ("underdog (+)" if hcp > 0 else "favorite (-)" if hcp < 0 else "level (0)"), r.model, pnew, y, r.market))
        elif pt == "ML":
            G = grid(w.lh, w.la); mk = dcm.grid_markets(G)
            low = pick.lower()
            if "draw" in low:
                pnew, y = mk["p_draw"], float(hg == ag); sd = "draw"
            elif nrm(r.team_a) and nrm(r.team_a).split()[0] in nrm(pick):
                pnew, y = mk["p_home"], float(hg > ag); sd = "home"
            else:
                pnew, y = mk["p_away"], float(ag > hg); sd = "away"
            out.append(("ML", sd, r.model, pnew, y, r.market))
    t = pd.DataFrame(out, columns=["market", "side", "old", "new", "y", "mkt"])
    t["mkt"] = pd.to_numeric(t["mkt"], errors="coerce")
    print("\n" + "=" * 78)
    print("FULL-STACK RE-CHECK (SOCCER-DIAG-STACK-1) -- graded ledger rows re-priced with today's math")
    print("=" * 78)
    print(f"  matched {len(t)} graded row(s) to the DC walk-forward ({unmatched} unmatched: league not in the DC fit, e.g. MLS/cups,"
          " or a name the matcher couldn't pair)")
    print("  gap = mean prob - real hit rate (positive = overconfident); pushes excluded")
    for (mkt, side), x in t.groupby(["market", "side"]):
        xg = x[x.y.notna()]
        if "quarter" in side:
            print(f"  {mkt:<4} {side:<20} n={len(x):<4} -- no longer priced")
            continue
        if not len(xg):
            continue
        hit = xg.y.mean()
        print(f"  {mkt:<4} {side:<20} n={len(xg):<4} real {hit:.1%} | old {pd.to_numeric(xg.old, errors='coerce').mean():.1%} "
              f"(gap {pd.to_numeric(xg.old, errors='coerce').mean() - hit:+.1%}) | new stack {xg.new.mean():.1%} (gap {xg.new.mean() - hit:+.1%}) | no-vig mkt {xg.mkt.mean():.1%} (gap {xg.mkt.mean() - hit:+.1%})")
    xg = t[t.y.notna() & t.new.notna()]
    if len(xg):
        print(f"  ALL priced rows n={len(xg)}: old gap {pd.to_numeric(xg.old, errors='coerce').mean() - xg.y.mean():+.1%} | "
              f"new-stack gap {xg.new.mean() - xg.y.mean():+.1%} | mkt gap {xg.mkt.mean() - xg.y.mean():+.1%} | Brier new-stack {((xg.new - xg.y) ** 2).mean():.4f} vs mkt {((xg.mkt - xg.y) ** 2).mean():.4f} vs old {((pd.to_numeric(xg.old, errors='coerce') - xg.y) ** 2).mean():.4f}")
    print("  NOTE: these are the plays the OLD model selected -- re-pricing them shows calibration on that selected set;"
          " the unselected full slate is what shadow_fullslate.csv measures going forward.")

    # SOCCER-CALIB-HAIRCUT-1 (2026-10-06 user request -- "print a per-market calibration haircut on the soccer
    # card, sized to the measured excess and shrunk for sample size ... n/(n+50) ... let full_stack_recheck()
    # refresh them weekly as the shadow log grows"). Excess = new-stack gap minus the no-vig market's gap on the
    # SAME rows (the market's own miss on these selected plays is luck/selection the model shouldn't be charged
    # for). haircut = max(0, excess) x n/(n+50), in probability points, subtracted from the edge in Cells 23/24.
    # Rows: re-priced ledger rows above + shown (non-below-bar) soccer shadow_log rows graded W/L since the
    # stack was completed (2026-10-07), which already carry new-stack model probs.
    t2 = t[t.y.notna() & t.new.notna() & t.mkt.notna()][["market", "side", "new", "mkt", "y"]].copy()
    n_sl = 0
    try:
        sl = pd.read_csv(ROOT / "tracking" / "shadow_log.csv")
        sl = sl[(sl["sport"] == "SOCCER") & (sl["card_date"] >= "2026-10-07") & sl["result"].isin(["W", "L"])
                & (sl["tier"].astype(str) != "BELOW BAR") & sl["market"].isin(["AH", "TOTAL"])]
        rows = []
        for r in sl.itertuples(index=False):
            ln = float(r.line)
            if r.market == "TOTAL":
                mk, side = "O/U", str(r.side).upper() + (" whole" if ln == int(ln) else " half")
            else:
                mk, side = "AH", ("underdog (+)" if ln > 0 else "favorite (-)" if ln < 0 else "level (0)")
            rows.append((mk, side, float(r.model_prob), float(r.market_novig), float(r.result == "W")))
        if rows:
            t2 = pd.concat([t2, pd.DataFrame(rows, columns=["market", "side", "new", "mkt", "y"])], ignore_index=True)
        n_sl = len(rows)
    except Exception as e:
        print(f"  (shadow_log rows not added: {e})")
    K = 50
    hc = {}
    print()
    print(f"  CALIBRATION HAIRCUT (excess over market gap x n/(n+{K})) -- {len(t2) - n_sl} ledger + {n_sl} shadow row(s)")
    for (mkt, side), x in t2.groupby(["market", "side"]):
        n = len(x)
        excess = (x.new.mean() - x.y.mean()) - (x.mkt.mean() - x.y.mean())
        cut = max(0.0, excess) * n / (n + K)
        hc[f"{mkt}|{side}"] = {"haircut": round(cut, 4), "excess": round(excess, 4), "n": n}
        print(f"    {mkt:<4} {side:<14} n={n:<4} excess {excess:+.1%} -> haircut {cut:.1%}")
    if write:
        import datetime as _dt
        out_p = ROOT / "sports" / "SOCCER" / "tracking" / "soccer_calib_haircut.json"
        out_p.write_text(_json.dumps({"updated": str(_dt.date.today()), "k": K,
                                      "rule": "haircut = max(0, new-stack gap - market gap) x n/(n+k); subtracted from edge",
                                      "categories": hc}, indent=1))
        print(f"  wrote {out_p}")
    return hc


if __name__ == "__main__":
    main()
    push_line_recheck()
    full_stack_recheck(write="--write-haircut" in sys.argv)
