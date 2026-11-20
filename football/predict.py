"""
predict.py - forward-test predictions for upcoming Premier League (E0) fixtures.

Run (from this folder):  python predict.py

1. Refreshes the current-season results (E0+E1) and downloads fixtures.csv.
2. Trains Model A and Model B exactly as in football_model_v1.py, on ALL completed
   matches up to today (first season = Elo warm-up, as in the backtest).
3. Computes features for each upcoming fixture from the state as of today
   (fixtures never update Elo/form, so there is no leakage).
4. Saves predictions/YYYY-MM-DD_E0.csv (+ a raw copy of fixtures.csv).
   Existing files are NEVER overwritten: a second run on the same day gets a
   _HHMMSS suffix. Evaluate later with evaluate_predictions.py.
"""
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import football_model_v1 as fm

warnings.filterwarnings("ignore")

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
PRED_DIR = Path("predictions")
DIV = "E0"
FEATS_A = ["elo_diff", "form_h", "form_a", "gf_h", "ga_h", "gf_a", "ga_a",
           "home_form_h", "away_form_a", "h2h_h", "is_champ"]
FEATS_B = FEATS_A + ["mo_h", "mo_d", "mo_a"]
MODEL_B_NAME = "B (features + opening odds, v1)"  # no blend version exists yet


def refresh_current_season():
    """Drop cached current-season files so load() re-downloads the latest results."""
    for lg in fm.LEAGUES:
        f = fm.DATA / f"{lg}_{fm.SEASONS[-1]}.csv"
        if f.exists():
            f.unlink()


def new_model():
    # identical to fm.walk_forward
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def unique_path(base: Path) -> Path:
    if not base.exists():
        return base
    stamp = datetime.now(timezone.utc).strftime("%H%M%S")
    return base.with_name(f"{base.stem}_{stamp}{base.suffix}")


def main():
    now = datetime.now(timezone.utc)
    today = pd.Timestamp(now.date())
    PRED_DIR.mkdir(exist_ok=True)
    (PRED_DIR / "raw").mkdir(exist_ok=True)

    # ---- data
    refresh_current_season()
    hist = fm.load()
    hist = hist[hist.Date < today]  # completed matches only
    raw = pd.read_csv(FIXTURES_URL, encoding="utf-8-sig")
    raw.to_csv(unique_path(PRED_DIR / "raw" / f"fixtures_{today:%Y-%m-%d}.csv"), index=False)
    fx = raw[raw.Div == DIV].copy()
    fx["Date"] = pd.to_datetime(fx["Date"], dayfirst=True)
    fx = fx[fx.Date >= today]
    if fx.empty:
        print("No upcoming E0 fixtures in fixtures.csv")
        return
    fx["league"], fx["season"] = DIV, fm.SEASONS[-1]
    for c in ("FTR", "FTHG", "FTAG"):
        fx[c] = np.nan
    fx["_fixture"] = True
    hist["_fixture"] = False

    known = set(hist.HomeTeam) | set(hist.AwayTeam)
    for t in sorted((set(fx.HomeTeam) | set(fx.AwayTeam)) - known):
        print(f"WARNING: team '{t}' not in history (default Elo 1450, default form)")

    # ---- features (fixtures sorted after all history; they never update state)
    df = pd.concat([hist, fx], ignore_index=True)
    df = df.sort_values(["_fixture", "Date"], kind="stable").reset_index(drop=True)
    df = fm.build_features(df)
    mo = fm.odds_from(df, fm.OPEN_SETS, True)
    df["mo_h"], df["mo_d"], df["mo_a"] = mo[:, 0], mo[:, 1], mo[:, 2]
    close = fm.odds_from(df, fm.CLOSE_SETS, True)
    bet = fm.odds_from(df, fm.BET_SETS, False)
    src = np.full(len(df), "", dtype=object)
    for cols in reversed(fm.OPEN_SETS):  # which odds set fed mo_* (first available wins)
        if all(c in df.columns for c in cols):
            src[df[cols[0]].notna().to_numpy()] = cols[0][:-1]

    # ---- training set: same filtering as fm.main()
    h = df[~df._fixture].copy()
    h["c_h"], h["b_h"] = close[~df._fixture.to_numpy(), 0], bet[~df._fixture.to_numpy(), 0]
    tr = h.dropna(subset=["mo_h", "c_h", "b_h"])
    tr = tr[tr.season > fm.SEASONS[0]]
    y = tr.FTR.map({"H": 0, "D": 1, "A": 2}).to_numpy()
    ma = new_model().fit(tr[FEATS_A], y)
    mb = new_model().fit(tr[FEATS_B], y)

    # ---- predict
    f = df[df._fixture].copy()
    f["odds_src"] = src[df._fixture.to_numpy()]
    pa = ma.predict_proba(f[FEATS_A])
    has_odds = f[["mo_h", "mo_d", "mo_a"]].notna().all(axis=1).to_numpy()
    pb = np.full((len(f), 3), np.nan)
    if has_odds.any():
        pb[has_odds] = mb.predict_proba(f.loc[has_odds, FEATS_B])
    for _, r in f[~has_odds].iterrows():
        print(f"NO ODDS: {r.Date:%Y-%m-%d} {r.HomeTeam} v {r.AwayTeam} -> Model B skipped")

    mkt = f[["mo_h", "mo_d", "mo_a"]].to_numpy()
    out = pd.DataFrame({
        "timestamp_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "date": f.Date.dt.strftime("%Y-%m-%d").to_numpy(),
        "time": f["Time"].to_numpy() if "Time" in f else "",
        "home": f.HomeTeam.to_numpy(), "away": f.AwayTeam.to_numpy(),
    })
    for k, lab in enumerate("HDA"):
        out[f"A_{lab}"] = pa[:, k]
    for k, lab in enumerate("HDA"):
        out[f"B_{lab}"] = pb[:, k]
    for k, lab in enumerate("HDA"):
        out[f"mkt_{lab}"] = mkt[:, k]
    for k, lab in enumerate("HDA"):
        out[f"A_minus_mkt_{lab}"] = pa[:, k] - mkt[:, k]
    for k, lab in enumerate("HDA"):
        out[f"B_minus_mkt_{lab}"] = pb[:, k] - mkt[:, k]
    out["mkt_source"] = f["odds_src"].to_numpy()
    for lab in "HDA":  # raw odds at prediction time, for later ROI/CLV
        for book in ("Avg", "B365", "Max"):
            c = f"{book}{lab}"
            out[c] = f[c].to_numpy() if c in f else np.nan
    out["elo_diff"] = f.elo_diff.to_numpy()
    out["model_B"] = MODEL_B_NAME
    out["train_matches"] = len(tr)
    out["train_through"] = f"{tr.Date.max():%Y-%m-%d}"

    path = unique_path(PRED_DIR / f"{today:%Y-%m-%d}_{DIV}.csv")
    with open(path, "x", newline="") as fh:  # "x" = fail rather than overwrite
        out.to_csv(fh, index=False, float_format="%.4f")

    # ---- console table
    print(f"\nTrained on {len(tr)} completed E0+E1 matches ({tr.Date.min():%Y-%m-%d} .. "
          f"{tr.Date.max():%Y-%m-%d}).  Market = margin-free {', '.join(sorted(set(out.mkt_source)))} odds.")
    print(f"Model B = {MODEL_B_NAME}\n")
    hdr = (f"{'date':10s} {'home':14s} {'away':14s} | {'A: H   D   A':>16s} | {'B: H   D   A':>16s} |"
           f" {'mkt: H  D   A':>16s} | {'B-mkt: H  D   A':>18s}")
    print(hdr); print("-" * len(hdr))
    pct = lambda v: "  - " if np.isnan(v) else f"{v*100:3.0f}%"
    sg = lambda v: "   - " if np.isnan(v) else f"{v*100:+4.0f}%"
    for _, r in out.iterrows():
        print(f"{r.date:10s} {r.home[:14]:14s} {r.away[:14]:14s} | "
              f"{pct(r.A_H)} {pct(r.A_D)} {pct(r.A_A)} | {pct(r.B_H)} {pct(r.B_D)} {pct(r.B_A)} | "
              f"{pct(r.mkt_H)} {pct(r.mkt_D)} {pct(r.mkt_A)} | "
              f"{sg(r.B_minus_mkt_H)} {sg(r.B_minus_mkt_D)} {sg(r.B_minus_mkt_A)}")
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
