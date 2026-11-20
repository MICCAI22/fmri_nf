"""
evaluate_predictions.py - score saved forward-test predictions once matches are played.

Run (from this folder):  python evaluate_predictions.py

- Reads every predictions/*_E0*.csv (never modifies them).
- For each match keeps the EARLIEST prediction made before kick-off
  (kick-off = date + time in UK local time, as in fixtures.csv).
- Downloads the latest E0 results for the current season and matches on date/home/away.
- Reports log loss / Brier for Model A, Model B, the market at prediction time,
  and the closing market from the results file, all on the same matches.
- Writes predictions/evaluated/evaluation_<run date>.csv (new file each run).
"""
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import football_model_v1 as fm

warnings.filterwarnings("ignore")

PRED_DIR = Path("predictions")
DIV = "E0"


def load_predictions():
    files = sorted(p for p in PRED_DIR.glob(f"*_{DIV}*.csv"))
    if not files:
        return pd.DataFrame()
    p = pd.concat([pd.read_csv(f).assign(file=f.name) for f in files], ignore_index=True)
    p["ts"] = pd.to_datetime(p.timestamp_utc, utc=True)
    ko = pd.to_datetime(p.date + " " + p.time.fillna("23:59").astype(str), format="%Y-%m-%d %H:%M")
    p["kickoff"] = ko.dt.tz_localize("Europe/London", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    late = p.ts >= p.kickoff
    if late.any():
        print(f"Ignoring {int(late.sum())} prediction rows made at/after kick-off")
    p = p[~late].sort_values("ts")
    return p.drop_duplicates(["date", "home", "away"], keep="first")


def load_results():
    season = fm.SEASONS[-1]
    url = f"https://www.football-data.co.uk/mmz4281/{season}/{DIV}.csv"
    r = pd.read_csv(url, encoding="latin-1")
    r.to_csv(fm.DATA / f"{DIV}_{season}.csv", index=False)  # refresh cache
    r = r.dropna(subset=["HomeTeam", "AwayTeam", "FTR"])
    r["date"] = pd.to_datetime(r.Date, dayfirst=True).dt.strftime("%Y-%m-%d")
    close = fm.odds_from(r, fm.CLOSE_SETS, True)
    r["close_H"], r["close_D"], r["close_A"] = close[:, 0], close[:, 1], close[:, 2]
    return r.rename(columns={"HomeTeam": "home", "AwayTeam": "away"})[
        ["date", "home", "away", "FTHG", "FTAG", "FTR", "close_H", "close_D", "close_A"]]


def main():
    p = load_predictions()
    if p.empty:
        print("No prediction files found in", PRED_DIR)
        return
    res = load_results()
    m = p.merge(res, on=["date", "home", "away"], how="left")
    done = m[m.FTR.notna()].copy()
    print(f"Predicted matches: {len(m)}   played: {len(done)}   pending: {len(m) - len(done)}")
    if done.empty:
        return
    y = done.FTR.map({"H": 0, "D": 1, "A": 2}).to_numpy()

    def P(prefix):
        return done[[f"{prefix}_H", f"{prefix}_D", f"{prefix}_A"]].to_numpy(dtype=float)

    sets = {"Model A": P("A"), "Model B": P("B"), "Market @prediction": P("mkt"), "Market closing": P("close")}
    common = np.all([~np.isnan(v).any(axis=1) for v in sets.values()], axis=0)
    print(f"Scored on {int(common.sum())} matches with all four sets available "
          f"(B needs odds, closing needs results file odds)\n")
    print(f"{'set':20s} {'logloss':>8s} {'brier':>8s}")
    for k, v in sets.items():
        if common.any():
            print(f"{k:20s} {fm.log_loss_(v[common], y[common]):8.4f} {fm.brier_(v[common], y[common]):8.4f}")
    for k in ("Model A", "Model B"):
        ok = ~np.isnan(sets[k]).any(axis=1)
        if ok.sum() != common.sum() and ok.any():
            print(f"{k} on all its {int(ok.sum())} matches: logloss {fm.log_loss_(sets[k][ok], y[ok]):.4f}")
    if common.sum() < 200:
        print("\nNote: small sample - differences of a few 0.01 in log loss are noise at this size.")

    for k, v in sets.items():  # per-match log loss for later pooling
        done[f"ll_{k.split()[0]}_{k.split()[-1][:4]}"] = -np.log(np.clip(v[np.arange(len(y)), y], 1e-12, 1))
    out_dir = PRED_DIR / "evaluated"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"evaluation_{datetime.now(timezone.utc):%Y-%m-%d_%H%M%S}.csv"
    done.drop(columns=["ts", "kickoff"]).to_csv(path, index=False, float_format="%.4f")
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
