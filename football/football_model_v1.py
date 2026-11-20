"""
football_model_v1.py
Phase 1: baseline model for Premier League (E0) + Championship (E1).

Data: free CSVs from football-data.co.uk (results + bookmaker odds).
Run:  pip install pandas numpy scikit-learn
      python football_model_v1.py

What it does
1. Downloads results/odds (cached in ./data).
2. Builds leak-free features (only information available BEFORE each match).
3. Walk-forward backtest: train on past seasons, test on the next one.
4. Compares:
     Model A = our features only (Elo, form, home/away form, head-to-head)
     Model B = our features + bookmaker opening odds
     Market  = bookmaker closing odds (the benchmark to beat)
5. Metrics: log loss, Brier score, accuracy, and a rough value-bet ROI.
"""
import math
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEASONS = ["1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526", "2627"]
TEST_SEASONS = ["2223", "2324", "2425", "2526", "2627"]
LEAGUES = ["E0", "E1"]  # Premier League, Championship
DATA = Path("data")

OPEN_SETS = [("PSH", "PSD", "PSA"), ("AvgH", "AvgD", "AvgA"), ("B365H", "B365D", "B365A")]
CLOSE_SETS = [("PSCH", "PSCD", "PSCA"), ("AvgCH", "AvgCD", "AvgCA"), ("B365CH", "B365CD", "B365CA")]
BET_SETS = [("B365H", "B365D", "B365A"), ("AvgH", "AvgD", "AvgA")]

HFA, K = 65, 20  # Elo home advantage and K-factor


# ---------------------------------------------------------------- data
def load():
    DATA.mkdir(exist_ok=True)
    frames = []
    for s in SEASONS:
        for lg in LEAGUES:
            f = DATA / f"{lg}_{s}.csv"
            if not f.exists():
                url = f"https://www.football-data.co.uk/mmz4281/{s}/{lg}.csv"
                try:
                    pd.read_csv(url, encoding="latin-1").to_csv(f, index=False)
                except Exception as e:
                    print("skip", url, e)
                    continue
            d = pd.read_csv(f)
            d = d.dropna(subset=["HomeTeam", "AwayTeam", "FTR"])
            d["Date"] = pd.to_datetime(d["Date"], dayfirst=True)
            d["league"], d["season"] = lg, s
            frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values("Date", kind="stable").reset_index(drop=True)


def odds_from(df, sets, normalize):
    """First available odds set per row. Returns raw odds or margin-free probabilities."""
    out = np.full((len(df), 3), np.nan)
    for cols in sets:
        if not all(c in df.columns for c in cols):
            continue
        o = df[list(cols)].to_numpy(dtype=float)
        if normalize:
            o = 1.0 / o
            o = o / o.sum(axis=1, keepdims=True)
        mask = np.isnan(out[:, 0]) & ~np.isnan(o[:, 0])
        out[mask] = o[mask]
    return out


# ---------------------------------------------------------------- features
def build_features(df):
    elo, seen = {}, set()
    last = defaultdict(lambda: deque(maxlen=5))        # (pts, gf, ga) all venues
    last_home = defaultdict(lambda: deque(maxlen=5))   # pts at home
    last_away = defaultdict(lambda: deque(maxlen=5))   # pts away
    h2h = defaultdict(lambda: deque(maxlen=5))         # (home_team, FTR)

    def avg(seq, i, default):
        return float(np.mean([x[i] if isinstance(x, tuple) else x for x in seq])) if seq else default

    rows, cur = [], None
    for r in df.itertuples():
        if r.season != cur:
            if cur is not None:  # new season: shrink ratings toward the mean
                for t in elo:
                    elo[t] = 1500 + 0.7 * (elo[t] - 1500)
            cur = r.season
        h, a = r.HomeTeam, r.AwayTeam
        for t in (h, a):
            if t not in seen:
                seen.add(t)
                elo[t] = 1500.0 if r.season == SEASONS[0] else 1450.0  # newly promoted

        key = frozenset((h, a))
        h2h_pts = []
        for ht, res in h2h[key]:
            same = ht == h
            pts = {"H": 3, "D": 1, "A": 0}[res] if same else {"H": 0, "D": 1, "A": 3}[res]
            h2h_pts.append(pts)

        rows.append({
            "elo_diff": elo[h] - elo[a],
            "form_h": avg(last[h], 0, 1.35), "form_a": avg(last[a], 0, 1.35),
            "gf_h": avg(last[h], 1, 1.35), "ga_h": avg(last[h], 2, 1.35),
            "gf_a": avg(last[a], 1, 1.35), "ga_a": avg(last[a], 2, 1.35),
            "home_form_h": avg(last_home[h], 0, 1.5), "away_form_a": avg(last_away[a], 0, 1.1),
            "h2h_h": float(np.mean(h2h_pts)) if h2h_pts else 1.35,
            "is_champ": 1.0 if r.league == "E1" else 0.0,
        })

        # ---- update state AFTER the match
        ph, pa = {"H": (3, 0), "D": (1, 1), "A": (0, 3)}[r.FTR]
        last[h].append((ph, r.FTHG, r.FTAG)); last[a].append((pa, r.FTAG, r.FTHG))
        last_home[h].append(ph); last_away[a].append(pa)
        h2h[key].append((h, r.FTR))
        exp_h = 1 / (1 + 10 ** (-(elo[h] - elo[a] + HFA) / 400))
        score_h = {"H": 1.0, "D": 0.5, "A": 0.0}[r.FTR]
        gd = abs(r.FTHG - r.FTAG)
        delta = K * (1 + math.log(1 + gd)) * (score_h - exp_h)
        elo[h] += delta; elo[a] -= delta
    return pd.concat([df, pd.DataFrame(rows)], axis=1)


# ---------------------------------------------------------------- evaluation
def log_loss_(p, y):
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))))


def brier_(p, y):
    oh = np.eye(3)[y]
    return float(np.mean(np.sum((p - oh) ** 2, axis=1)))


def acc_(p, y):
    return float(np.mean(p.argmax(axis=1) == y))


def roi(p, odds, y, thr):
    edge = p * odds - 1
    bets = edge > thr
    n = int(bets.sum())
    if n == 0:
        return n, float("nan")
    won = np.eye(3)[y] == 1
    profit = np.where(won, odds - 1, -1.0)[bets].sum()
    return n, float(profit / n)


def walk_forward(df, feats, y):
    preds = pd.DataFrame(np.nan, index=df.index, columns=[0, 1, 2])
    for s in TEST_SEASONS:
        tr = (df.season < s) & (df.season > SEASONS[0])  # first season = Elo warm-up
        te = df.season == s
        if tr.sum() == 0 or te.sum() == 0:
            continue
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        m.fit(df.loc[tr, feats], y[tr])
        preds.loc[te] = m.predict_proba(df.loc[te, feats])
    return preds.to_numpy()


def main():
    df = load()
    df = build_features(df)
    for i, c in enumerate(["mo_h", "mo_d", "mo_a"]):
        df[c] = odds_from(df, OPEN_SETS, True)[:, i]
    close = odds_from(df, CLOSE_SETS, True)
    bet_odds = odds_from(df, BET_SETS, False)
    df = df.assign(c_h=close[:, 0], c_d=close[:, 1], c_a=close[:, 2],
                   b_h=bet_odds[:, 0], b_d=bet_odds[:, 1], b_a=bet_odds[:, 2])
    df = df.dropna(subset=["mo_h", "c_h", "b_h"]).reset_index(drop=True)
    y = df["FTR"].map({"H": 0, "D": 1, "A": 2}).to_numpy()

    feats_a = ["elo_diff", "form_h", "form_a", "gf_h", "ga_h", "gf_a", "ga_a",
               "home_form_h", "away_form_a", "h2h_h", "is_champ"]
    feats_b = feats_a + ["mo_h", "mo_d", "mo_a"]

    pa = walk_forward(df, feats_a, pd.Series(y))
    pb = walk_forward(df, feats_b, pd.Series(y))
    mask = ~np.isnan(pa[:, 0])
    y_t, odds = y[mask], df.loc[mask, ["b_h", "b_d", "b_a"]].to_numpy()
    sets = {
        "A  (our features)": pa[mask],
        "B  (features+odds)": pb[mask],
        "Market opening": df.loc[mask, ["mo_h", "mo_d", "mo_a"]].to_numpy(),
        "Market closing": df.loc[mask, ["c_h", "c_d", "c_a"]].to_numpy(),
    }
    print(f"\nTest matches: {mask.sum()}  (seasons {TEST_SEASONS[0]}-{TEST_SEASONS[-1]})")
    print(f"{'model':22s} {'logloss':>8s} {'brier':>8s} {'acc':>7s}")
    for name, p in sets.items():
        print(f"{name:22s} {log_loss_(p, y_t):8.4f} {brier_(p, y_t):8.4f} {acc_(p, y_t):7.1%}")

    print("\nValue-bet test (flat 1 unit, vs B365 pre-match odds): bets / ROI per bet")
    for name in ["A  (our features)", "B  (features+odds)"]:
        line = [f"edge>{t:.0%}: {roi(sets[name], odds, y_t, t)[0]:4d} / {roi(sets[name], odds, y_t, t)[1]:+.1%}"
                for t in (0.03, 0.05, 0.10)]
        print(f"{name:22s} " + "   ".join(line))
    print("\nLower logloss/brier = better. To beat the market, a model must get below 'Market closing'.")


if __name__ == "__main__":
    main()
