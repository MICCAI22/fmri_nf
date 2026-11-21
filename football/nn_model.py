"""
nn_model.py - multi-task neural network, walk-forward round by round.

Predicts, for every match, from information available BEFORE kick-off:
  * 1X2 (softmax head)
  * full count distributions (negative binomial heads) for
      FT goals home/away, HT goals home/away, corners home/away, yellow cards home/away
  * derived markets: goals O/U 1.5/2.5/3.5, BTTS, exact total goals, HT 1X2,
      HT O/U 0.5/1.5, total corners O/U 8.5-11.5, total cards O/U 2.5-4.5, double chance

Walk-forward ("learn round by round"):
  - initial fit on all seasons before the first test season (epochs picked by a
    chronological hold-out inside the training data, never on test data);
  - for each round (Fri-Mon / Tue-Thu block) in the test period:
        predict the round -> store predictions -> reveal results
        -> fine-tune (warm start) on all matches seen so far -> next round;
  - full refit from scratch at the start of each test season.

Variants: NN-A = match statistics only; NN-B = NN-A + opening market odds (1X2, O/U 2.5).
Baselines: market (opening/closing odds) where odds exist, and a "naive" Poisson built
from the same rolling team averages (no learning) for markets without odds.

Run (from this folder):  python nn_model.py            # full backtest, writes nn_results/
                         python nn_model.py --quick    # 1 seed, for testing
"""
import argparse
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import nbinom, poisson

import football_model_v1 as fm

TEST_SEASONS = ["2223", "2324", "2425", "2526", "2627"]
OUT = Path("nn_results")
COUNT_TARGETS = ["FTHG", "FTAG", "HTHG", "HTAG", "HC", "AC", "HY", "AY"]
OU_OPEN_SETS = [("P>2.5", "P<2.5"), ("Avg>2.5", "Avg<2.5"), ("BbAv>2.5", "BbAv<2.5"), ("B365>2.5", "B365<2.5")]
OU_CLOSE_SETS = [("PC>2.5", "PC<2.5"), ("AvgC>2.5", "AvgC<2.5"), ("B365C>2.5", "B365C<2.5")]
STATS = [  # (name, home column, away column): "for" = own value, "against" = opponent's
    ("g", "FTHG", "FTAG"), ("s", "HS", "AS"), ("st", "HST", "AST"), ("c", "HC", "AC"),
    ("y", "HY", "AY"), ("f", "HF", "AF"), ("htg", "HTHG", "HTAG"),
]
HL_SHORT, HL_LONG, HL_VENUE, HL_REF = 5, 20, 8, 10  # EWMA half-lives (matches)
VALID_SEASONS = ["2021", "2122"]  # used only to choose hyper-parameters, never the test seasons
CFG = {"hidden": (64, 32), "drop": 0.5, "wd": 5e-2,  # chosen on VALID_SEASONS
       "lr": 1e-3,
       "upd_lr": 1e-4, "upd_replay": 3, "upd_steps": 2}


# =============================================================== features
def ew(old, x, hl):
    a = 0.5 ** (1 / hl)
    return a * old + (1 - a) * x


def build_features(df):
    """Sequential, leak-free: each row sees only matches played before it."""
    n_st = len(STATS)
    lg_avg = {lg: np.array([1.35, 12.0, 4.3, 5.2, 1.7, 10.5, 0.6]) for lg in fm.LEAGUES}  # init
    team = {}                       # team -> dict of state arrays
    elo = {}
    ref = defaultdict(lambda: None)  # referee -> EWMA total yellows
    last_date = {}
    rows, cur = [], None
    for r in df.itertuples(index=False):
        if r.season != cur:
            if cur is not None:
                for t in elo:
                    elo[t] = 1500 + 0.7 * (elo[t] - 1500)
            cur = r.season
        h, a, lg = r.HomeTeam, r.AwayTeam, r.league
        base = lg_avg[lg]
        for t in (h, a):
            if t not in team:
                newbie = r.season != fm.SEASONS[0]
                init = base * np.where(np.arange(n_st) == 0, 0.9 if newbie else 1.0, 1.0)
                team[t] = {"fs": init.copy(), "as": base.copy(), "fl": init.copy(), "al": base.copy(),
                           "hv": np.array([base[0], base[0], base[3], base[3]]),  # home venue gf,ga,cf,ca
                           "av": np.array([base[0], base[0], base[3], base[3]]), "n": 0}
                elo[t] = 1500.0 if not newbie else 1450.0
        th, ta = team[h], team[a]
        rest = lambda t: min((r.Date - last_date[t]).days, 21) if t in last_date else 14
        ref_name = getattr(r, "Referee", None)
        ref_y = ref[ref_name] if isinstance(ref_name, str) and ref[ref_name] is not None else None
        feat = {
            "elo_h": elo[h], "elo_a": elo[a], "elo_diff": elo[h] - elo[a],
            "rest_h": rest(h), "rest_a": rest(a),
            "n_h": math.log1p(th["n"]), "n_a": math.log1p(ta["n"]),
            "is_champ": float(lg == "E1"),
            "lg_g": base[0], "lg_c": base[3], "lg_y": base[4],
            "ref_y": ref_y if ref_y is not None else 2 * base[4], "ref_known": float(ref_y is not None),
        }
        for side, t in (("h", th), ("a", ta)):
            for i, (nm, _, _) in enumerate(STATS):
                feat[f"{nm}F_s_{side}"] = t["fs"][i]; feat[f"{nm}A_s_{side}"] = t["as"][i]
                feat[f"{nm}F_l_{side}"] = t["fl"][i]; feat[f"{nm}A_l_{side}"] = t["al"][i]
        for i, nm in enumerate(["gf", "ga", "cf", "ca"]):
            feat[f"hv_{nm}"] = th["hv"][i]; feat[f"av_{nm}"] = ta["av"][i]
        rows.append(feat)

        # ---- update state after the match (skip unplayed fixtures / missing stats)
        if not isinstance(r.FTR, str):
            continue
        xh = np.array([getattr(r, c) for _, c, _ in STATS], dtype=float)
        xa = np.array([getattr(r, c) for _, _, c in STATS], dtype=float)
        okm = ~(np.isnan(xh) | np.isnan(xa))
        for t, own, opp in ((th, xh, xa), (ta, xa, xh)):
            t["fs"][okm] = ew(t["fs"][okm], own[okm], HL_SHORT); t["as"][okm] = ew(t["as"][okm], opp[okm], HL_SHORT)
            t["fl"][okm] = ew(t["fl"][okm], own[okm], HL_LONG); t["al"][okm] = ew(t["al"][okm], opp[okm], HL_LONG)
            t["n"] += 1
        vh = np.array([xh[0], xa[0], xh[3], xa[3]]); va = np.array([xa[0], xh[0], xa[3], xh[3]])
        mh, ma = ~np.isnan(vh), ~np.isnan(va)
        th["hv"][mh] = ew(th["hv"][mh], vh[mh], HL_VENUE); ta["av"][ma] = ew(ta["av"][ma], va[ma], HL_VENUE)
        both = (xh + xa) / 2
        lg_avg[lg][okm] = ew(lg_avg[lg][okm], both[okm], 200)
        if isinstance(ref_name, str) and not np.isnan(r.HY + r.AY):
            ref[ref_name] = (r.HY + r.AY) if ref[ref_name] is None else ew(ref[ref_name], r.HY + r.AY, HL_REF)
        last_date[h] = last_date[a] = r.Date
        # Elo identical to v1
        exp_h = 1 / (1 + 10 ** (-(elo[h] - elo[a] + fm.HFA) / 400))
        score_h = {"H": 1.0, "D": 0.5, "A": 0.0}[r.FTR]
        delta = fm.K * (1 + math.log(1 + abs(r.FTHG - r.FTAG))) * (score_h - exp_h)
        elo[h] += delta; elo[a] -= delta
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def add_market(df):
    mo = fm.odds_from(df, fm.OPEN_SETS, True)
    mc = fm.odds_from(df, fm.CLOSE_SETS, True)
    df["mo_h"], df["mo_d"], df["mo_a"] = mo.T
    df["mc_h"], df["mc_d"], df["mc_a"] = mc.T

    def ou(sets):
        out = np.full(len(df), np.nan)
        for o, u in sets:
            if o in df.columns and u in df.columns:
                po, pu = 1 / df[o].to_numpy(float), 1 / df[u].to_numpy(float)
                p = po / (po + pu)
                m = np.isnan(out) & ~np.isnan(p)
                out[m] = p[m]
        return out
    df["ou_open"], df["ou_close"] = ou(OU_OPEN_SETS), ou(OU_CLOSE_SETS)
    df["ou_open_missing"] = df.ou_open.isna().astype(float)
    df["ou_open_f"] = df.ou_open.fillna(0.5)
    # rough raw odds for a value-bet check (opening average)
    for lab, cols in (("1x2", [("AvgH", "AvgD", "AvgA"), ("BbAvH", "BbAvD", "BbAvA"), ("B365H", "B365D", "B365A")]),):
        o = fm.odds_from(df, cols, False)
        df["odd_h"], df["odd_d"], df["odd_a"] = o.T
    oo = np.full(len(df), np.nan)
    for o, _ in OU_OPEN_SETS[1:]:
        if o in df.columns:
            v = df[o].to_numpy(float); m = np.isnan(oo) & ~np.isnan(v); oo[m] = v[m]
    df["odd_over"] = oo
    return df


# =============================================================== network
class Net(nn.Module):
    def __init__(self, n_in, hidden=(64, 32), drop=0.3):
        super().__init__()
        layers, d = [], n_in
        for hdim in hidden:
            layers += [nn.Linear(d, hdim), nn.GELU(), nn.Dropout(drop)]
            d = hdim
        self.trunk = nn.Sequential(*layers)
        self.head_1x2 = nn.Linear(d, 3)
        self.head_mu = nn.Linear(d, len(COUNT_TARGETS))
        self.log_r = nn.Parameter(torch.full((len(COUNT_TARGETS),), 2.5))  # NB dispersion
        for hd in (self.head_1x2, self.head_mu):
            nn.init.zeros_(hd.weight); nn.init.zeros_(hd.bias)

    def forward(self, x, off3, offc):
        """Residual heads: the net learns a CORRECTION to a prior.
        off3 = log market 1X2 probabilities (NN-B) or zeros (NN-A);
        offc = log of the naive rolling-average rate for each count target."""
        z = self.trunk(x)
        return self.head_1x2(z) + off3, self.head_mu(z) + offc


def nb_nll(y, log_mu, log_r):
    r = torch.exp(log_r).clamp(max=1e4)
    mu = torch.exp(log_mu)
    return -(torch.lgamma(y + r) - torch.lgamma(r) - torch.lgamma(y + 1)
             + r * (torch.log(r) - torch.log(r + mu)) + y * (log_mu - torch.log(r + mu)))


def loss_fn(model, x, off3, offc, y3, yc, mc):
    logits, log_mu = model(x, off3, offc)
    l1 = nn.functional.cross_entropy(logits, y3)
    lc = nb_nll(torch.nan_to_num(yc), log_mu, model.log_r) * mc
    return l1 + lc.sum() / mc.sum().clamp(min=1) * len(COUNT_TARGETS) / 4  # weight counts vs 1X2


class Member:
    """One network + its scaler. Fit from scratch or fine-tune (warm start)."""

    def __init__(self, feats, seed, offset=False):
        self.feats, self.seed, self.offset = feats, seed, offset

    def _off(self, d):
        offc = torch.tensor(np.log(np.clip(naive_rates(d), 0.05, None)), dtype=torch.float32)
        if not self.offset:
            return torch.zeros((len(d), 3)), offc
        off3 = np.log(np.clip(d[["mo_h", "mo_d", "mo_a"]].to_numpy(float), 1e-6, 1))
        return torch.tensor(off3, dtype=torch.float32), offc

    def _tensors(self, d):
        x = torch.tensor(((d[self.feats].to_numpy(float) - self.mu) / self.sd), dtype=torch.float32)
        y3 = torch.tensor(d.FTR.map({"H": 0, "D": 1, "A": 2}).to_numpy(), dtype=torch.long)
        yc_np = d[COUNT_TARGETS].to_numpy(float)
        return (x, *self._off(d), y3, torch.tensor(yc_np, dtype=torch.float32),
                torch.tensor(~np.isnan(yc_np), dtype=torch.float32))

    def _epochs(self, tensors, n_epochs, lr, bs=256, val=None):
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=CFG["wd"])
        n = len(tensors[0]); best = (np.inf, 0, None)
        for ep in range(n_epochs):
            self.model.train()
            perm = torch.randperm(n, generator=self.g)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                loss_fn(self.model, *[t[idx] for t in tensors]).backward()
                opt.step()
            if val is not None:
                self.model.eval()
                with torch.no_grad():
                    v = float(loss_fn(self.model, *val))
                if v < best[0]:
                    best = (v, ep + 1, {k: t.clone() for k, t in self.model.state_dict().items()})
        return best

    def fit(self, d, max_epochs=150):
        torch.manual_seed(self.seed); self.g = torch.Generator().manual_seed(self.seed)
        X = d[self.feats].to_numpy(float)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        # 1) choose epochs on the last 15% of the training period (chronological)
        cut = int(len(d) * 0.85)
        self.model = Net(len(self.feats), CFG["hidden"], CFG["drop"])
        _, best_ep, _ = self._epochs(self._tensors(d.iloc[:cut]), max_epochs, CFG["lr"],
                                     val=self._tensors(d.iloc[cut:]))
        # 2) refit on everything for that many epochs
        torch.manual_seed(self.seed)
        self.model = Net(len(self.feats), CFG["hidden"], CFG["drop"])
        self._epochs(self._tensors(d), best_ep, CFG["lr"])
        self.best_ep = best_ep
        return self

    def update(self, new, hist):
        """Learn from the newly revealed round: a few small steps on the new matches mixed with a
        replay sample of older matches (so the net adapts without forgetting / over-training)."""
        if CFG["upd_steps"] == 0:
            return
        rng = np.random.default_rng(self.seed + len(hist))
        rep = hist.iloc[rng.choice(len(hist), min(len(hist), CFG["upd_replay"] * len(new)), replace=False)]
        batch = pd.concat([new, rep])
        self._epochs(self._tensors(batch), CFG["upd_steps"], CFG["upd_lr"], bs=len(batch))

    def predict(self, d):
        self.model.eval()
        x = torch.tensor(((d[self.feats].to_numpy(float) - self.mu) / self.sd), dtype=torch.float32)
        with torch.no_grad():
            logits, log_mu = self.model(x, *self._off(d))
        return torch.softmax(logits, 1).numpy(), torch.exp(log_mu).numpy(), torch.exp(self.model.log_r).detach().numpy()


# =============================================================== markets from distributions
def pmf(mu, r, kmax):
    k = np.arange(kmax + 1)
    if r > 5e3:
        p = poisson.pmf(k[None, :], mu[:, None])
    else:
        p = nbinom.pmf(k[None, :], r, (r / (r + mu))[:, None])
    p[:, -1] += 1 - p.sum(1)  # tail mass into last bin
    return p


def conv_total(pa, pb):
    out = np.zeros((len(pa), pa.shape[1] + pb.shape[1] - 1))
    for i in range(len(pa)):
        out[i] = np.convolve(pa[i], pb[i])
    return out


def derive(p1x2, mu, r):
    """All market probabilities from the network outputs (mu,r per COUNT_TARGETS)."""
    idx = {t: i for i, t in enumerate(COUNT_TARGETS)}
    g = lambda t, k: pmf(mu[:, idx[t]], r[idx[t]], k)
    gh, ga = g("FTHG", 12), g("FTAG", 12)
    score = gh[:, :, None] * ga[:, None, :]
    tot = conv_total(gh, ga)
    hth, hta = g("HTHG", 8), g("HTAG", 8)
    htscore = hth[:, :, None] * hta[:, None, :]
    ii, jj = np.meshgrid(np.arange(9), np.arange(9), indexing="ij")
    corners = conv_total(g("HC", 25), g("AC", 25))
    cards = conv_total(g("HY", 12), g("AY", 12))
    out = {"1x2": p1x2,
           "dc_1x": p1x2[:, 0] + p1x2[:, 1], "dc_x2": p1x2[:, 1] + p1x2[:, 2], "dc_12": p1x2[:, 0] + p1x2[:, 2],
           "btts": 1 - gh[:, 0] - ga[:, 0] + gh[:, 0] * ga[:, 0],
           "tg_dist": np.c_[tot[:, :6], tot[:, 6:].sum(1)],  # 0..5, 6+
           "ht_1x2": np.c_[(htscore * (ii > jj)).sum((1, 2)), (htscore * (ii == jj)).sum((1, 2)),
                           (htscore * (ii < jj)).sum((1, 2))],
           "exp_goals": mu[:, idx["FTHG"]] + mu[:, idx["FTAG"]],
           "exp_corners": mu[:, idx["HC"]] + mu[:, idx["AC"]],
           "exp_cards": mu[:, idx["HY"]] + mu[:, idx["AY"]]}
    for L in (1.5, 2.5, 3.5):
        out[f"over{L}"] = tot[:, int(L) + 1:].sum(1)
    for L in (0.5, 1.5):
        htt = conv_total(hth, hta)
        out[f"ht_over{L}"] = htt[:, int(L) + 1:].sum(1)
    for L in (8.5, 9.5, 10.5, 11.5):
        out[f"corners_over{L}"] = corners[:, int(L) + 1:].sum(1)
    for L in (2.5, 3.5, 4.5):
        out[f"cards_over{L}"] = cards[:, int(L) + 1:].sum(1)
    out["_score"] = score  # kept for 1X2-from-goals diagnostics
    return out


def naive_rates(d):
    """Rates without learning: mean of own 'for' and opponent 'against' (long EWMA), COUNT_TARGETS order."""
    lam = lambda nm: np.c_[(d[f"{nm}F_l_h"] + d[f"{nm}A_l_a"]) / 2, (d[f"{nm}F_l_a"] + d[f"{nm}A_l_h"]) / 2]
    return np.c_[lam("g"), lam("htg"), lam("c"), lam("y")].astype(float)


def naive(d):
    """Baseline without learning: independent Poisson with naive_rates."""
    mu = naive_rates(d)
    gh, ga = poisson.pmf(np.arange(13)[None], mu[:, [0]]), poisson.pmf(np.arange(13)[None], mu[:, [1]])
    s = gh[:, :, None] * ga[:, None, :]
    ii, jj = np.meshgrid(np.arange(13), np.arange(13), indexing="ij")
    p1x2 = np.c_[(s * (ii > jj)).sum((1, 2)), (s * (ii == jj)).sum((1, 2)), (s * (ii < jj)).sum((1, 2))]
    p1x2 /= p1x2.sum(1, keepdims=True)
    return derive(p1x2, mu, np.full(len(COUNT_TARGETS), 1e9))


# =============================================================== metrics
def ll_multi(p, y):
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))))


def ll_bin(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier_bin(p, y):
    return float(np.mean((p - y) ** 2))


def outcomes(d):
    tg = (d.FTHG + d.FTAG).to_numpy()
    tc, ty = (d.HC + d.AC).to_numpy(), (d.HY + d.AY).to_numpy()
    o = {"1x2": d.FTR.map({"H": 0, "D": 1, "A": 2}).to_numpy(),
         "ht_1x2": d.HTR.map({"H": 0, "D": 1, "A": 2}).to_numpy(),
         "btts": ((d.FTHG > 0) & (d.FTAG > 0)).to_numpy(float),
         "tg_dist": np.minimum(tg, 6).astype(int)}
    for L in (1.5, 2.5, 3.5):
        o[f"over{L}"] = (tg > L).astype(float)
    for L in (0.5, 1.5):
        o[f"ht_over{L}"] = ((d.HTHG + d.HTAG).to_numpy() > L).astype(float)
    for L in (8.5, 9.5, 10.5, 11.5):
        o[f"corners_over{L}"] = (tc > L).astype(float)
    for L in (2.5, 3.5, 4.5):
        o[f"cards_over{L}"] = (ty > L).astype(float)
    o["dc_1x"] = (o["1x2"] != 2).astype(float)
    o["dc_x2"] = (o["1x2"] != 0).astype(float)
    o["dc_12"] = (o["1x2"] != 1).astype(float)
    o["tot_goals"], o["tot_corners"], o["tot_cards"] = tg, tc, ty
    return o


# =============================================================== walk-forward
def rounds(dates):
    """Round id: Fri-Mon block or Tue-Thu block of the same week."""
    wk = dates.dt.to_period("W-THU").astype(str)
    mid = dates.dt.dayofweek.isin([1, 2, 3])  # Tue, Wed, Thu
    return wk + np.where(mid, "-mid", "-wkd")


def main(quick=False, seasons=None, out=None):
    global OUT
    seasons = seasons or TEST_SEASONS
    OUT = Path(out) if out else OUT
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    df = fm.load()
    df = df[df.FTR.isin(["H", "D", "A"])].reset_index(drop=True)
    df = add_market(build_features(df))
    df = df[df.season > fm.SEASONS[0]].dropna(subset=["mo_h"]).reset_index(drop=True)  # 1718 = warm-up
    df["round"] = rounds(df.Date)
    feat_cols = [c for c in df.columns if c.startswith(tuple(["elo_", "rest_", "n_", "is_champ", "lg_", "ref_",
                                                              "hv_", "av_"])) or "_s_" in c or "_l_" in c]
    variants = {"NN-A": (feat_cols, False),
                "NN-B": (feat_cols + ["mo_h", "mo_d", "mo_a", "ou_open_f", "ou_open_missing"], True)}
    seeds = [0] if quick else [0, 1, 2]
    df = df[df.season <= seasons[-1]]
    print(f"{len(df)} matches, {len(feat_cols)} stat features, seeds={seeds}, seasons={seasons}, CFG={CFG}")

    preds = {v: [] for v in variants}
    log = []
    for s in seasons:
        season_rounds = sorted(df.loc[df.season == s, "round"].unique(),
                               key=lambda rr: df.loc[df["round"] == rr, "Date"].min())
        if not season_rounds:
            continue
        hist = df[df.Date < df.loc[df.season == s, "Date"].min()]
        members = {v: [Member(f, sd, off).fit(hist) for sd in seeds] for v, (f, off) in variants.items()}
        print(f"[{time.time()-t0:5.0f}s] season {s}: refit on {len(hist)} matches "
              f"(epochs chosen: {[m.best_ep for m in members['NN-A']]}/{[m.best_ep for m in members['NN-B']]}), "
              f"{len(season_rounds)} rounds")
        for rr in season_rounds:
            te = df[df["round"] == rr]
            for v, ms in members.items():
                outs = [m.predict(te) for m in ms]
                p3 = np.mean([o[0] for o in outs], 0)
                mu = np.mean([o[1] for o in outs], 0)
                r_ = np.mean([o[2] for o in outs], 0)
                preds[v].append((te.index.to_numpy(), p3, mu, r_))
            # round report: 1X2 log loss model vs market closing
            y = te.FTR.map({"H": 0, "D": 1, "A": 2}).to_numpy()
            pa, pb = preds["NN-A"][-1][1], preds["NN-B"][-1][1]
            mc = te[["mc_h", "mc_d", "mc_a"]].to_numpy()
            okc = ~np.isnan(mc).any(1)
            log.append({"round": rr, "season": s, "n": len(te), "ll_A": ll_multi(pa, y), "ll_B": ll_multi(pb, y),
                        "ll_close": ll_multi(mc[okc], y[okc]) if okc.any() else np.nan,
                        "hit_A": float(np.mean(pa.argmax(1) == y)), "hit_B": float(np.mean(pb.argmax(1) == y))})
            # learn from the revealed round
            seen = df[df.Date < te.Date.min()]
            for ms in members.values():
                for m in ms:
                    m.update(te, seen)
        lg = pd.DataFrame(log); lg = lg[lg.season == s]
        print(f"         done: 1X2 logloss A {np.average(lg.ll_A, weights=lg.n):.4f}  "
              f"B {np.average(lg.ll_B, weights=lg.n):.4f}  close {np.average(lg.ll_close.dropna(), weights=lg.n[lg.ll_close.notna()]) if lg.ll_close.notna().any() else float('nan'):.4f}"
              f" (close only on rounds with closing odds)")

    pd.DataFrame(log).to_csv(OUT / "round_log.csv", index=False)
    evaluate(df, preds)
    print(f"total time {time.time()-t0:.0f}s")


def evaluate(df, preds):
    idx = np.concatenate([p[0] for p in preds["NN-A"]])
    d = df.loc[idx].reset_index(drop=True)
    o = outcomes(d)
    models = {}
    for v in preds:
        p3 = np.concatenate([p[1] for p in preds[v]]); mu = np.concatenate([p[2] for p in preds[v]])
        # r differs per round; derive per block to keep it exact
        parts = [derive(p[1], p[2], p[3]) for p in preds[v]]
        models[v] = {k: np.concatenate([q[k] for q in parts]) for k in parts[0] if k != "_score"}
    models["naive"] = {k: v for k, v in naive(d).items() if k != "_score"}

    mkt = {"1x2_open": d[["mo_h", "mo_d", "mo_a"]].to_numpy(), "1x2_close": d[["mc_h", "mc_d", "mc_a"]].to_numpy(),
           "over2.5_open": d.ou_open.to_numpy(), "over2.5_close": d.ou_close.to_numpy()}

    rows = []
    def add(market, name, p, y, kind):
        ok = ~np.isnan(p).any(1) if p.ndim == 2 else ~np.isnan(p)
        ok &= ~pd.isna(y)
        if not ok.any():
            return
        yy = y[ok].astype(int) if kind == "multi" else y[ok]
        ll = ll_multi(p[ok], yy) if kind == "multi" else ll_bin(p[ok], yy)
        br = (np.mean(np.sum((p[ok] - np.eye(p.shape[1])[yy]) ** 2, 1)) if kind == "multi" else brier_bin(p[ok], yy))
        rows.append({"market": market, "model": name, "n": int(ok.sum()), "logloss": ll, "brier": br})

    multi = {"1x2", "ht_1x2", "tg_dist"}
    for k in o:
        if k.startswith("tot_"):
            continue
        for name, m in models.items():
            add(k, name, m[k], o[k], "multi" if k in multi else "bin")
    # market benchmarks on the matches where they exist
    for k, p in mkt.items():
        add(k.split("_")[0], "market " + k.split("_")[1], p, o[k.split("_")[0]], "multi" if k.startswith("1x2") else "bin")
    res = pd.DataFrame(rows)
    # same-sample comparison for markets with odds: restrict models to rows where closing exists
    for k, ycol in (("1x2", "1x2_close"), ("over2.5", "over2.5_close")):
        ok = ~np.isnan(mkt[ycol]).any(1) if mkt[ycol].ndim == 2 else ~np.isnan(mkt[ycol])
        for name in ["NN-A", "NN-B", "naive"]:
            p = models[name][k]
            pp = p[ok]; yy = o[k][ok]
            ll = ll_multi(pp, yy.astype(int)) if k == "1x2" else ll_bin(pp, yy)
            res.loc[(res.market == k) & (res.model == name), "logloss_same_rows_as_close"] = ll
        for nm in ("open", "close"):
            p = mkt[f"{k}_{nm}"][ok]
            ll = ll_multi(p, o[k][ok].astype(int)) if k == "1x2" else ll_bin(p, o[k][ok])
            res.loc[(res.market == k) & (res.model == f"market {nm}"), "logloss_same_rows_as_close"] = ll
    res.to_csv(OUT / "market_metrics.csv", index=False)

    # expected-count accuracy
    cnt = []
    for nm, key in (("goals", "tot_goals"), ("corners", "tot_corners"), ("cards", "tot_cards")):
        for name, m in models.items():
            e = m[f"exp_{nm}"]
            cnt.append({"stat": nm, "model": name, "MAE": float(np.mean(np.abs(e - o[key]))),
                        "mean_pred": float(e.mean()), "mean_actual": float(o[key].mean()),
                        "corr": float(np.corrcoef(e, o[key])[0, 1])})
    cnt = pd.DataFrame(cnt); cnt.to_csv(OUT / "count_metrics.csv", index=False)

    # calibration (NN-B) for a few binary markets
    cal = []
    for k in ["over2.5", "btts", "corners_over9.5", "cards_over3.5", "dc_1x"]:
        b = pd.cut(models["NN-B"][k], [0, .2, .3, .4, .5, .6, .7, .8, 1])
        t = pd.DataFrame({"p": models["NN-B"][k], "y": o[k], "b": b}).groupby("b", observed=True).agg(
            n=("y", "size"), pred=("p", "mean"), actual=("y", "mean")).reset_index()
        t.insert(0, "market", k); cal.append(t)
    cal = pd.concat(cal); cal["b"] = cal.b.astype(str); cal.to_csv(OUT / "calibration_NN-B.csv", index=False)

    # value check vs opening Avg odds (1X2 and over 2.5), flat stakes, edge > 5%
    val = []
    won3 = np.eye(3)[o["1x2"]] == 1
    odds3 = d[["odd_h", "odd_d", "odd_a"]].to_numpy()
    for name in ("NN-A", "NN-B"):
        e = models[name]["1x2"] * odds3 - 1
        b = e > 0.05
        pr = np.where(won3, odds3 - 1, -1)[b & ~np.isnan(odds3)]
        val.append({"market": "1x2", "model": name, "bets": len(pr), "roi": pr.mean() if len(pr) else np.nan})
        po, oo = models[name]["over2.5"], d.odd_over.to_numpy()
        b = (po * oo - 1 > 0.05) & ~np.isnan(oo)
        pr = np.where(o["over2.5"][b] == 1, oo[b] - 1, -1)
        val.append({"market": "over2.5", "model": name, "bets": len(pr), "roi": pr.mean() if len(pr) else np.nan})
    pd.DataFrame(val).to_csv(OUT / "value_check.csv", index=False)

    # per-match predictions for inspection ("where were we right / wrong")
    pm = d[["Date", "league", "season", "round", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HC", "AC", "HY", "AY"]].copy()
    for name in ("NN-B", "NN-A"):
        m = models[name]
        pm[f"{name}_pH"], pm[f"{name}_pD"], pm[f"{name}_pA"] = m["1x2"].T
        for k in ("over2.5", "btts", "corners_over9.5", "cards_over3.5", "dc_1x"):
            pm[f"{name}_{k}"] = m[k]
        pm[f"{name}_xG"], pm[f"{name}_xCorners"], pm[f"{name}_xCards"] = m["exp_goals"], m["exp_corners"], m["exp_cards"]
    pm["mkt_close_pH"], pm["mkt_close_pD"], pm["mkt_close_pA"] = mkt["1x2_close"].T
    pm["mkt_close_over2.5"] = mkt["over2.5_close"]
    pm.to_csv(OUT / "predictions_all.csv", index=False, float_format="%.4f")

    pd.set_option("display.width", 200)
    print("\n=== market metrics (lower = better) ===")
    print(res.sort_values(["market", "logloss"]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== expected counts ===")
    print(cnt.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n=== value check (opening Avg odds, edge>5%) ===")
    print(pd.DataFrame(val).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--valid", action="store_true", help="run on validation seasons (hyper-parameter choice)")
    ap.add_argument("--cfg", default="", help='overrides, e.g. "drop=0.5,upd_steps=0"')
    a = ap.parse_args()
    for kv in filter(None, a.cfg.split(",")):
        k, v = kv.split("=")
        CFG[k] = tuple(int(x) for x in v.split("x")) if k == "hidden" else type(CFG[k])(float(v))
    main(a.quick, VALID_SEASONS if a.valid else None, ("nn_valid/" + (a.cfg or "base")) if a.valid else None)
