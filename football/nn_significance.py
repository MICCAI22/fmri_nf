"""Bootstrap check: is NN-B really better than the naive baseline / the market? (run after nn_model.py)"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import football_model_v1 as fm, nn_model as nm

p = pd.read_csv("nn_results/predictions_all.csv", parse_dates=["Date"])
df = fm.load(); df = df[df.FTR.isin(["H", "D", "A"])].reset_index(drop=True)
df = nm.add_market(nm.build_features(df))
df = df[df.season > fm.SEASONS[0]].dropna(subset=["mo_h"])
d = p.merge(df, on=["Date", "HomeTeam", "AwayTeam"], suffixes=("", "_x"))
nv = nm.naive(d); o = nm.outcomes(d)
rng = np.random.default_rng(0)
def bll(q, y): q = np.clip(q, 1e-12, 1 - 1e-12); return -(y * np.log(q) + (1 - y) * np.log(1 - q))
rows = []
for k, ref_name, ref in [("over2.5", "market close", d.ou_close.to_numpy()), ("over2.5", "naive", nv["over2.5"]),
                         ("btts", "naive", nv["btts"]), ("corners_over9.5", "naive", nv["corners_over9.5"]),
                         ("cards_over3.5", "naive", nv["cards_over3.5"])]:
    ok = ~np.isnan(ref)
    diff = bll(d[f"NN-B_{k}"].to_numpy(), o[k])[ok] - bll(ref, o[k])[ok]
    bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
    rows.append({"market": k, "NN-B minus": ref_name, "diff": diff.mean(), "ci_lo": np.percentile(bs, 2.5), "ci_hi": np.percentile(bs, 97.5)})
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
print("(negative = NN-B better; CI containing 0 = no real difference)")
