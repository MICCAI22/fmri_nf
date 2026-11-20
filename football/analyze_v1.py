import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import numpy as np, pandas as pd
import football_model_v1 as fm
df = fm.load()
print("rows per league/season:\n", df.groupby(["season","league"]).size().unstack())
print("date range 2627:", df[df.season=="2627"].Date.min(), df[df.season=="2627"].Date.max())
df = fm.build_features(df)
for i,c in enumerate(["mo_h","mo_d","mo_a"]): df[c]=fm.odds_from(df,fm.OPEN_SETS,True)[:,i]
cl=fm.odds_from(df,fm.CLOSE_SETS,True); bo=fm.odds_from(df,fm.BET_SETS,False)
df=df.assign(c_h=cl[:,0],c_d=cl[:,1],c_a=cl[:,2],b_h=bo[:,0],b_d=bo[:,1],b_a=bo[:,2])
print("missing PSCH per season:", df.groupby("season").PSCH.apply(lambda s: s.isna().mean()).round(2).to_dict() if "PSCH" in df else "no col")
df=df.dropna(subset=["mo_h","c_h","b_h"]).reset_index(drop=True)
y=df.FTR.map({"H":0,"D":1,"A":2}).to_numpy()
fa=["elo_diff","form_h","form_a","gf_h","ga_h","gf_a","ga_a","home_form_h","away_form_a","h2h_h","is_champ"]
pa=fm.walk_forward(df,fa,pd.Series(y)); pb=fm.walk_forward(df,fa+["mo_h","mo_d","mo_a"],pd.Series(y))
m=~np.isnan(pa[:,0]); d=df[m].copy(); yt=y[m]; pa,pb=pa[m],pb[m]
mc=d[["c_h","c_d","c_a"]].to_numpy(); od=d[["b_h","b_d","b_a"]].to_numpy()
print("\nmean probs H/D/A  actual:", np.bincount(yt)/len(yt))
for n,p in [("A",pa),("B",pb),("close",mc)]: print(n, p.mean(0).round(3), "std per class", p.std(0).round(3))
print("\nB365 overround mean:", (1/od).sum(1).mean().round(3))
print("\nper season / league logloss (A, B, close):")
for (s,l),g in d.groupby(["season","league"]):
    ix=(d.season==s)&(d.league==l); ix=ix.to_numpy()
    print(s,l,ix.sum(), *[round(fm.log_loss_(p[ix],yt[ix]),4) for p in (pa,pb,mc)])
print("\nbets by outcome (A, edge>3%):", ((pa*od-1)>0.03).sum(0), " B:", ((pb*od-1)>0.03).sum(0))
# calibration for A, per class
for k,name in enumerate("HDA"):
    bins=pd.cut(pa[:,k],[0,.15,.25,.35,.45,.55,.7,1])
    t=pd.DataFrame({"p":pa[:,k],"o":yt==k,"b":bins}).groupby("b",observed=True).agg(n=("o","size"),pred=("p","mean"),obs=("o","mean"))
    print(f"\ncalibration A [{name}]\n", t.round(3))
# CLV: does B365 price beat closing for picked bets?
clo = 1/mc  # fair closing odds
for n,p in [("A",pa),("B",pb)]:
    bets=(p*od-1)>0.03
    print(n,"CLV mean (bet odds / fair close -1):", ((od/clo-1)[bets]).mean().round(4))
print("\nMI: coef corr elo vs close home prob:", np.corrcoef(d.elo_diff, d.c_h)[0,1].round(3))
