import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import numpy as np, pandas as pd
import football_model_v1 as fm
df = fm.build_features(fm.load())
for s in ["2526","2627"]:
    g=df[df.season==s]; print(s, {c: round(g[c].notna().mean(),2) for c in ["PSH","PSCH","AvgCH","B365CH","MaxH","MaxCH"] if c in g})
for i,c in enumerate(["mo_h","mo_d","mo_a"]): df[c]=fm.odds_from(df,fm.OPEN_SETS,True)[:,i]
cl=fm.odds_from(df,fm.CLOSE_SETS,True); bo=fm.odds_from(df,fm.BET_SETS,False)
mx=fm.odds_from(df,[("MaxH","MaxD","MaxA")],False)
bc=fm.odds_from(df,[("B365CH","B365CD","B365CA")],False)
df=df.assign(c_h=cl[:,0],c_d=cl[:,1],c_a=cl[:,2],b_h=bo[:,0],b_d=bo[:,1],b_a=bo[:,2],x_h=mx[:,0],x_d=mx[:,1],x_a=mx[:,2],bc_h=bc[:,0],bc_d=bc[:,1],bc_a=bc[:,2])
df=df.dropna(subset=["mo_h","c_h","b_h"]).reset_index(drop=True)
df["lo_h"]=np.log(df.mo_h/df.mo_a); df["lo_d"]=np.log(df.mo_d/(df.mo_h+df.mo_a))
y=df.FTR.map({"H":0,"D":1,"A":2}).to_numpy()
fa=["elo_diff","form_h","form_a","gf_h","ga_h","gf_a","ga_a","home_form_h","away_form_a","h2h_h","is_champ"]
pa=fm.walk_forward(df,fa,pd.Series(y)); pb=fm.walk_forward(df,fa+["mo_h","mo_d","mo_a"],pd.Series(y))
pb2=fm.walk_forward(df,fa+["lo_h","lo_d"],pd.Series(y))
pm=fm.walk_forward(df,["lo_h","lo_d"],pd.Series(y))  # market-only recalibrated
m=~np.isnan(pa[:,0]); d=df[m]; yt=y[m]
mo=d[["mo_h","mo_d","mo_a"]].to_numpy(); mc=d[["c_h","c_d","c_a"]].to_numpy()
P={"A":pa[m],"B":pb[m],"B logodds":pb2[m],"market-only LR":pm[m],"open":mo,"close":mc}
for k,p in P.items(): print(f"{k:15s} {fm.log_loss_(p,yt):.4f}")
# bootstrap diff vs close
rng=np.random.default_rng(0)
def ll(p): return -np.log(p[np.arange(len(yt)),yt])
for k in ["A","B","B logodds"]:
    dd=ll(P[k])-ll(mc); bs=[dd[rng.integers(0,len(dd),len(dd))].mean() for _ in range(2000)]
    print(k,"- close: %.4f  95%% CI [%.4f, %.4f]"%(dd.mean(),*np.percentile(bs,[2.5,97.5])))
od=d[["b_h","b_d","b_a"]].to_numpy(); ox=d[["x_h","x_d","x_a"]].to_numpy(); oc=d[["bc_h","bc_d","bc_a"]].to_numpy()
ok=~np.isnan(ox).any(1)
for k in ["A","B logodds"]:
    for t in (0.03,0.05):
        n,r=fm.roi(P[k][ok],ox[ok],yt[ok],t); bets=(P[k]*od-1)>t; okc=bets&~np.isnan(oc)
        print(f"{k} edge>{t:.0%} @Max odds: {n} bets ROI {r:+.1%} | CLV B365 open vs B365 close: {(od[okc]/oc[okc]-1).mean():+.2%}")
print("CLV baseline all outcomes:", np.nanmean(od/oc-1).round(4))
print("Max-odds overround:", np.nanmean((1/ox).sum(1)).round(4))
won=np.eye(3)[yt]==1
print("blind bet everything @Max ROI:", np.nanmean(np.where(won,ox-1,-1)).round(4))
bets=(P["B logodds"]*od-1)>0.03
for s in sorted(d.season.unique()):
    ix=(d.season==s).to_numpy()[:,None]&bets&ok[:,None]
    print(s, ix.sum(), np.where(won,ox-1,-1)[ix].mean().round(3))
