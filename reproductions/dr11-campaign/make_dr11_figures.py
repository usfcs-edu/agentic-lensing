#!/usr/bin/env python3
"""Generate DR11 campaign report figures from the campaign artifacts."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd, numpy as np
from pathlib import Path
D=Path("/home2/benson/git/agentic-lensing/reproductions")
FIG=D/"dr11-campaign"/"papers"/"figures"; FIG.mkdir(parents=True,exist_ok=True)

# Fig 1: DESI -> HSC p_lens flip (the money plot)
c=pd.read_parquet(D/"lensjudge"/"outputs"/"dr11s_cascade_full.parquet")
t2=c[c.highres_survey.notna()].copy()
col={"A":"#1a9850","B":"#91cf60","C":"#fee08b","D":"#d73027"}
fig,ax=plt.subplots(figsize=(5,4.6))
ax.plot([0,1],[0,1],"--",color="gray",lw=1,zorder=0)
for g in ["A","B","C","D"]:
    s=t2[t2.grade_pred==g]
    if len(s): ax.scatter(s.p_lens_tier1,s.p_lens_tier2,c=col[g],s=60,edgecolor="k",lw=0.4,label=f"HSC grade {g} (n={len(s)})",zorder=3)
ax.set_xlabel("DESI tier-1 $p_{\\rm lens}$ (0.26\"/px)"); ax.set_ylabel("HSC tier-2 $p_{\\rm lens}$ (0.168\"/px)")
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_title("DR11-south: DESI$\\to$HSC resolution flip (26 escalated)")
ax.legend(fontsize=8,loc="lower right"); fig.tight_layout(); fig.savefig(FIG/"desi_hsc_flip.png",dpi=150); plt.close(fig)
print("wrote desi_hsc_flip.png (n_tier2=%d, A/B=%d)"%(len(t2),(t2.grade_pred.isin(['A','B'])).sum()))

# Fig 2: SuGOHI enrichment vs survivor v3blend8 distribution
sug=pd.read_csv(D/"dr11-campaign"/"data"/"dr11s_sugohi_recovered.csv")
surv=pd.read_parquet(D/"claudenet"/"data"/"v3"/"survivors_dr11s_recal.parquet").astype({"row_id":str})
FEATS=["effnet_B","zoobot_N","effnet_S2_hard","effnet_B3_hard","resnet46_C_hard","effnet_S2_b50","effnet_B3_b50","resnet46_C_b50"]
df=surv[["row_id"]].copy()
for m in FEATS:
    sc=pd.read_parquet(D/"claudenet"/"data"/"v3"/f"scores_member_{m}_survdr11s.parquet")[["row_id","pc"]].rename(columns={"pc":m}).astype({"row_id":str})
    df=df.merge(sc,on="row_id",how="inner")
allb=df[FEATS].mean(axis=1)
fig,ax=plt.subplots(figsize=(5,4))
b=np.linspace(0,1,41)
ax.hist(allb,bins=b,density=True,alpha=0.5,color="gray",label=f"all survivors (n={len(allb)})")
ax.hist(sug.v3blend8,bins=b,density=True,alpha=0.6,color="#1a9850",label=f"SuGOHI lenses (n={len(sug)})")
ax.axvline(allb.median(),color="gray",ls="--",lw=1); ax.axvline(sug.v3blend8.median(),color="#1a9850",ls="--",lw=1)
ax.set_xlabel("v3blend8 score"); ax.set_ylabel("density"); ax.set_title("v3blend8 enriches SuGOHI HSC lenses (2.7$\\times$ median)")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG/"sugohi_enrichment.png",dpi=150); plt.close(fig)
print("wrote sugohi_enrichment.png")
