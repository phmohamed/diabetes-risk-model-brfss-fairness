import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from models import Logit, GBM, auc

rng = np.random.default_rng(0)
d = pd.read_csv("data/raw.csv").drop(columns="ID")
y = d.pop("Diabetes_binary").values.astype(float)
feats = list(d.columns)
X = d.values.astype(float)
idx = rng.permutation(len(d)); cut = int(.7 * len(d))
tr, te = idx[:cut], idx[cut:]
print("rows", len(d), "prevalence", y.mean().round(4), "train", len(tr), "test", len(te))

def run(cols, name):
    ci = [feats.index(c) for c in cols]
    lg = Logit().fit(X[tr][:, ci], y[tr]); pl = lg.predict(X[te][:, ci])
    gb = GBM().fit(X[tr][:, ci], y[tr]); pg = gb.predict(X[te][:, ci])
    print(name, "logit AUC", round(auc(y[te], pl), 4), "GBM AUC", round(auc(y[te], pg), 4))
    return lg, gb, pl, pg

lg, gb, pl, pg = run(feats, "all features")
noself = [c for c in feats if c not in ("GenHlth",)]
_, _, pl2, pg2 = run(noself, "without self-rated health")
simple = ["HighBP", "HighChol", "BMI", "Age", "GenHlth", "PhysActivity", "Smoker", "Sex"]
_, _, pl3, pg3 = run(simple, "8 simple features")
yt = y[te]

# bootstrap CI for GBM AUC
bs = []
for _ in range(200):
    i = rng.integers(0, len(yt), len(yt)); bs.append(auc(yt[i], pg[i]))
print("GBM AUC 95% CI", np.percentile(bs, [2.5, 97.5]).round(4))

# calibration
def calib(p, q=10):
    b = pd.qcut(p, q, duplicates="drop")
    return pd.DataFrame({"p": p, "y": yt}).groupby(b, observed=True).mean()
cal = calib(pg)
cal.to_csv("data/calibration_gbm.csv")

# threshold: flag top 20% of risk (a fixed screening budget)
thr = np.quantile(pg, 0.8)
flag = pg >= thr
print("flag top 20%: sensitivity", (flag & (yt == 1)).sum() / yt.sum(), "PPV", yt[flag].mean())

# subgroup audit at single threshold
te_df = d.iloc[te].copy()
groups = {
    "Sex": te_df.Sex.map({0: "Female", 1: "Male"}),
    "Age": pd.cut(te_df.Age, [0, 4, 8, 10, 13], labels=["18-39", "40-59", "60-69", "70+"]),
    "Income": pd.cut(te_df.Income, [0, 3, 6, 8], labels=["<$20k", "$20-50k", ">$50k"]),
    "Education": pd.cut(te_df.Education, [0, 4, 5, 6], labels=["High school or less", "Some college", "College grad"]),
}
rows = []
for gname, gs in groups.items():
    for lv in gs.cat.categories if hasattr(gs, "cat") else sorted(gs.unique()):
        m = (gs == lv).values
        if m.sum() < 200 or yt[m].sum() < 20: continue
        rows.append(dict(attribute=gname, group=lv, n=int(m.sum()), prevalence=yt[m].mean(),
                         auc=auc(yt[m], pg[m]), flagged=flag[m].mean(),
                         sensitivity=(flag[m] & (yt[m] == 1)).sum() / yt[m].sum(),
                         false_pos_rate=(flag[m] & (yt[m] == 0)).sum() / (yt[m] == 0).sum(),
                         ppv=yt[m][flag[m]].mean(),
                         mean_pred=pg[m].mean()))
sg = pd.DataFrame(rows).round(3)
sg.to_csv("data/subgroup_audit.csv", index=False)
print(sg.to_string())

# group-specific thresholds to equalise sensitivity (shows the tradeoff)
tgt = (flag & (yt == 1)).sum() / yt.sum()
eq = []
for gname, gs in groups.items():
    for lv in gs.cat.categories if hasattr(gs, "cat") else sorted(gs.unique()):
        m = (gs == lv).values
        if m.sum() < 200 or yt[m].sum() < 20: continue
        s = np.sort(pg[m][yt[m] == 1]); t = s[int((1 - tgt) * len(s))]
        f = pg[m] >= t
        eq.append(dict(attribute=gname, group=lv, threshold=round(t, 3), flagged=round(f.mean(), 3),
                       false_pos_rate=round((f & (yt[m] == 0)).sum() / (yt[m] == 0).sum(), 3)))
pd.DataFrame(eq).to_csv("data/equal_sensitivity_thresholds.csv", index=False)
print(pd.DataFrame(eq).to_string())

res = pd.DataFrame([
    ("Logistic, all 21 features", auc(yt, pl)), ("GBM, all 21 features", auc(yt, pg)),
    ("Logistic, no self-rated health", auc(yt, pl2)), ("GBM, no self-rated health", auc(yt, pg2)),
    ("Logistic, 8 features", auc(yt, pl3)), ("GBM, 8 features", auc(yt, pg3))], columns=["model", "test_auc"]).round(4)
res.to_csv("data/model_auc.csv", index=False); print(res)

# logistic odds ratios (per 1 SD)
od = pd.DataFrame({"feature": feats, "odds_ratio_per_SD": np.exp(lg.w[1:])}).sort_values("odds_ratio_per_SD", ascending=False).round(2)
od.to_csv("data/logit_odds_ratios.csv", index=False); print(od.to_string())

# figures
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot([0, .6], [0, .6], "k--", lw=1); ax[0].plot(cal.p, cal.y, "o-", color="#b2182b")
ax[0].set_xlabel("Mean predicted risk (decile)"); ax[0].set_ylabel("Observed diabetes rate"); ax[0].set_title("GBM calibration, test set")
s2 = sg[sg.attribute.isin(["Age", "Income", "Sex", "Education"])].copy()
lab = s2.attribute + ": " + s2.group.astype(str)
yp = np.arange(len(s2))[::-1]
ax[1].barh(yp - .18, s2.sensitivity, .36, label="Sensitivity", color="#2166ac")
ax[1].barh(yp + .18, s2.false_pos_rate, .36, label="False positive rate", color="#d6604d")
ax[1].set_yticks(yp); ax[1].set_yticklabels(lab, fontsize=8); ax[1].legend(fontsize=8)
ax[1].set_title("Same threshold (top 20% risk), by subgroup")
plt.tight_layout(); plt.savefig("figures/calibration_and_fairness.png", dpi=110)
