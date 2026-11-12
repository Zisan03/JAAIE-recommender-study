"""Exploratory: does making CF time-aware close the gap to Recent Popularity?
Item-based CF as in the main analysis, but each student-material interaction is weighted by exp(-age/tau),
age = days between the student's LAST opening of the material and the cutoff (pre-cutoff data only).
tau = 28 days is the primary setting (matching the recent-use window, fixed before running); tau = 14 and 56 are sensitivity checks."""
import numpy as np, pandas as pd
from scipy import stats
import pipeline as P
from recent_popularity import recent_scores

TAUS = (14, 28, 56); PRIMARY = 28
def weights(c, tau):
    d, cut, items, U = c["d"], c["cut"], c["items"], list(c["meta"].index)
    tr = d[(d.date < cut) & d.id_student.isin(U) & d.id_site.isin(items)]
    last = tr.groupby(["id_student", "id_site"]).date.max().reset_index()
    ui = {u: i for i, u in enumerate(U)}; ii = {s: i for i, s in enumerate(items)}
    W = np.zeros((len(U), len(items)), dtype=np.float32)
    for u, s, dt in last.itertuples(index=False):
        W[ui[u], ii[s]] = np.exp(-(cut - dt) / tau)
    return W

rows, tests = [], []
for mod, pres in P.COURSES:
    c = P.build(mod, pres, 70); X, T = c["X"], c["T"]
    lam = P.tune_lambda(mod, pres, c["cut"])
    R = {"cf": P.evaluate(P.knn_scores(X), X, T), "ease": P.evaluate(P.ease_scores(X, lam), X, T),
         "recent28": P.evaluate(recent_scores(c, 28), X, T)}
    for tau in TAUS:
        W = weights(c, tau)
        assert np.array_equal(W > 0, X > 0)
        R[f"tcf{tau}"] = P.evaluate(P.knn_scores(W), X, T)
    for k, v in R.items():
        rows.append(dict(course=f"{mod}-{pres}", model=k, ndcg10=v["ndcg10"].mean(), hit10=v["hit10"].mean(), recall10=v["recall10"].mean()))
    for a, b in ((f"tcf{PRIMARY}", "cf"), (f"tcf{PRIMARY}", "ease"), (f"tcf{PRIMARY}", "recent28")):
        dd = R[a]["ndcg10"] - R[b]["ndcg10"]
        tests.append(dict(course=f"{mod}-{pres}", contrast=f"{a} minus {b}", mean_diff=dd.mean(), r_rb=P.rank_biserial(dd),
                          p=stats.wilcoxon(R[a]["ndcg10"], R[b]["ndcg10"]).pvalue))
df = pd.DataFrame(rows); df.to_csv(P.OUT + "/time_aware_cf.csv", index=False)
t = pd.DataFrame(tests); t["p_holm"] = P.holm(t.p.values); t.to_csv(P.OUT + "/time_aware_cf_tests.csv", index=False)
pd.set_option("display.width", 200)
pv = df.pivot(index="model", columns="course", values="ndcg10"); pv["mean"] = pv.mean(axis=1)
print(pv.loc[["cf", "tcf14", "tcf28", "tcf56", "ease", "recent28"]].round(3))
print(df.pivot(index="model", columns="course", values="hit10").assign(mean=lambda x: x.mean(1)).loc[["cf", "tcf28", "recent28"]].round(3))
print(t.round(4).to_string(index=False))
