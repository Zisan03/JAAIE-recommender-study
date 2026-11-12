"""Leakage check. In the main analysis every model (and the recent-popularity counts) is built from the EVALUATED
students, who were selected partly by their activity AFTER the cutoff (>=3 new materials). Here the contributor
cohort is defined from pre-cutoff information only: all students with >=10 opened materials before the cutoff.
Evaluation (students, targets) is unchanged, so results stay paired with the main analysis."""
import numpy as np, pandas as pd
from scipy import stats
import pipeline as P
from recent_popularity import recent_scores

def cohort_matrix(c):
    d, cut, items = c["d"], c["cut"], c["items"]
    tr = d[(d.date < cut) & d.id_site.isin(items)][["id_student", "id_site"]].drop_duplicates()
    n = tr.groupby("id_student").size()
    users = sorted(n[n >= P.MIN_TRAIN].index)
    ui = {u: i for i, u in enumerate(users)}; ii = {s: i for i, s in enumerate(items)}
    X = np.zeros((len(users), len(items)), dtype=np.float32)
    t = tr[tr.id_student.isin(users)]
    for u, s in t.itertuples(index=False): X[ui[u], ii[s]] = 1
    return users, X

def recent_cohort(c, users, window=28):
    d, cut, items = c["d"], c["cut"], c["items"]
    r = d[(d.date >= cut - window) & (d.date < cut) & d.id_student.isin(users) & d.id_site.isin(items)][["id_student", "id_site"]].drop_duplicates()
    rc = r.groupby("id_site").size().reindex(items).fillna(0).values.astype(float)
    return rc

rows, summ = [], {}
for mod, pres in P.COURSES:
    c = P.build(mod, pres, 70); X, T = c["X"], c["T"]
    lam = P.tune_lambda(mod, pres, c["cut"])
    U = list(c["meta"].index)
    UA, XA = cohort_matrix(c)
    pos = {u: i for i, u in enumerate(UA)}
    rowsA = np.array([pos[u] for u in U])
    assert np.array_equal(XA[rowsA], X), "evaluated rows must match"
    totA = XA.sum(0)
    def tile(v): return np.tile(v, (len(X), 1))
    recA = recent_cohort(c, UA); recA_s = recA + 0.5 * totA / (totA.max() + 1.0)
    # any pre-cutoff clicker on warm items (even looser cohort)
    allu = c["d"][(c["d"].date < c["cut"])].id_student.unique()
    prB = c["d"][(c["d"].date < c["cut"]) & c["d"].id_student.isin(allu) & c["d"].id_site.isin(c["items"])][["id_student", "id_site"]].drop_duplicates()
    totB = prB.groupby("id_site").size().reindex(c["items"]).fillna(0).values.astype(float)   # overall popularity among the SAME contributors
    recB = recent_cohort(c, allu); recB_s = recB + 0.5 * totB / (totB.max() + 1.0)
    cbf = P.cbf_scores(X, c["F_week"])
    cfA = P.knn_scores(XA)[rowsA]; ezA = P.ease_scores(XA, lam)[rowsA]
    S = {"popularity_main": tile(X.sum(0).astype(float)), "popularity_cohortA": tile(totA.astype(float)),
         "recent_main": recent_scores(c, 28), "recent_cohortA": tile(recA_s), "recent_anyclicker": tile(recB_s),
         "cf_main": P.knn_scores(X), "cf_cohortA": cfA, "ease_main": P.ease_scores(X, lam), "ease_cohortA": ezA,
         "hybrid_cohortA": 0.5 * P.mm(cbf, X) + 0.5 * P.mm(cfA, X)}
    R = {k: P.evaluate(v, X, T) for k, v in S.items()}
    for k, v in R.items():
        rows.append(dict(course=f"{mod}-{pres}", model=k, ndcg10=v["ndcg10"].mean(), hit10=v["hit10"].mean(), n_eval=len(X), n_cohortA=len(UA), n_anyclicker=len(allu)))
    for a, b in (("recent_cohortA", "cf_cohortA"), ("recent_cohortA", "ease_cohortA"), ("recent_cohortA", "hybrid_cohortA"), ("recent_cohortA", "popularity_cohortA"), ("recent_anyclicker", "cf_cohortA")):
        dd = R[a]["ndcg10"] - R[b]["ndcg10"]
        summ[(f"{mod}-{pres}", f"{a} minus {b}")] = (dd.mean(), P.rank_biserial(dd), stats.wilcoxon(R[a]["ndcg10"], R[b]["ndcg10"]).pvalue)
df = pd.DataFrame(rows); df.to_csv(P.OUT + "/recent_cohort_check.csv", index=False)
pd.set_option("display.width", 200)
print(df.pivot(index="model", columns="course", values="ndcg10").assign(mean=lambda x: x.mean(1)).round(3))
print(df.groupby("course")[["n_eval", "n_cohortA", "n_anyclicker"]].first())
s = pd.DataFrame([(k[0], k[1], *v) for k, v in summ.items()], columns=["course", "contrast", "mean_diff", "r_rb", "p"])
s["p_holm"] = P.holm(s.p.values); s.to_csv(P.OUT + "/recent_cohort_check_tests.csv", index=False)
print(s.round(4).to_string(index=False))
