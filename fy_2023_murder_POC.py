# -*- coding: utf-8 -*-
# POC reproducer: (A) Full-dataset 800-sample TDA+XAI figure
#                 (B) Murder-only bar & intersectional panels (White/Black/Hispanic only)
# Outputs: tda_xai_analysis.png, murder_sentences_2023.png,
#          intersectional_real_analysis_2023.png, cluster_table.tex, summary_table.tex,
#          murder_all.csv, murder_sample800.csv

import os, re, sys, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Safer threading on Windows/ARM
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib
if not hasattr(sys, "ps1"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, r2_score
from sklearn.ensemble import RandomForestRegressor

import shap
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ---------------- Config ----------------
SEED = 42
#N_SAMPLE = 800
K_CLUSTERS = 4
DBSCAN_EPS = 0.5
DBSCAN_MIN = 8
MAX_DISTRICT_LEVELS = 30

TARGET_CHOICES = ["SENTTOT", "TOTSENTN", "SENTENCE_MONTHS"]
CAT_FEATS = ["CITIZEN", "WEAPON", "DISTRICT", "MONRACE"]
NUM_FEATS = ["AGE", "CRIMHIST", "EDUCATN", "ACCGDLN"]

RACE_CODE_MAP = {1:"White", 2:"Black", 3:"Hispanic", 4:"Other", 5:"Other", 6:"Other", 7:"Other", 8:"Other", 9:"Other"}
RACES_POC = ["White", "Black", "Hispanic"]
BAR_ORDER = ["White Male","White Female","Black Male","Black Female","Hispanic Male","Hispanic Female"]
BAR_COLORS = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b"]  # fixed per-bar colors

def log(s): print(f"[POC] {s}", flush=True)
def _norm(s): return re.sub(r"[\W_]+","",str(s).strip().upper())
def _ensure_dir(p): p=Path(p); p.mkdir(parents=True, exist_ok=True); return p

def smart_read(path):
    p=Path(path); 
    if p.suffix.lower()==".parquet":
        try: return pd.read_parquet(p)
        except Exception as e:
            log(f"Parquet engine error, using duckdb: {e}")
            import duckdb
            return duckdb.query(f"SELECT * FROM parquet_scan('{str(p)}')").to_df()
    if p.suffix.lower()==".csv": return pd.read_csv(p)
    raise ValueError("Provide .parquet or .csv")

def first_existing(df, names):
    for n in names:
        if n in df.columns: return n
    raise ValueError(f"None of {names} in dataframe")

# -------- Murder detection (broad but safe) --------
_MURDER_PAT = r"(murder|homicide|manslaughter|2A1\.1|2A1\.2|2A1\.3|\b1111\b|\b1112\b|\b1113\b)"
def is_murder_rowblock(df):
    mask = pd.Series(False, index=df.index); cols=[]
    for c in df.columns:
        s=df[c]
        try:
            if pd.api.types.is_string_dtype(s) or s.dtype=="object":
                m=s.astype("string").str.contains(_MURDER_PAT, case=False, na=False, regex=True)
            else:
                m=pd.to_numeric(s, errors="coerce").astype("Int64").astype("string").str.contains(r"\b111[123]\b", na=False)
            if m.any(): cols.append(c); mask|=m
        except Exception:
            pass
    return mask, cols

# -------- Mapping helpers --------
def map_race(series):
    try:
        out = pd.to_numeric(series, errors="coerce").astype("Int64").map(RACE_CODE_MAP)
    except Exception:
        out = series.astype("string").str.title()
    # collapse to our three classes for murder charts
    out = out.replace({r"^(?!White|Black|Hispanic).*$":"Other"}, regex=True)
    return pd.Categorical(out)

def map_gender(df):
    monsex = None
    for c in df.columns:
        if _norm(c) in {"MONSEX","MON SEX"}: monsex=c; break
    if monsex:
        g = pd.to_numeric(df[monsex], errors="coerce").astype("Int64").map({0:"Male",1:"Female",2:"Other"})
        return pd.Categorical(g.fillna("Unknown"))
    # fallback
    for alt in ["GENDER","SEX","DEFSEX"]:
        if alt in df.columns:
            s=df[alt].astype("string").str.upper().str.strip()
            g=s.replace({"M":"Male","F":"Female","MALE":"Male","FEMALE":"Female"})
            return pd.Categorical(g.fillna("Unknown"))
    return pd.Categorical(["Unknown"]*len(df))

# -------- Cache --------
def cache_paths(root):
    root=_ensure_dir(root)
    return root/"murder_all.csv", root/"murder_sample800.csv", root/"murder_meta.json"

# ================== A) TDA/XAI on full-dataset sample (N=800) ==================
def tda_xai_full(df, out_dir):
    target = first_existing(df, TARGET_CHOICES)
    log(f"Target column: {target}")
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target])

    ## sample from full dataset
   # sample = df.sample(n=min(N_SAMPLE,len(df)), random_state=SEED)
   # log(f"Sampled {len(sample)} of {len(df)} rows for POC")

    # Use the full dataset
    sample = df.copy()
    log(f"Using all {len(sample)} rows for analysis")
    
    # keep model features if present
    feats = [c for c in (NUM_FEATS + CAT_FEATS) if c in sample.columns]
    work = sample[feats + [target]].copy()

    # numeric clean
    for c in NUM_FEATS:
        if c in work.columns:
            s = pd.to_numeric(work[c], errors="coerce")
            work[c] = s.fillna(s.median(skipna=True))

    # cap target 99th
    q99 = work[target].quantile(0.99); work.loc[work[target]>q99, target]=q99

    # categorical as strings (limit DISTRICT cardinality for stability)
    if "DISTRICT" in work.columns:
        top = work["DISTRICT"].value_counts(dropna=False).index[:MAX_DISTRICT_LEVELS]
        work["DISTRICT"] = np.where(work["DISTRICT"].isin(top), work["DISTRICT"], "Other")
    for c in [f for f in CAT_FEATS if f in work.columns]:
        work[c] = work[c].astype("string").fillna("Unknown").replace("<NA>","Unknown")

    # X / y
    X_num = work[[c for c in NUM_FEATS if c in work.columns]]
    X_cat = pd.get_dummies(work[[c for c in CAT_FEATS if c in work.columns]],
                           drop_first=False, dtype=np.uint8)
    X = pd.concat([X_num, X_cat], axis=1).fillna(0)
    y = work[target].astype(float)

    log(f"X shape: {X.shape} (num={len(X_num.columns)}, cat={len(X_cat.columns)})")

    # PCA lens
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=SEED)
    lens = pca.fit_transform(Xs)

    # clustering
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN).fit(lens)
    db_labels = db.labels_
    sil = np.nan
    try:
        m = db_labels != -1
        if m.sum()>=3 and len(set(db_labels[m]))>1:
            sil = float(silhouette_score(lens[m], db_labels[m]))
    except Exception:
        pass

    km = KMeans(n_clusters=K_CLUSTERS, n_init=20, random_state=SEED)
    klabels = km.fit_predict(lens)

    # model + shap
    Xtr,Xte,ytr,yte,cltr,clte = train_test_split(X, y, klabels, test_size=0.2, random_state=SEED)
    rf = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
    rf.fit(Xtr,ytr); yhat = rf.predict(Xte); r2 = float(r2_score(yte,yhat))
    log(f"RandomForest R² = {r2:.3f}")

    expl = shap.TreeExplainer(rf)
    sv = expl.shap_values(Xte, check_additivity=False)
    shap_abs = pd.Series(np.abs(sv).mean(axis=0), index=Xte.columns)

    def base(col):
        for f in CAT_FEATS:
            if col.startswith(f+"_"): return f
        return col
    agg = shap_abs.groupby(base).sum().sort_values(ascending=False)

    # cluster summary table
    rows=[]
    for k in range(K_CLUSTERS):
        mk = (clte==k)
        rows.append({
            "Cluster": k+1,
            "Size": int(mk.sum()),
            "Mean Sentence (mo)": float(yte[mk].mean()) if mk.sum() else np.nan,
            "Std Dev": float(yte[mk].std()) if mk.sum() else np.nan,
            "Dominant Feature": agg.index[0] if not agg.empty else "—"
        })
    df_clusters = pd.DataFrame(rows).sort_values("Cluster")
    Path(out_dir,"cluster_table.tex").write_text(df_clusters.to_latex(index=False), encoding="utf-8")

    # SHAP summary table (top 8)
    interp = {"CRIMHIST":"Recidivism elevates variance","WEAPON":"Weapon increases guideline ranges",
              "AGE":"Age moderates severity","CITIZEN":"Distinct patterns for non-citizens",
              "DISTRICT":"Geographic practice differences","EDUCATN":"Education correlates with outcomes",
              "ACCGDLN":"Guideline acceptance matters","MONRACE":"Race-linked disparities"}
    top = agg.head(8).index.tolist()
    df_shap = pd.DataFrame({
        "Key Feature": top,
        "Legal Interpretation": [interp.get(f,"Context-dependent") for f in top],
        "Traditional Method Limitation": [
            "Masked by average effects" if f in ("CRIMHIST","AGE")
            else "Nonlinear interactions" if f in ("WEAPON","ACCGDLN")
            else "Heterogeneous across groups" if f in ("CITIZEN","MONRACE","DISTRICT")
            else "Ignored in linear baselines" for f in top
        ],
    })
    Path(out_dir,"summary_table.tex").write_text(df_shap.to_latex(index=False), encoding="utf-8")

    # 2×2 proposal-style figure
    fig = plt.figure(figsize=(12,10), dpi=120); gs = gridspec.GridSpec(2,2, hspace=0.25, wspace=0.25)
    ax = fig.add_subplot(gs[0,0])
    sc = ax.scatter(lens[:,0], lens[:,1], c=klabels, s=16, alpha=0.75)
    ttl = "TDA Clusters in PCA Space" + (f" (DBSCAN silhouette={sil:.3f})" if not np.isnan(sil) else "")
    ax.set_title(ttl); ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")

    ax = fig.add_subplot(gs[0,1])
    ax.boxplot([y.values[klabels==k] for k in range(K_CLUSTERS)], labels=[str(k+1) for k in range(K_CLUSTERS)])
    ax.set_title("Sentence Distribution by Cluster"); ax.set_xlabel("Cluster ID"); ax.set_ylabel("Sentence Length (months)")

    ax = fig.add_subplot(gs[1,0]); agg.head(8)[::-1].plot(kind="barh", ax=ax)
    ax.set_title("Feature Importance (SHAP)"); ax.set_xlabel("Mean |SHAP Value|")

    ax = fig.add_subplot(gs[1,1])
    ax.scatter(yte, yhat, alpha=0.7)
    lim = [0, max(yte.max(), yhat.max())*1.05]; ax.plot(lim,lim,'r--',lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title(f"Model Performance (R² = {r2:.3f})"); ax.set_xlabel("Actual Sentence"); ax.set_ylabel("Predicted Sentence")
    plt.tight_layout(); fig.savefig(Path(out_dir,"tda_xai_analysis.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

    return {"r2":r2, "silhouette":sil, "df_clusters":df_clusters, "df_shap":df_shap}

# ================== B) Murder-only figures (3 races × 2 genders) ==================
def murder_outputs(df, out_dir, cache_dir=".", use_cache=True, force_refresh=False):
    all_csv, samp_csv, meta_json = cache_paths(cache_dir)

    murder_all = murder_sample = meta = None
    if use_cache and not force_refresh and all_csv.exists() and samp_csv.exists() and meta_json.exists():
        try:
            murder_all = pd.read_csv(all_csv)
            murder_sample = pd.read_csv(samp_csv)
            meta = json.loads(Path(meta_json).read_text(encoding="utf-8"))
            log("Loaded murder cache.")
        except Exception:
            murder_all = murder_sample = meta = None

    if murder_sample is None:
        target = first_existing(df, TARGET_CHOICES)
        df[target] = pd.to_numeric(df[target], errors="coerce")
        df = df.dropna(subset=[target])
        mask, cols = is_murder_rowblock(df)
        if not mask.any():
            log("No murder cases detected; skipping murder charts.")
            return
        m = df[mask].copy()

        # keep and clean
        keep = [c for c in (NUM_FEATS+CAT_FEATS+[target]) if c in m.columns]
        m = m[keep + [c for c in m.columns if _norm(c) in {"MONSEX","MON SEX","GENDER","SEX","DEFSEX"}]].copy()

        # numerics
        for c in NUM_FEATS:
            if c in m.columns:
                s = pd.to_numeric(m[c], errors="coerce")
                m[c] = s.fillna(s.median(skipna=True))
        q99 = m[target].quantile(0.99); m.loc[m[target]>q99,target]=q99

        # race & gender
        if "MONRACE" in m.columns:
            m["RaceName"] = map_race(m["MONRACE"])
        else:
            m["RaceName"] = pd.Categorical(["Other"]*len(m))
        m["GenderName"] = map_gender(m)

        # collapse to proposal races and filter
        m = m[m["RaceName"].astype(str).isin(RACES_POC)].copy()
        m["RaceName"] = pd.Categorical(m["RaceName"].astype(str), categories=RACES_POC, ordered=True)
        m["GenderName"] = pd.Categorical(m["GenderName"].astype(str), categories=["Female","Male","Other","Unknown"], ordered=True)
        m["Group"] = (m["RaceName"].astype(str) + " " + m["GenderName"].astype(str)).astype(str)

        # cache
        murder_all = m
        murder_sample = m.sample(n=min(N_SAMPLE,len(m)), random_state=SEED)
        meta = {"target":target, "hits":cols, "n_all":int(len(m)), "n_sample":int(len(murder_sample))}
        murder_all.to_csv(all_csv, index=False)
        murder_sample.to_csv(samp_csv, index=False)
        Path(meta_json).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        log(f"Saved murder cache (N_all={len(murder_all)}, N_sample={len(murder_sample)})")

    m = murder_sample.copy()

    # ---------- Bar chart (fixed 6 groups) ----------
    grp = (m.groupby(["RaceName","GenderName"])["SENTTOT" if "SENTTOT" in m.columns else m.columns[-1]]
           .mean().rename("mean").reset_index())
    grp["Group"] = (grp["RaceName"].astype(str) + " " + grp["GenderName"].astype(str)).astype(str)
    grp = grp.set_index("Group").reindex(BAR_ORDER).dropna().copy()

    # counts
    cnt = m.groupby("Group").size().reindex(BAR_ORDER).fillna(0).astype(int)

    fig = plt.figure(figsize=(12,8), dpi=140)
    bars = plt.bar(grp.index, grp["mean"].values, color=BAR_COLORS[:len(grp)])
    plt.xticks(rotation=45, ha="right")
    for b, n in zip(bars, cnt.values):
        plt.text(b.get_x()+b.get_width()/2, b.get_height()+4, f"n={n}", ha="center", va="bottom", fontsize=9)
    plt.title("2023 Murder Sentences by Gender and Race")
    plt.ylabel("Average Sentence (Months)"); plt.xlabel("Demographic Group")
    plt.tight_layout(); fig.savefig(Path(out_dir,"murder_sentences_2023.png")); plt.close(fig)

    # ---------- Intersectional panels (A–D) ----------
    target = first_existing(m, TARGET_CHOICES)
    # force 3×2 ordering
    r_order = RACES_POC
    g_order = ["Female","Male"]

    # pivot means & counts
    means = (m.pivot_table(index="RaceName", columns="GenderName", values=target, aggfunc="mean")
             .reindex(index=r_order, columns=g_order))
    counts = (m.pivot_table(index="RaceName", columns="GenderName", values=target, aggfunc="count")
              .reindex(index=r_order, columns=g_order)).fillna(0)

    # additive expectation & intersectional effects
    overall = m[target].mean()
    race_mean = m.groupby("RaceName")[target].mean()
    gen_mean  = m.groupby("GenderName")[target].mean()
    pairs = means.stack().rename("actual").reset_index()
    pairs["pred_add"] = pairs.apply(lambda r: overall +
                                    (race_mean[r["RaceName"]] - overall) +
                                    (gen_mean[r["GenderName"]] - overall), axis=1)
    pairs["effect"] = pairs["actual"] - pairs["pred_add"]

    # ANOVA (3×2)
    a = m[[target,"RaceName","GenderName"]].dropna().copy()
    a["RaceName"]   = a["RaceName"].astype("category")
    a["GenderName"] = a["GenderName"].astype("category")
    lm = smf.ols(f"{target} ~ C(RaceName)*C(GenderName)", data=a).fit()
    an = sm.stats.anova_lm(lm, typ=2)

    fig = plt.figure(figsize=(13,11), dpi=130)
    gs = gridspec.GridSpec(3,2, hspace=0.35, wspace=0.25, height_ratios=[1,1,0.9])

    # A
    ax = fig.add_subplot(gs[0,0])
    im = ax.imshow(means.to_numpy(dtype=float), aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(len(g_order))); ax.set_xticklabels(g_order)
    ax.set_yticks(range(len(r_order))); ax.set_yticklabels(r_order)
    ax.set_title("A. Mean Sentence Length by Race and Gender")
    cb = fig.colorbar(im, ax=ax); cb.set_label("Mean Sentence (Months)")
    # annotate cells
    for i, r in enumerate(r_order):
        for j, g in enumerate(g_order):
            val = means.loc[r, g]
            if pd.notna(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", color="white" if val>means.values.mean() else "black")

    # B
    ax = fig.add_subplot(gs[0,1])
    lab = [f"{r} {g}" for r in r_order for g in g_order]
    eff = pairs.set_index(["RaceName","GenderName"]).loc[pd.MultiIndex.from_product([r_order,g_order]), "effect"].values
    ax.bar(lab, eff); ax.axhline(0, color="k", lw=0.8)
    ax.set_xticklabels(lab, rotation=45, ha="right")
    ax.set_title("B. Intersectional Effects: Actual vs Additive"); ax.set_ylabel("Difference (Months)")

    # C
    ax = fig.add_subplot(gs[1,0])
    groups = lab
    data = [m[(m["RaceName"]==r)&(m["GenderName"]==g)][target].values for r in r_order for g in g_order]
    ax.violinplot(data, showmeans=True, showmedians=True, showextrema=False)
    ax.set_xticks(range(1,len(groups)+1)); ax.set_xticklabels(groups, rotation=45, ha="right")
    ax.set_title("C. Sentence Distribution by Intersectional Identity"); ax.set_ylabel("Sentence Length (Months)")

    # D
    ax = fig.add_subplot(gs[1,1])
    im2 = ax.imshow(counts.to_numpy(dtype=float), aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(g_order))); ax.set_xticklabels(g_order)
    ax.set_yticks(range(len(r_order))); ax.set_yticklabels(r_order)
    for i, r in enumerate(r_order):
        for j, g in enumerate(g_order):
            ax.text(j, i, str(int(counts.loc[r,g] or 0)), ha="center", va="center")
    ax.set_title("D. Sample Sizes by Group")
    fig.colorbar(im2, ax=ax).set_label("Count")

    # E – compact text box
    ax = fig.add_subplot(gs[2,:]); ax.axis("off")
    def pes(term):
        ss_err = float(an.loc["Residual","sum_sq"])
        if term not in an.index: return np.nan
        ss = float(an.loc[term,"sum_sq"]); return ss/(ss+ss_err) if ss_err>0 else np.nan
    lines = [
        f"Race: F={an.loc['C(RaceName)','F']:.2f}, p={an.loc['C(RaceName)','PR(>F)']:.4g}, η²ₚ={pes('C(RaceName)'):.3f}",
        f"Gender: F={an.loc['C(GenderName)','F']:.2f}, p={an.loc['C(GenderName)','PR(>F)']:.4g}, η²ₚ={pes('C(GenderName)'):.3f}",
        f"Race×Gender: F={an.loc['C(RaceName):C(GenderName)','F']:.2f}, p={an.loc['C(RaceName):C(GenderName)','PR(>F)']:.4g}, η²ₚ={pes('C(RaceName):C(GenderName)'):.3f}",
    ]
    mx = pairs.loc[pairs["effect"].idxmax()]
    mn = pairs.loc[pairs["effect"].idxmin()]
    text = ("STATISTICAL ANALYSIS RESULTS\n"
            + "\n".join(lines) + "\n\n"
            + "KEY FINDINGS:\n"
            + f" - Largest + effect: {mx['RaceName']} {mx['GenderName']} (+{mx['effect']:.1f} mo)\n"
            + f" - Largest − effect: {mn['RaceName']} {mn['GenderName']} ({mn['effect']:.1f} mo)\n"
            + f" - Murder subset N={len(m)}")
    ax.text(0.02, 0.98, text, va="top", ha="left", fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", fc="#f2f2f2", ec="#999"))

    plt.tight_layout(); fig.savefig(Path(out_dir,"intersectional_real_analysis_2023.png")); plt.close(fig)

# ---------------- Runner ----------------
def run_all(input_path, out_dir=".", cache_dir=".", use_cache=True, force_refresh=False):
    out_dir = _ensure_dir(out_dir)
    df = smart_read(input_path); log(f"Loaded {len(df):,} rows")

    # A) full sample TDA/XAI
    res = tda_xai_full(df, out_dir)
    log("Saved: tda_xai_analysis.png, cluster_table.tex, summary_table.tex")

    # B) murder-only figures
    murder_outputs(df, out_dir, cache_dir=cache_dir, use_cache=use_cache, force_refresh=force_refresh)
    log("Saved: murder_sentences_2023.png, intersectional_real_analysis_2023.png")

    return {"r2":res["r2"], "silhouette":res["silhouette"], "figure":"tda_xai_analysis.png",
            "murder":"murder_sentences_2023.png", "intersectional":"intersectional_real_analysis_2023.png"}
