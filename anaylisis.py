import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import spearmanr, chi2_contingency, kruskal
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings("ignore")

# =========================
# CONFIGURATION
# =========================

DATASET_PATH = "memotion_dataset_7k/labels.csv"
OUTPUT_DIR = "."  # change if needed

SARCASM_MAP = {
    "not_sarcastic": 0,
    "general": 1,
    "twisted_meaning": 2,
    "very_twisted": 3
}

SENTIMENT_MAP = {
    "very_negative": -2,
    "negative": -1,
    "neutral": 0,
    "positive": 1,
    "very_positive": 2
}

# Memotion emotion columns are string-encoded — map them explicitly
EMOTION_COLS = ["humour", "offensive", "motivational"]

HUMOUR_MAP = {
    "not_funny": 0,
    "funny": 1,
    "very_funny": 2,
    "hilarious": 3
}

OFFENSIVE_MAP = {
    "not_offensive": 0,
    "slight": 1,
    "very_offensive": 2,
    "hateful_offensive": 3
}

MOTIVATIONAL_MAP = {
    "not_motivational": 0,
    "motivational": 1
}

EMOTION_MAPS = {
    "humour": HUMOUR_MAP,
    "offensive": OFFENSIVE_MAP,
    "motivational": MOTIVATIONAL_MAP,
}

SARCASM_ORDER = ["not_sarcastic", "general", "twisted_meaning", "very_twisted"]
SENTIMENT_ORDER = ["very_negative", "negative", "neutral", "positive", "very_positive"]

plt.rcParams.update({
    "figure.facecolor": "#0f0f14",
    "axes.facecolor": "#1a1a24",
    "axes.edgecolor": "#3a3a50",
    "axes.labelcolor": "#c8c8e8",
    "xtick.color": "#9090b8",
    "ytick.color": "#9090b8",
    "text.color": "#e0e0f8",
    "grid.color": "#2a2a3a",
    "grid.alpha": 0.5,
    "font.family": "monospace",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

PALETTE = {
    "not_sarcastic": "#4fc3f7",
    "general":        "#7e57c2",
    "twisted_meaning":"#ef5350",
    "very_twisted":   "#ff6f00",
    "accent":         "#a259ff",
    "positive":       "#66bb6a",
    "negative":       "#ef5350",
    "neutral":        "#78909c",
}

# =========================
# HELPERS
# =========================

def cramers_v(table: pd.DataFrame) -> float:
    """Effect size for chi-square on a contingency table."""
    chi2 = chi2_contingency(table)[0]
    n = table.values.sum()
    min_dim = min(table.shape) - 1
    return float(np.sqrt(chi2 / (n * min_dim))) if min_dim > 0 else 0.0


def significance_label(p: float) -> str:
    if p < 0.001:
        return "p < 0.001 ***"
    elif p < 0.01:
        return "p < 0.01 **"
    elif p < 0.05:
        return "p < 0.05 *"
    else:
        return f"p = {p:.4f} (n.s.)"


def section(title: str) -> None:
    bar = "=" * 50
    print(f"\n{bar}\n  {title}\n{bar}")


# =========================
# LOAD & VALIDATE
# =========================

section("LOADING DATASET")

df = pd.read_csv(DATASET_PATH)
print(f"Raw shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Normalise column names (strip whitespace, lowercase)
df.columns = df.columns.str.strip().str.lower()
print(f"Normalised columns: {df.columns.tolist()}")

# Verify required columns exist
required = ["sarcasm", "overall_sentiment"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(
        f"Required columns not found: {missing}\n"
        f"Available: {df.columns.tolist()}"
    )

# =========================
# MAP TO NUMERIC
# =========================

section("ENCODING LABELS")

df["sarcasm_raw"] = df["sarcasm"].str.strip().str.lower()
df["sentiment_raw"] = df["overall_sentiment"].str.strip().str.lower()

df["sarcasm_score"] = df["sarcasm_raw"].map(SARCASM_MAP)
df["sentiment_score"] = df["sentiment_raw"].map(SENTIMENT_MAP)

# Report unmapped values
for col, raw_col in [("sarcasm_score", "sarcasm_raw"),
                     ("sentiment_score", "sentiment_raw")]:
    n_nan = df[col].isna().sum()
    if n_nan > 0:
        unique_bad = df.loc[df[col].isna(), raw_col].unique()
        print(f"  WARNING: {n_nan} NaNs in {col} from values: {unique_bad}")

before = len(df)
df = df.dropna(subset=["sarcasm_score", "sentiment_score"])
after = len(df)
print(f"\nDropped {before - after} rows with unmapped labels.")
print(f"Working dataset: {after} rows\n")

# Encode sarcasm as binary (sarcastic vs not)
df["is_sarcastic"] = (df["sarcasm_score"] > 0).astype(int)

# =========================
# DISTRIBUTIONS
# =========================

section("CLASS DISTRIBUTIONS")

print("Sarcasm distribution:")
vc = df["sarcasm_raw"].value_counts()
for label, cnt in vc.items():
    pct = 100 * cnt / len(df)
    print(f"  {label:<20} {cnt:>5}  ({pct:.1f}%)")

print("\nSentiment distribution:")
vc2 = df["sentiment_raw"].value_counts()
for label, cnt in vc2.items():
    pct = 100 * cnt / len(df)
    print(f"  {label:<20} {cnt:>5}  ({pct:.1f}%)")

# =========================
# SPEARMAN CORRELATION
# =========================

section("SPEARMAN RANK CORRELATION  (sarcasm ↔ sentiment)")

corr, p = spearmanr(df["sarcasm_score"], df["sentiment_score"])
print(f"  Spearman ρ : {corr:+.4f}")
print(f"  {significance_label(p)}")
print(f"\n  Interpretation:")
if abs(corr) < 0.1:
    strength = "negligible"
elif abs(corr) < 0.3:
    strength = "weak"
elif abs(corr) < 0.5:
    strength = "moderate"
else:
    strength = "strong"
direction = "positive" if corr > 0 else "negative"
print(f"  → {strength.capitalize()} {direction} correlation between sarcasm level and sentiment.")

# =========================
# CHI-SQUARE + CRAMER'S V
# =========================

section("CHI-SQUARE TEST  (sarcasm category × sentiment)")

# Re-order for readability
df["sarcasm_raw"] = pd.Categorical(df["sarcasm_raw"],
                                   categories=SARCASM_ORDER, ordered=True)
df["sentiment_raw"] = pd.Categorical(df["sentiment_raw"],
                                     categories=SENTIMENT_ORDER, ordered=True)

table = pd.crosstab(df["sarcasm_raw"], df["sentiment_raw"])
print("\nContingency table (counts):")
print(table.to_string())

chi2, chi_p, dof, expected = chi2_contingency(table)
cv = cramers_v(table)

print(f"\n  Chi-square : {chi2:.2f}")
print(f"  df         : {dof}")
print(f"  {significance_label(chi_p)}")
print(f"  Cramér's V : {cv:.4f}  (effect size: {'small' if cv < 0.1 else 'medium' if cv < 0.3 else 'large'})")

# =========================
# KRUSKAL-WALLIS TEST
# =========================

section("KRUSKAL-WALLIS TEST  (sentiment score across sarcasm groups)")

groups = [
    df.loc[df["sarcasm_raw"] == cat, "sentiment_score"].values
    for cat in SARCASM_ORDER
    if cat in df["sarcasm_raw"].values
]
h_stat, kw_p = kruskal(*groups)
print(f"  H statistic : {h_stat:.4f}")
print(f"  {significance_label(kw_p)}")
print("\n  → If significant, sarcasm categories differ in sentiment distribution.")

# =========================
# MEAN SENTIMENT PER SARCASM LEVEL
# =========================

section("DESCRIPTIVE STATS  (sentiment by sarcasm level)")

stats = df.groupby("sarcasm_raw", observed=True)["sentiment_score"].agg(
    ["mean", "median", "std", "count"]
).rename(columns={"mean": "Mean", "median": "Median",
                   "std": "Std", "count": "N"})
print(stats.to_string())

# =========================
# EMOTION COLUMNS (if available)
# =========================

section("EMOTION COLUMNS  (if available in dataset)")

available_emotions = [c for c in ["humour", "offensive", "motivational"]
                      if c in df.columns]

if available_emotions:
    print(f"Found emotion columns: {available_emotions}\n")

    # Map string labels to ordinal numbers using explicit maps
    for col in available_emotions:
        mapped_col = f"{col}_score"
        emap = EMOTION_MAPS.get(col, {})
        df[mapped_col] = df[col].str.strip().str.lower().map(emap)
        n_nan = df[mapped_col].isna().sum()
        if n_nan > 0:
            unrecog = df.loc[df[mapped_col].isna(), col].unique()
            print(f"  WARNING: {n_nan} unmapped values in '{col}': {unrecog}")
        else:
            print(f"  '{col}' mapped OK — {df[mapped_col].notna().sum()} valid rows")

    print("\nSpearman rho between sarcasm_score and each emotion:")
    emotion_corrs = {}
    for col in available_emotions:
        mapped_col = f"{col}_score"
        series = df[mapped_col]
        valid = df["sarcasm_score"].notna() & series.notna()
        if valid.sum() < 10:
            print(f"  {col:<15}  (too few valid rows to correlate)")
            emotion_corrs[col] = (float("nan"), float("nan"))
            continue
        r, p_val = spearmanr(df.loc[valid, "sarcasm_score"], series[valid])
        emotion_corrs[col] = (r, p_val)
        print(f"  {col:<15}  rho = {r:+.4f}   {significance_label(p_val)}")
else:
    print("  No additional emotion columns found (humour / offensive / motivational).")
    print("  NOTE: If your CSV version has them under different names,")
    print("  add them to EMOTION_COLS at the top of this script.")

# =========================
# BINARY SARCASM ANALYSIS
# =========================

section("BINARY SARCASM  (sarcastic vs not-sarcastic)")

sarcastic = df.loc[df["is_sarcastic"] == 1, "sentiment_score"]
not_sarcastic = df.loc[df["is_sarcastic"] == 0, "sentiment_score"]

print(f"  Not sarcastic  — mean sentiment: {not_sarcastic.mean():+.3f}  (n={len(not_sarcastic)})")
print(f"  Sarcastic      — mean sentiment: {sarcastic.mean():+.3f}  (n={len(sarcastic)})")

u_stat, u_p = mannwhitneyu(sarcastic, not_sarcastic, alternative="two-sided")
print(f"\n  Mann-Whitney U = {u_stat:.1f}")
print(f"  {significance_label(u_p)}")

# =========================
# VISUALIZATIONS
# =========================

section("GENERATING FIGURES")

SARCASM_COLORS = [PALETTE[s] for s in SARCASM_ORDER
                  if s in df["sarcasm_raw"].cat.categories]

# ------------------------------------------------------------------
# FIGURE 1 — Sentiment distribution by sarcasm category
# ------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("#0f0f14")

counts = pd.crosstab(df["sentiment_raw"], df["sarcasm_raw"])
counts = counts.reindex(index=SENTIMENT_ORDER,
                        columns=[c for c in SARCASM_ORDER
                                 if c in counts.columns])
counts_pct = counts.div(counts.sum(axis=0), axis=1) * 100

x = np.arange(len(counts_pct.index))
width = 0.2
n_groups = len(counts_pct.columns)
offsets = np.linspace(-(n_groups - 1) / 2, (n_groups - 1) / 2, n_groups) * width

for i, (cat, color) in enumerate(zip(counts_pct.columns, SARCASM_COLORS)):
    bars = ax.bar(x + offsets[i], counts_pct[cat], width * 0.9,
                  label=cat, color=color, alpha=0.85, zorder=3)

ax.set_xticks(x)
ax.set_xticklabels(SENTIMENT_ORDER, rotation=15, ha="right")
ax.set_xlabel("Sentiment")
ax.set_ylabel("% within sarcasm group")
ax.set_title("Sentiment Profile by Sarcasm Level  (% within group)", pad=14)
ax.legend(title="Sarcasm", bbox_to_anchor=(1.01, 1), loc="upper left",
          framealpha=0.2, labelcolor="#e0e0f8", title_fontsize=9)
ax.grid(axis="y", linewidth=0.5)
ax.set_facecolor("#1a1a24")

plt.tight_layout()
fname1 = f"{OUTPUT_DIR}/fig1_sentiment_profile.png"
plt.savefig(fname1, dpi=200, bbox_inches="tight")
plt.show()
print(f"  Saved: {fname1}")

# ------------------------------------------------------------------
# FIGURE 2 — Heatmap (normalised by row = sarcasm category)
# ------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.patch.set_facecolor("#0f0f14")

# Raw counts
sns.heatmap(
    table,
    annot=True, fmt="d", cmap="rocket_r",
    linewidths=0.4, linecolor="#0f0f14",
    ax=axes[0], cbar_kws={"shrink": 0.7}
)
axes[0].set_title("Raw Counts")
axes[0].set_xlabel("Sentiment")
axes[0].set_ylabel("Sarcasm category")
axes[0].tick_params(axis="x", rotation=30)

# Row-normalised (% within sarcasm level)
table_norm = table.div(table.sum(axis=1), axis=0) * 100
sns.heatmap(
    table_norm,
    annot=True, fmt=".1f", cmap="rocket_r",
    linewidths=0.4, linecolor="#0f0f14",
    ax=axes[1], cbar_kws={"shrink": 0.7, "label": "%"}
)
axes[1].set_title("Row-Normalised  (% within sarcasm level)")
axes[1].set_xlabel("Sentiment")
axes[1].set_ylabel("")
axes[1].tick_params(axis="x", rotation=30)

for ax in axes:
    ax.set_facecolor("#1a1a24")

fig.suptitle("Sarcasm × Sentiment Contingency", y=1.01, fontsize=14)
plt.tight_layout()
fname2 = f"{OUTPUT_DIR}/fig2_heatmap.png"
plt.savefig(fname2, dpi=200, bbox_inches="tight")
plt.show()
print(f"  Saved: {fname2}")

# ------------------------------------------------------------------
# FIGURE 3 — Violin + strip (richer than boxplot)
# ------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor("#0f0f14")

valid_cats = [c for c in SARCASM_ORDER if c in df["sarcasm_raw"].cat.categories]
plot_df = df[df["sarcasm_raw"].isin(valid_cats)].copy()

sns.violinplot(
    data=plot_df,
    x="sarcasm_raw",
    y="sentiment_score",
    order=valid_cats,
    palette={c: PALETTE[c] for c in valid_cats},
    inner=None,
    alpha=0.55,
    ax=ax
)
sns.stripplot(
    data=plot_df.sample(min(1500, len(plot_df)), random_state=42),
    x="sarcasm_raw",
    y="sentiment_score",
    order=valid_cats,
    palette={c: PALETTE[c] for c in valid_cats},
    alpha=0.25, size=2.5, jitter=True, ax=ax
)

# Overlay mean markers
means = plot_df.groupby("sarcasm_raw", observed=True)["sentiment_score"].mean()
for i, cat in enumerate(valid_cats):
    ax.plot(i, means[cat], "o", color="white", markersize=8, zorder=5,
            markeredgecolor="#0f0f14", markeredgewidth=1.5)
    ax.text(i, means[cat] + 0.15, f"{means[cat]:+.2f}",
            ha="center", va="bottom", fontsize=8.5, color="white")

ax.set_yticks([-2, -1, 0, 1, 2])
ax.set_yticklabels(["very_negative\n(−2)", "negative\n(−1)",
                    "neutral\n(0)", "positive\n(+1)", "very_positive\n(+2)"])
ax.set_xlabel("Sarcasm Category")
ax.set_ylabel("Sentiment Score")
ax.set_title("Sentiment Score Distribution by Sarcasm Category\n(white dot = mean)", pad=12)
ax.grid(axis="y", linewidth=0.5)
ax.set_facecolor("#1a1a24")

plt.tight_layout()
fname3 = f"{OUTPUT_DIR}/fig3_violin.png"
plt.savefig(fname3, dpi=200, bbox_inches="tight")
plt.show()
print(f"  Saved: {fname3}")

# ------------------------------------------------------------------
# FIGURE 4 — Emotion correlations (if available)
# ------------------------------------------------------------------

if available_emotions and emotion_corrs:
    fig, axes = plt.subplots(1, len(available_emotions),
                             figsize=(5 * len(available_emotions), 5), squeeze=False)
    fig.patch.set_facecolor("#0f0f14")
    axes = axes[0]

    for ax, col in zip(axes, available_emotions):
        mapped_col = f"{col}_score"
        series = df[mapped_col]
        valid = df["sarcasm_score"].notna() & series.notna()
        means_e = df[valid].groupby("sarcasm_raw", observed=True)[mapped_col].mean().reindex(valid_cats)

        ax.bar(range(len(valid_cats)), means_e.values,
               color=SARCASM_COLORS, alpha=0.8, zorder=3)
        ax.set_xticks(range(len(valid_cats)))
        ax.set_xticklabels(valid_cats, rotation=20, ha="right")
        ax.set_title(f"{col.capitalize()}\n(mean score per sarcasm level)")
        ax.set_ylabel("Mean score")
        r, p_val = emotion_corrs[col]
        lbl = f"rho = {r:+.3f}  {significance_label(p_val)}" if not np.isnan(r) else "no data"
        ax.set_xlabel(lbl, fontsize=8)
        ax.grid(axis="y", linewidth=0.5)
        ax.set_facecolor("#1a1a24")

    fig.suptitle("Emotion Scores by Sarcasm Level", y=1.02, fontsize=14)
    plt.tight_layout()
    fname4 = f"{OUTPUT_DIR}/fig4_emotions.png"
    plt.savefig(fname4, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {fname4}")

# ------------------------------------------------------------------
# FIGURE 5 — Summary statistics panel
# ------------------------------------------------------------------

fig = plt.figure(figsize=(14, 4))
fig.patch.set_facecolor("#0f0f14")

gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.45)

# Panel A: mean sentiment ± std
ax_a = fig.add_subplot(gs[0])
means_v = stats["Mean"].values
stds_v = stats["Std"].values
cats_v = list(stats.index)
colors_v = [PALETTE.get(c, "#aaaaaa") for c in cats_v]
bars_a = ax_a.barh(cats_v, means_v, xerr=stds_v,
                   color=colors_v, alpha=0.8, capsize=4,
                   error_kw={"ecolor": "white", "elinewidth": 1})
ax_a.axvline(0, color="#606080", linewidth=1, linestyle="--")
ax_a.set_xlabel("Mean sentiment score")
ax_a.set_title("Mean ± Std Sentiment\nper Sarcasm Level")
ax_a.set_facecolor("#1a1a24")

# Panel B: sarcasm pie
ax_b = fig.add_subplot(gs[1])
sarcasm_counts = df["sarcasm_raw"].value_counts().reindex(valid_cats).fillna(0)
wedge_colors = [PALETTE[c] for c in valid_cats]
wedges, texts, autotexts = ax_b.pie(
    sarcasm_counts.values,
    labels=valid_cats,
    colors=wedge_colors,
    autopct="%1.1f%%",
    startangle=140,
    textprops={"color": "#e0e0f8", "fontsize": 8},
    wedgeprops={"linewidth": 1.2, "edgecolor": "#0f0f14"}
)
for at in autotexts:
    at.set_fontsize(7.5)
ax_b.set_title("Sarcasm Class\nDistribution")

# Panel C: key stats text box
ax_c = fig.add_subplot(gs[2])
ax_c.axis("off")
text_lines = [
    "KEY STATISTICS",
    "",
    f"Spearman ρ = {corr:+.4f}",
    f"{significance_label(p)}",
    "",
    f"Cramér's V = {cv:.4f}",
    f"Chi² = {chi2:.1f}  df={dof}",
    f"{significance_label(chi_p)}",
    "",
    f"Kruskal-Wallis H = {h_stat:.2f}",
    f"{significance_label(kw_p)}",
    "",
    f"Mann-Whitney (binary)",
    f"  Sarcastic μ = {sarcastic.mean():+.3f}",
    f"  Not-sarc  μ = {not_sarcastic.mean():+.3f}",
    f"  {significance_label(u_p)}",
    "",
    f"N (after cleaning) = {len(df)}",
]
ax_c.text(0.05, 0.97, "\n".join(text_lines),
          transform=ax_c.transAxes,
          fontsize=8.5, va="top", ha="left",
          family="monospace",
          color="#e0e0f8",
          bbox=dict(facecolor="#1a1a24", edgecolor="#3a3a50",
                    boxstyle="round,pad=0.6", alpha=0.9))

fig.suptitle("Sarcasm–Sentiment Correlation Summary", fontsize=13, y=1.01)
plt.tight_layout()
fname5 = f"{OUTPUT_DIR}/fig5_summary.png"
plt.savefig(fname5, dpi=200, bbox_inches="tight")
plt.show()
print(f"  Saved: {fname5}")

# =========================
# FINAL SUMMARY
# =========================

section("ANALYSIS COMPLETE")

print(f"""
Figures produced:
  {fname1}   — Sentiment profile (% within group)
  {fname2}   — Heatmap (raw counts + row-normalised)
  {fname3}   — Violin + strip plot with means
  {"fig4_emotions.png" if available_emotions else "(fig4 skipped — no emotion cols found)"}
  {fname5}   — Summary panel

Statistical results at a glance:
  Spearman ρ = {corr:+.4f}   [{significance_label(p)}]
  Cramér's V = {cv:.4f}       (chi²={chi2:.1f}, {significance_label(chi_p)})
  Kruskal-Wallis H = {h_stat:.2f} [{significance_label(kw_p)}]

Sarcastic vs not-sarcastic mean sentiments:
  Sarcastic     : {sarcastic.mean():+.4f}
  Not sarcastic : {not_sarcastic.mean():+.4f}
  Mann-Whitney  : {significance_label(u_p)}
""")
