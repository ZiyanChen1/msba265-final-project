"""
Data quality audit and cleaning pipeline for Predicting Loan Payback (Kaggle S5E11).

Mirrors the steps in 01_eda.ipynb (Steps 1-11 and 13):
  1. Audit: schema, target, missing values, duplicates, value ranges, categories, outliers
  2. Transform: log income, log DTI, ordinal grade
  3. Save: audit tables, figures, and cleaned train/test files

Usage:
    python src/eda_audit.py
    python src/eda_audit.py --data-dir data --out-dir reports
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

TARGET = "loan_paid_back"
ID_COL = "id"
NUM_COLS = ["annual_income", "debt_to_income_ratio", "credit_score", "loan_amount", "interest_rate"]
CAT_COLS = ["gender", "marital_status", "education_level", "employment_status", "loan_purpose", "grade_subgrade"]
FEATURES = NUM_COLS + CAT_COLS

RULES = {
    "annual_income": (0, 10_000_000),
    "debt_to_income_ratio": (0, 5),
    "credit_score": (300, 850),
    "loan_amount": (1, 10_000_000),
    "interest_rate": (0, 100),
}

GRADE_ORDER = [f"{letter}{num}" for letter in "ABCDEF" for num in range(1, 6)]
GRADE_MAP = {grade: i + 1 for i, grade in enumerate(GRADE_ORDER)}


# ---------------------------------------------------------------- load
def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    return train, test


def check_schema(train: pd.DataFrame, test: pd.DataFrame) -> None:
    expected = set(FEATURES + [ID_COL, TARGET])
    if set(train.columns) != expected:
        raise ValueError(f"Unexpected train columns: {set(train.columns) ^ expected}")
    if set(test.columns) != expected - {TARGET}:
        raise ValueError(f"Unexpected test columns: {set(test.columns) ^ (expected - {TARGET})}")


def fix_target(train: pd.DataFrame) -> None:
    values = set(train[TARGET].unique())
    if not values <= {0, 1}:
        raise ValueError(f"Target has values other than 0/1: {values}")
    train[TARGET] = train[TARGET].astype(int)


# ---------------------------------------------------------------- checks
def check_missing(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "train_missing": train.isna().sum(),
        "test_missing": test.isna().sum(),
    }).fillna(0).astype(int)


def check_duplicates(train: pd.DataFrame) -> dict:
    return {
        "duplicate_rows": int(train.duplicated(subset=FEATURES).sum()),
        "conflicting_labels": int((train.groupby(FEATURES)[TARGET].nunique() > 1).sum()),
    }


def check_boundaries(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, (lo, hi) in RULES.items():
        for name, df in [("train", train), ("test", test)]:
            rows.append({
                "column": col,
                "set": name,
                "rule_min": lo,
                "rule_max": hi,
                "actual_min": round(df[col].min(), 2),
                "actual_max": round(df[col].max(), 2),
                "violations": int(((df[col] < lo) | (df[col] > hi)).sum()),
            })
    return pd.DataFrame(rows)


def check_categories(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in CAT_COLS:
        train_values = set(train[col].unique())
        new_in_test = set(test[col].unique()) - train_values
        rows.append({
            "column": col,
            "n_levels": len(train_values),
            "has_spaces": any(v != v.strip() for v in train_values),
            "case_clash": len({v.strip().lower() for v in train_values}) < len(train_values),
            "new_in_test": ", ".join(sorted(new_in_test)) if new_in_test else "none",
            "smallest_level_%": round(train[col].value_counts(normalize=True).min() * 100, 2),
        })
    return pd.DataFrame(rows)


def category_payback(train: pd.DataFrame) -> pd.DataFrame:
    tables = []
    for col in CAT_COLS:
        counts = train[col].value_counts()
        table = pd.DataFrame({
            "feature": col,
            "level": counts.index,
            "count": counts.values,
            "share_%": (counts / len(train) * 100).round(2).values,
            "payback_rate_%": (train.groupby(col)[TARGET].mean() * 100).round(1).loc[counts.index].values,
        })
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def tukey_bounds(s: pd.Series, k: float = 1.5) -> tuple[float, float]:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def check_outliers(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in NUM_COLS:
        lo, hi = tukey_bounds(train[col])
        is_out = (train[col] < lo) | (train[col] > hi)
        rows.append({
            "column": col,
            "lower": round(lo, 3),
            "upper": round(hi, 3),
            "outlier_%": round(is_out.mean() * 100, 2),
            "payback_rate_outliers": round(train.loc[is_out, TARGET].mean(), 3),
            "payback_rate_others": round(train.loc[~is_out, TARGET].mean(), 3),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- transform
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Fixed-rule transforms only (nothing learned from data), so safe for train and test."""
    df = df.copy()
    df["log_annual_income"] = np.log1p(df["annual_income"])
    df["log_debt_to_income_ratio"] = np.log(df["debt_to_income_ratio"])
    df["grade_ordinal"] = df["grade_subgrade"].map(GRADE_MAP)

    new_cols = ["log_annual_income", "log_debt_to_income_ratio", "grade_ordinal"]
    if df[new_cols].isna().any().any():
        raise ValueError("Transform produced missing values (unknown grade or non-positive DTI).")
    return df


def skewness_table(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    pairs = {
        "annual_income": "log_annual_income",
        "debt_to_income_ratio": "log_debt_to_income_ratio",
    }
    return pd.DataFrame({
        "column": list(pairs),
        "skew_before": [round(before[raw].skew(), 2) for raw in pairs],
        "skew_after": [round(after[logged].skew(), 2) for logged in pairs.values()],
    })


# ---------------------------------------------------------------- figures
def save_boxplots(train: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, len(NUM_COLS), figsize=(20, 4))
    for ax, col in zip(axes, NUM_COLS):
        sns.boxplot(y=train[col], ax=ax, color="#8fb3d9")
        ax.set_title(col)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_skew_plots(clean: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sns.histplot(clean["annual_income"], kde=True, ax=axes[0, 0])
    sns.histplot(clean["log_annual_income"], kde=True, ax=axes[0, 1])
    sns.histplot(clean["debt_to_income_ratio"], kde=True, ax=axes[1, 0])
    sns.histplot(clean["log_debt_to_income_ratio"], kde=True, ax=axes[1, 1])
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_correlation(clean: pd.DataFrame, path: Path) -> pd.DataFrame:
    corr_cols = [
        "log_annual_income",
        "log_debt_to_income_ratio",
        "credit_score",
        "loan_amount",
        "interest_rate",
        "grade_ordinal",
        TARGET,
    ]
    corr = clean[corr_cols].corr().round(2)

    fig = plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True)
    plt.title("Correlation Heatmap (after transformation)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return corr


def save_visualizations(clean: pd.DataFrame, out_dir: Path) -> None:
    """The six charts from Step 11 of 01_eda.ipynb."""
    overall = clean[TARGET].mean() * 100

    # 1. target distribution
    counts = clean[TARGET].value_counts().sort_index()
    fig = plt.figure(figsize=(5, 4))
    plt.bar(["Not repaid (0)", "Repaid (1)"], counts.values, color=["#F58518", "#4C78A8"])
    plt.title("Target distribution")
    plt.ylabel("Number of loans")
    fig.tight_layout()
    fig.savefig(out_dir / "target_distribution.png", dpi=120)
    plt.close(fig)

    # 2. payback rate by category
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, col in zip(axes.flat, CAT_COLS[:-1]):
        rates = clean.groupby(col)[TARGET].mean().sort_values(ascending=False) * 100
        ax.bar(rates.index, rates.values, color="#4C78A8")
        ax.axhline(overall, color="#F58518", linestyle="--")
        ax.set_title(col)
        ax.set_ylabel("Payback rate (%)")
        ax.set_ylim(0, 100)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    axes.flat[-1].axis("off")
    fig.suptitle(f"Payback rate by category (dashed line = overall {overall:.1f}%)")
    fig.tight_layout()
    fig.savefig(out_dir / "payback_by_category.png", dpi=120)
    plt.close(fig)

    # 3. numeric features by target
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, col in zip(axes, ["log_debt_to_income_ratio", "credit_score", "interest_rate"]):
        sns.kdeplot(data=clean, x=col, hue=TARGET, common_norm=False, fill=True, ax=ax)
        ax.set_title(col)
    fig.tight_layout()
    fig.savefig(out_dir / "numeric_by_target.png", dpi=120)
    plt.close(fig)

    # 4. payback rate by DTI decile
    dti_band = pd.qcut(clean["debt_to_income_ratio"], 10)
    band_rate = clean.groupby(dti_band, observed=True)[TARGET].mean() * 100
    fig = plt.figure(figsize=(10, 4))
    plt.plot(range(1, 11), band_rate.values, marker="o")
    plt.xticks(range(1, 11))
    plt.title("Payback rate by DTI decile (1 = lowest DTI, 10 = highest)")
    plt.xlabel("DTI decile")
    plt.ylabel("Payback rate (%)")
    fig.tight_layout()
    fig.savefig(out_dir / "payback_by_dti_decile.png", dpi=120)
    plt.close(fig)

    # 5. payback rate by loan grade
    grade_rate = clean.groupby("grade_subgrade")[TARGET].mean().reindex(GRADE_ORDER) * 100
    fig = plt.figure(figsize=(12, 4))
    plt.plot(grade_rate.index, grade_rate.values, marker="o")
    plt.title("Payback rate by loan grade (A1 = best, F5 = worst)")
    plt.ylabel("Payback rate (%)")
    plt.xticks(rotation=90)
    fig.tight_layout()
    fig.savefig(out_dir / "payback_by_grade.png", dpi=120)
    plt.close(fig)

    # 6. credit score vs interest rate
    sample = clean.sample(5000, random_state=42)
    fig = plt.figure(figsize=(7, 5))
    sns.scatterplot(data=sample, x="credit_score", y="interest_rate", hue=TARGET, alpha=0.5, s=15)
    plt.title("Credit score vs interest rate (5,000-loan sample)")
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_credit_vs_rate.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- summary
def print_summary(train, test, missing, dups, bounds, cats, outliers, skew, corr) -> None:
    repaid = train[TARGET].mean() * 100
    high_corr = [
        (a, b, corr.loc[a, b])
        for i, a in enumerate(corr.columns)
        for b in corr.columns[i + 1:]
        if abs(corr.loc[a, b]) >= 0.85
    ]

    print("=" * 60)
    print("DATA QUALITY AUDIT — Predicting Loan Payback")
    print("=" * 60)
    print(f"Train rows: {len(train):,}   Test rows: {len(test):,}")
    print(f"Target: {repaid:.2f}% repaid / {100 - repaid:.2f}% not repaid")
    print(f"Missing values (train/test): {missing['train_missing'].sum()} / {missing['test_missing'].sum()}")
    print(f"Duplicate rows: {dups['duplicate_rows']}   Conflicting labels: {dups['conflicting_labels']}")
    print(f"Boundary violations: {bounds['violations'].sum()}")
    print(f"Category issues: {int(cats['has_spaces'].sum() + cats['case_clash'].sum())}   "
          f"New levels in test: {int((cats['new_in_test'] != 'none').sum())}")

    print("\nOutliers (Tukey 1.5 x IQR):")
    print(outliers.to_string(index=False))

    print("\nSkewness before / after log transform:")
    print(skew.to_string(index=False))

    print("\nHighly correlated pairs (|r| >= 0.85):")
    for a, b, r in high_corr:
        print(f"  {a} vs {b}: r = {r:.2f}")
    print("=" * 60)


# ---------------------------------------------------------------- main
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the data quality audit and save cleaned data.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. load and audit
    train, test = load_data(args.data_dir)
    check_schema(train, test)
    fix_target(train)

    missing = check_missing(train, test)
    dups = check_duplicates(train)
    bounds = check_boundaries(train, test)
    cats = check_categories(train, test)
    payback = category_payback(train)
    outliers = check_outliers(train)

    # 2. transform (same fixed rules for train and test)
    train_clean = transform(train)
    test_clean = transform(test)
    skew = skewness_table(train, train_clean)

    # 3. save tables, figures, and cleaned data
    missing.to_csv(args.out_dir / "audit_missing.csv")
    bounds.to_csv(args.out_dir / "audit_boundaries.csv", index=False)
    cats.to_csv(args.out_dir / "audit_categories.csv", index=False)
    payback.to_csv(args.out_dir / "category_payback.csv", index=False)
    outliers.to_csv(args.out_dir / "audit_outliers.csv", index=False)
    skew.to_csv(args.out_dir / "skewness.csv", index=False)

    save_boxplots(train, args.out_dir / "boxplots.png")
    save_skew_plots(train_clean, args.out_dir / "skewness_before_after.png")
    corr = save_correlation(train_clean, args.out_dir / "correlation_heatmap.png")
    save_visualizations(train_clean, args.out_dir)

    train_clean.to_csv(args.data_dir / "train_clean.csv", index=False)
    test_clean.to_csv(args.data_dir / "test_clean.csv", index=False)

    print_summary(train, test, missing, dups, bounds, cats, outliers, skew, corr)
    print(f"Reports saved to:      {args.out_dir.resolve()}")
    print(f"Cleaned data saved to: {args.data_dir.resolve()}  "
          f"(train {train_clean.shape}, test {test_clean.shape})")


if __name__ == "__main__":
    main()
