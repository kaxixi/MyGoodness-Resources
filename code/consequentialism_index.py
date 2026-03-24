#!/usr/bin/env python3
"""
MyGoodness Individual Preference Indices

Computes per-person indices and standard errors from MyGoodness study data:

1. Relative Consequentialism Index — sensitivity to efficiency (lives saved)
2. Locality Preference Index — preference for local (near) charities

Usage:
    python consequentialism_index.py export.json [--output results.csv]

Input:  JSON file from the export_study_data() RPC function.
Output: CSV with per-person indices, SEs, and metadata.

Methodology
-----------
Each index follows the same structure:

1. Estimate a "representative agent" model: pooled OLS of an oriented
   binary DV on task attributes. SEs clustered by individual.

2. For each person's eligible decisions, predict the representative
   agent's choice probability (y_hat_rep).

3. Define the maximal agent (y_hat_max = 1 for the preferred direction).

4. Index:
       I_i = sum(Y_it - y_hat_rep_it) / sum(1 - y_hat_rep_it)

5. Per-person SE (Bernoulli plug-in):
       SE(I_i) = sqrt(sum(p_hat_it * (1 - p_hat_it))) / D_i
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ── Configuration ────────────────────────────────────────────────

# Minimum eligible decisions per person to compute a raw index.
# With EB shrinkage, even n=1 produces a usable (heavily shrunk) estimate,
# but we still need at least 1 task with a nonzero denominator.
MIN_ELIGIBLE = 1

# Efficiency difference bins for the consequentialism model.
# Individual dummies for 1-15, wider bins for 16+.
# Reference category: diff = 0.
EFFICIENCY_BINS = [
    (1, 1), (2, 2), (3, 3), (4, 4), (5, 5),
    (6, 6), (7, 7), (8, 8), (9, 9), (10, 10),
    (11, 11), (12, 12), (13, 13), (14, 14), (15, 15),
    (16, 20),
    (21, 30),
    (31, 40),
    (41, 50),
    (51, 75),
    (76, 100),
    (101, 150),
    (151, 200),
    (201, 250),
    (251, 300),
]

# Location interaction structure for consequentialism model (ref: both_far):
#   Diffs 1-12:  interact with both_near and one_far
#   Diffs 13-15: interact with one_far only (both_near too sparse)
#   Diffs 16+:   per-bin one_far interactions


# ── Data Loading ─────────────────────────────────────────────────


def load_export(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load JSON export from export_study_data() RPC.

    The RPC returns: {"sessions": [...], "decisions": [...], "survey_responses": [...]}
    """
    with open(path) as f:
        data = json.load(f)

    sessions = pd.DataFrame(data["sessions"])
    decisions = pd.DataFrame(data["decisions"])

    print(f"Loaded {len(sessions)} sessions, {len(decisions)} decisions")

    # Filter to completed sessions
    completed = sessions[sessions["ended_at"].notna()]
    print(f"  {len(completed)} completed sessions")

    decisions = decisions[decisions["session_id"].isin(completed["id"])]
    print(f"  {len(decisions)} decisions from completed sessions")

    return sessions, decisions


# ── Variable Construction ────────────────────────────────────────


def bin_label(lo: int, hi: int) -> str:
    return str(lo) if lo == hi else f"{lo}-{hi}"


def assign_eff_bin(diff: int) -> str:
    """Assign an efficiency difference to a bin label."""
    if diff == 0:
        return "0"
    for lo, hi in EFFICIENCY_BINS:
        if lo <= diff <= hi:
            return bin_label(lo, hi)
    # Above max bin
    if diff > EFFICIENCY_BINS[-1][1]:
        return f"{EFFICIENCY_BINS[-1][0]}+"
    return "0"


def orient_attr(df: pd.DataFrame, side_col: str, left_col: str, right_col: str):
    """Return the attribute value from the side indicated by side_col ('left' or 'right')."""
    return np.where(df[side_col] == "left", df[left_col], df[right_col])


def orient_attr_opposite(df: pd.DataFrame, side_col: str, left_col: str, right_col: str):
    """Return the attribute value from the OPPOSITE side of side_col."""
    return np.where(df[side_col] == "left", df[right_col], df[left_col])


def prepare_decisions(decisions: pd.DataFrame) -> pd.DataFrame:
    """Add all analysis variables to the decisions DataFrame.

    Adds variables for both the consequentialism and locality indices.
    """
    df = decisions.copy()

    # Drop decisions without a choice
    n_before = len(df)
    df = df[df["choice"].notna()].copy()
    print(f"  Dropped {n_before - len(df)} unanswered decisions")

    # Ensure hidden fields are lists
    for col in ["left_hidden", "right_hidden"]:
        df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])

    # ── Shared flags ──
    df["nothing_hidden"] = (df["left_hidden"].apply(len) == 0) & (
        df["right_hidden"].apply(len) == 0
    )
    df["is_stranger_stranger"] = (
        ~df["left_is_kin"]
        & ~df["right_is_kin"]
        & ~df["left_who_self"]
        & ~df["right_who_self"]
    )

    # ── Location categories ──
    df["both_near"] = df["left_where_near"] & df["right_where_near"]
    df["both_far"] = ~df["left_where_near"] & ~df["right_where_near"]
    df["one_far"] = ~df["both_near"] & ~df["both_far"]
    df["location"] = np.where(
        df["both_near"], "both_near", np.where(df["one_far"], "one_far", "both_far")
    )

    # ── Efficiency (unsigned) ──
    df["eff_diff"] = (df["left_count"] - df["right_count"]).abs()
    df["eff_bin"] = df["eff_diff"].apply(assign_eff_bin)

    # ═══════════════════════════════════════════════════════════════
    # CONSEQUENTIALISM INDEX variables
    # ═══════════════════════════════════════════════════════════════

    # Which side is more effective?
    df["more_eff_side"] = np.where(
        df["left_count"] > df["right_count"],
        "left",
        np.where(df["right_count"] > df["left_count"], "right", "equal"),
    )
    df["chose_more_effective"] = np.where(
        df["more_eff_side"] == "equal",
        np.nan,
        (df["choice"] == df["more_eff_side"]).astype(float),
    )

    # Controls oriented to more-effective / less-effective side
    for attr, raw_col in [("victim", "who_name"), ("charity", "what_charity_name")]:
        left_has = df[f"left_{raw_col}"].notna()
        right_has = df[f"right_{raw_col}"].notna()
        df[f"c_{attr}_more_eff"] = np.where(
            df["more_eff_side"] == "left", left_has,
            np.where(df["more_eff_side"] == "right", right_has, False),
        ).astype(int)
        df[f"c_{attr}_less_eff"] = np.where(
            df["more_eff_side"] == "left", right_has,
            np.where(df["more_eff_side"] == "right", left_has, False),
        ).astype(int)

    for attr, raw_col in [("cause", "what_cause"), ("gender", "who_gender"), ("age", "who_age_group")]:
        df[f"c_{attr}_more_eff"] = np.where(
            df["more_eff_side"] == "left", df[f"left_{raw_col}"], df[f"right_{raw_col}"]
        )
        df[f"c_{attr}_less_eff"] = np.where(
            df["more_eff_side"] == "left", df[f"right_{raw_col}"], df[f"left_{raw_col}"]
        )

    # Eligibility: stranger-stranger, nothing hidden, efficiency differs
    df["elig_consequentialism"] = (
        df["is_stranger_stranger"]
        & df["nothing_hidden"]
        & df["chose_more_effective"].notna()
        & (df["eff_diff"] > 0)
    )

    n = df["elig_consequentialism"].sum()
    p = df.loc[df["elig_consequentialism"], "session_id"].nunique()
    print(f"  Consequentialism: {n} eligible decisions from {p} participants")

    # ═══════════════════════════════════════════════════════════════
    # LOCALITY INDEX variables
    # ═══════════════════════════════════════════════════════════════

    # Which side is local? (only defined when exactly one is near)
    df["local_side"] = np.where(
        df["left_where_near"] & ~df["right_where_near"],
        "left",
        np.where(~df["left_where_near"] & df["right_where_near"], "right", "neither"),
    )
    df["chose_local"] = np.where(
        df["local_side"] == "neither",
        np.nan,
        (df["choice"] == df["local_side"]).astype(float),
    )

    # Signed efficiency difference: local_count - far_count
    df["eff_diff_local_far"] = np.where(
        df["local_side"] == "left",
        df["left_count"] - df["right_count"],
        np.where(df["local_side"] == "right",
                 df["right_count"] - df["left_count"], 0),
    ).astype(float)

    # Controls oriented to local / far side
    for attr, raw_col in [("victim", "who_name"), ("charity", "what_charity_name")]:
        left_has = df[f"left_{raw_col}"].notna()
        right_has = df[f"right_{raw_col}"].notna()
        df[f"l_{attr}_local"] = np.where(
            df["local_side"] == "left", left_has,
            np.where(df["local_side"] == "right", right_has, False),
        ).astype(int)
        df[f"l_{attr}_far"] = np.where(
            df["local_side"] == "left", right_has,
            np.where(df["local_side"] == "right", left_has, False),
        ).astype(int)

    for attr, raw_col in [("cause", "what_cause"), ("gender", "who_gender"), ("age", "who_age_group")]:
        df[f"l_{attr}_local"] = orient_attr(df, "local_side", f"left_{raw_col}", f"right_{raw_col}")
        df[f"l_{attr}_far"] = orient_attr_opposite(df, "local_side", f"left_{raw_col}", f"right_{raw_col}")

    # Eligibility: stranger-stranger, nothing hidden, one near + one far
    df["elig_locality"] = (
        df["is_stranger_stranger"]
        & df["nothing_hidden"]
        & df["chose_local"].notna()
    )

    n = df["elig_locality"].sum()
    p = df.loc[df["elig_locality"], "session_id"].nunique()
    print(f"  Locality: {n} eligible decisions from {p} participants")

    return df


# ── Design Matrices ──────────────────────────────────────────────


def build_consequentialism_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Design matrix for the consequentialism representative agent model.

    Structure (reference: both_far for location, diff=0 for bins):
    1. Efficiency bin dummies (all bins, main effects)
    2. Bin x both_near interactions   — diffs 1-12 only
    3. Bin x one_far interactions     — all bins
    4. Controls: victim, charity, cause, gender, age (main effects)
    """
    parts = []

    # 1. Efficiency bin main effects
    bin_dummies = pd.get_dummies(data["eff_bin"], prefix="diff")
    bin_dummies = bin_dummies.drop(columns=["diff_0"], errors="ignore")
    parts.append(bin_dummies)

    # 2. Bin x both_near interactions (diffs 1-12)
    for d in range(1, 13):
        col = f"diff_{d}:both_near"
        data[col] = ((data["eff_diff"] == d) & data["both_near"]).astype(int)
        parts.append(data[[col]])

    # 3. Bin x one_far interactions (all bins)
    for d in range(1, 16):
        col = f"diff_{d}:one_far"
        data[col] = ((data["eff_diff"] == d) & data["one_far"]).astype(int)
        parts.append(data[[col]])
    for lo, hi in EFFICIENCY_BINS:
        if lo < 16:
            continue
        label = bin_label(lo, hi)
        col = f"diff_{label}:one_far"
        data[col] = ((data["eff_diff"] >= lo) & (data["eff_diff"] <= hi) & data["one_far"]).astype(int)
        parts.append(data[[col]])

    # 4. Controls
    parts.append(
        data[["c_victim_more_eff", "c_victim_less_eff",
              "c_charity_more_eff", "c_charity_less_eff"]].astype(float)
    )
    for attr in ["cause", "gender", "age"]:
        for side in ["more_eff", "less_eff"]:
            dummies = pd.get_dummies(data[f"c_{attr}_{side}"], prefix=f"c_{attr}_{side}", drop_first=True)
            parts.append(dummies)

    X = pd.concat(parts, axis=1)
    X = X.loc[:, X.std() > 0]  # drop empty cells
    X = sm.add_constant(X)
    return X


def build_locality_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Design matrix for the locality representative agent model.

    Structure:
    1. Signed efficiency polynomial (local_count - far_count): linear, quadratic, cubic
    2. Controls: victim, charity, cause, gender, age oriented to local/far side
    """
    parts = []

    # 1. Signed efficiency polynomial
    eff = data["eff_diff_local_far"].copy()
    parts.append(pd.DataFrame({
        "eff_local_far": eff,
        "eff_local_far_sq": eff ** 2,
        "eff_local_far_cu": eff ** 3,
    }, index=data.index))

    # 2. Controls
    parts.append(
        data[["l_victim_local", "l_victim_far",
              "l_charity_local", "l_charity_far"]].astype(float)
    )
    for attr in ["cause", "gender", "age"]:
        for side in ["local", "far"]:
            dummies = pd.get_dummies(data[f"l_{attr}_{side}"], prefix=f"l_{attr}_{side}", drop_first=True)
            parts.append(dummies)

    X = pd.concat(parts, axis=1)
    X = X.loc[:, X.std() > 0]
    X = sm.add_constant(X)
    return X


# ── Model Estimation & Prediction ────────────────────────────────


def estimate_model(
    df: pd.DataFrame,
    elig_col: str,
    dv_col: str,
    matrix_builder,
    model_type: str = "ols",
    label: str = "",
):
    """Estimate a representative agent model on eligible decisions.

    Returns (fitted_model, feature_columns).
    """
    est = df[df[elig_col]].copy()

    X = matrix_builder(est)
    y = est[dv_col].astype(float)

    cluster_kwds = {"groups": est["session_id"].values}

    if model_type == "logit":
        model = sm.Logit(y, X).fit(
            cov_type="cluster", cov_kwds=cluster_kwds, disp=False,
        )
        print(f"\n{label} representative agent model (logit):")
        print(f"  N = {model.nobs:.0f} decisions, {len(X.columns)} parameters")
        print(f"  Pseudo R² = {model.prsquared:.3f}")
    else:
        model = sm.OLS(y, X).fit(
            cov_type="cluster", cov_kwds=cluster_kwds,
        )
        print(f"\n{label} representative agent model (OLS):")
        print(f"  N = {model.nobs:.0f} decisions, {len(X.columns)} parameters, R² = {model.rsquared:.3f}")

    print(f"  Intercept = {model.params['const']:.3f}")

    return model, X.columns.tolist()


def predict_model(
    df: pd.DataFrame,
    elig_col: str,
    model,
    feature_cols: list[str],
    matrix_builder,
) -> pd.Series:
    """Generate representative-agent predictions for eligible decisions."""
    eligible = df[df[elig_col]].copy()

    X = matrix_builder(eligible)

    # Align columns
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feature_cols]

    preds = model.predict(X)
    preds = preds.clip(0.01, 0.99)

    return pd.Series(preds.values, index=eligible.index, name="y_hat_rep")


# ── Generic Index Computation ────────────────────────────────────


def compute_index(
    df: pd.DataFrame,
    elig_col: str,
    dv_col: str,
    index_name: str,
) -> pd.DataFrame:
    """Compute per-person index and Bernoulli plug-in SE.

    For each person i:
        Numerator:   N_i = sum(Y_it - y_hat_rep_it)
        Denominator: D_i = sum(1 - y_hat_rep_it)
        Index:       I_i = N_i / D_i
        SE:          sqrt(sum(p_hat_it * (1 - p_hat_it))) / D_i
    """
    eligible = df[df[elig_col]].copy()

    eligible["_residual"] = eligible[dv_col] - eligible["y_hat_rep"]
    eligible["_gap"] = 1.0 - eligible["y_hat_rep"]

    grouped = eligible.groupby("session_id")

    results = []
    for session_id, group in grouped:
        n_tasks = len(group)
        numerator = group["_residual"].sum()
        denominator = group["_gap"].sum()

        if n_tasks < MIN_ELIGIBLE or denominator <= 0:
            results.append({
                "session_id": session_id,
                f"n_elig_{index_name}": n_tasks,
                f"{index_name}_index": np.nan,
                f"{index_name}_se": np.nan,
            })
            continue

        idx = numerator / denominator

        alpha_i = group["_residual"].mean()
        p_hat = (group["y_hat_rep"] + alpha_i).clip(0.01, 0.99)
        bernoulli_var = (p_hat * (1.0 - p_hat)).sum()
        se = np.sqrt(bernoulli_var) / denominator

        results.append({
            "session_id": session_id,
            f"n_elig_{index_name}": n_tasks,
            f"{index_name}_index": idx,
            f"{index_name}_se": se,
        })

    return pd.DataFrame(results)


def apply_eb_shrinkage(results: pd.DataFrame, index_name: str) -> pd.DataFrame:
    """Apply empirical Bayes shrinkage to a raw index.

    Shrinks noisy estimates toward the grand mean, with shrinkage
    proportional to each person's estimation variance:

        I_i^EB = (1 - B_i) * I_i + B_i * mu
        B_i = SE_i^2 / (sigma^2_theta + SE_i^2)

    where sigma^2_theta = Var(I_i) - Mean(SE_i^2) is the estimated
    variance of true parameters (method of moments).

    References:
        Morris (1983, JASA); Kane & Staiger (2008, NBER WP 14607);
        Chetty, Friedman & Rockoff (2014, AER); Walters (2024, NBER WP 33091).
    """
    idx_col = f"{index_name}_index"
    se_col = f"{index_name}_se"
    eb_col = f"{index_name}_eb"
    shrinkage_col = f"{index_name}_shrinkage"

    results[eb_col] = np.nan
    results[shrinkage_col] = np.nan

    valid = results[idx_col].notna()
    if valid.sum() < 2:
        return results

    raw = results.loc[valid, idx_col].values
    se = results.loc[valid, se_col].values

    # Method of moments: sigma^2_theta = Var(I) - Mean(SE^2)
    var_raw = np.var(raw, ddof=1)
    mean_se_sq = np.mean(se**2)
    sigma2_theta = max(0.0, var_raw - mean_se_sq)

    mu = np.mean(raw)

    # Shrinkage factor B_i: 1 = full shrinkage, 0 = no shrinkage
    B = se**2 / (sigma2_theta + se**2) if sigma2_theta > 0 else np.ones_like(se)

    eb = (1.0 - B) * raw + B * mu

    results.loc[valid, eb_col] = eb
    results.loc[valid, shrinkage_col] = B

    return results


# ── Summary Statistics ───────────────────────────────────────────


def print_index_summary(results: pd.DataFrame, index_name: str, display_name: str) -> None:
    """Print summary statistics for one index."""
    idx_col = f"{index_name}_index"
    se_col = f"{index_name}_se"
    n_col = f"n_elig_{index_name}"

    valid = results[results[idx_col].notna()].copy()
    n_flagged = results[idx_col].isna().sum()

    print(f"\n{'=' * 60}")
    print(f"{display_name.upper()} SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total participants:           {len(results)}")
    print(f"  Valid indices:              {len(valid)}")
    print(f"  Flagged/missing:            {n_flagged}")

    if len(valid) == 0:
        print("\nNo valid indices to summarize.")
        return

    ci = valid[idx_col]
    se = valid[se_col]
    n_tasks = valid[n_col]

    print(f"\n  Index distribution:")
    print(f"    Mean:   {ci.mean():.3f}")
    print(f"    Median: {ci.median():.3f}")
    print(f"    SD:     {ci.std():.3f}")
    print(f"    Min:    {ci.min():.3f}")
    print(f"    Max:    {ci.max():.3f}")
    print(f"    IQR:    [{ci.quantile(0.25):.3f}, {ci.quantile(0.75):.3f}]")

    print(f"\n  Standard errors:")
    print(f"    Mean SE:   {se.mean():.3f}")
    print(f"    Median SE: {se.median():.3f}")
    print(f"    Min SE:    {se.min():.3f}")
    print(f"    Max SE:    {se.max():.3f}")

    print(f"\n  Eligible tasks per person:")
    print(f"    Mean:   {n_tasks.mean():.1f}")
    print(f"    Median: {n_tasks.median():.1f}")
    print(f"    Min:    {n_tasks.min()}")
    print(f"    Max:    {n_tasks.max()}")

    # Reliability
    var_observed = ci.var()
    mean_se_sq = (se**2).mean()
    if var_observed > 0:
        reliability = max(0, 1.0 - mean_se_sq / var_observed)
        print(f"\n  Reliability (rho):")
        print(f"    Var(I_obs):   {var_observed:.4f}")
        print(f"    Mean(SE²):    {mean_se_sq:.4f}")
        print(f"    rho = 1 - Mean(SE²)/Var(I_obs) = {reliability:.3f}")

    sig_frac = ((ci.abs() > 2 * se)).mean()
    print(f"\n  Fraction with |I_i| > 2*SE_i: {sig_frac:.1%}")

    # EB shrinkage summary
    eb_col = f"{index_name}_eb"
    shrinkage_col = f"{index_name}_shrinkage"
    if eb_col in valid.columns and valid[eb_col].notna().any():
        eb = valid[eb_col]
        B = valid[shrinkage_col]
        print(f"\n  Empirical Bayes shrinkage:")
        print(f"    EB index SD:        {eb.std():.3f}  (raw: {ci.std():.3f})")
        print(f"    Mean shrinkage B_i: {B.mean():.3f}  (1 = full shrinkage, 0 = none)")
        print(f"    Median shrinkage:   {B.median():.3f}")
        print(f"    Max shrinkage:      {B.max():.3f}")
        print(f"    Corr(raw, EB):      {ci.corr(eb):.3f}")


# ── Main ─────────────────────────────────────────────────────────

# Index registry: (name, display_name, elig_col, dv_col, matrix_builder)
INDEX_REGISTRY = {
    "consequentialism": (
        "Consequentialism",
        "elig_consequentialism",
        "chose_more_effective",
        build_consequentialism_matrix,
    ),
    "locality": (
        "Locality Preference",
        "elig_locality",
        "chose_local",
        build_locality_matrix,
    ),
}


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-person preference indices from MyGoodness data."
    )
    parser.add_argument("input", help="Path to JSON export from export_study_data()")
    parser.add_argument(
        "--output",
        "-o",
        default="mygoodness_indices.csv",
        help="Output CSV path (default: mygoodness_indices.csv)",
    )
    parser.add_argument(
        "--indices",
        default="all",
        help="Comma-separated list of indices to compute (default: all). "
             "Options: consequentialism, locality",
    )
    parser.add_argument(
        "--model-type",
        choices=["ols", "logit"],
        default="ols",
        help="Choice model: ols (linear probability, default) or logit",
    )
    parser.add_argument(
        "--model-output",
        default=None,
        help="If set, save model summaries to files with this prefix "
             "(e.g., --model-output models produces models_consequentialism.txt, etc.)",
    )
    args = parser.parse_args()

    # Determine which indices to compute
    if args.indices == "all":
        requested = list(INDEX_REGISTRY.keys())
    else:
        requested = [s.strip() for s in args.indices.split(",")]
        for r in requested:
            if r not in INDEX_REGISTRY:
                print(f"Error: unknown index '{r}'. Options: {', '.join(INDEX_REGISTRY.keys())}")
                sys.exit(1)

    # 1. Load data
    sessions, decisions = load_export(args.input)

    # 2. Construct all analysis variables
    df = prepare_decisions(decisions)

    # 3. For each requested index: estimate model, predict, compute index
    all_results = None

    for idx_name in requested:
        display_name, elig_col, dv_col, matrix_builder = INDEX_REGISTRY[idx_name]

        # Estimate
        model, feature_cols = estimate_model(
            df, elig_col, dv_col, matrix_builder,
            model_type=args.model_type, label=display_name,
        )

        if args.model_output:
            model_path = f"{args.model_output}_{idx_name}.txt"
            with open(model_path, "w") as f:
                f.write(model.summary().as_text())
            print(f"  Model summary saved to {model_path}")

        # Predict
        df.loc[df[elig_col], "y_hat_rep"] = predict_model(
            df, elig_col, model, feature_cols, matrix_builder,
        )

        # Compute index + EB shrinkage
        results = compute_index(df, elig_col, dv_col, idx_name)
        results = apply_eb_shrinkage(results, idx_name)

        if all_results is None:
            all_results = results
        else:
            all_results = all_results.merge(results, on="session_id", how="outer")

    # 4. Merge participant_id from sessions
    pid_map = sessions.set_index("id")["participant_id"]
    all_results["participant_id"] = all_results["session_id"].map(pid_map)

    # Reorder: participant_id, session_id, then all index columns
    id_cols = ["participant_id", "session_id"]
    other_cols = [c for c in all_results.columns if c not in id_cols]
    all_results = all_results[id_cols + sorted(other_cols)]

    # 5. Print summaries
    for idx_name in requested:
        display_name = INDEX_REGISTRY[idx_name][0]
        print_index_summary(all_results, idx_name, display_name)

    # Cross-index correlation (when both computed)
    if len(requested) > 1:
        idx_cols = [f"{r}_index" for r in requested]
        valid = all_results.dropna(subset=idx_cols)
        if len(valid) > 10:
            print(f"\n{'=' * 60}")
            print("CROSS-INDEX CORRELATIONS")
            print(f"{'=' * 60}")
            print(f"  (N = {len(valid)} participants with all indices)")
            corr = valid[idx_cols].corr()
            for i, r1 in enumerate(idx_cols):
                for r2 in idx_cols[i + 1:]:
                    name1 = r1.replace("_index", "")
                    name2 = r2.replace("_index", "")
                    print(f"  {name1} x {name2}: r = {corr.loc[r1, r2]:.3f}")

    # 6. Save
    all_results.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
