#!/usr/bin/env python3
"""
Consequentialism Index Calculator

Computes per-person Relative Consequentialism Index and standard errors
from MyGoodness study data.

Usage:
    python consequentialism_index.py export.json [--output results.csv]

Input:  JSON file from the export_study_data() RPC function.
Output: CSV with per-person consequentialism index, SE, and metadata.

Methodology
-----------
1. Estimate a "representative agent" model: pooled OLS of binary choice
   (chose-more-effective = 1) on efficiency-difference bins x location
   interactions, plus controls. Estimation sample: stranger-stranger
   decisions with nothing hidden. SEs clustered by individual.

2. For each person's eligible decisions, predict the representative
   agent's choice probability (y_hat_rep).

3. Define the maximal agent: always chooses the more effective option
   (y_hat_max = 1 when efficiency differs).

4. Consequentialism index:
       C_i = sum(Y_it - y_hat_rep_it) / sum(y_hat_max_it - y_hat_rep_it)
   where the sums run over person i's eligible decisions.

5. Per-person SE (Bernoulli plug-in):
       SE(C_i) = sqrt(sum(p_hat_it * (1 - p_hat_it))) / D_i
   where p_hat_it = clip(y_hat_rep_it + alpha_i, 0.01, 0.99),
   alpha_i is person i's mean residual, and D_i is the denominator.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ── Configuration ────────────────────────────────────────────────

# Efficiency difference bins (non-overlapping, inclusive).
# Individual dummies for 1-15 (smooth curve where most data lives),
# then wider bins for 16+.
# Reference category: diff = 0 (equal efficiency, omitted from dummies).
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

# Location interaction structure (reference: both_far):
#   Diffs 1-12:  interact with both_near and one_far
#   Diffs 13-15: interact with one_far only (both_near too sparse)
#   Diffs 16+:   one_far main effect only (no bin-level interactions)


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


def prepare_decisions(decisions: pd.DataFrame) -> pd.DataFrame:
    """Add analysis variables to the decisions DataFrame."""
    df = decisions.copy()

    # Drop decisions without a choice
    n_before = len(df)
    df = df[df["choice"].notna()].copy()
    print(f"  Dropped {n_before - len(df)} unanswered decisions")

    # Ensure hidden fields are lists
    for col in ["left_hidden", "right_hidden"]:
        df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])

    # ── Dependent variable ──
    # Which side is more effective?
    df["more_eff_side"] = np.where(
        df["left_count"] > df["right_count"],
        "left",
        np.where(df["right_count"] > df["left_count"], "right", "equal"),
    )
    # Did they choose the more effective option?
    df["chose_more_effective"] = np.where(
        df["more_eff_side"] == "equal",
        np.nan,
        (df["choice"] == df["more_eff_side"]).astype(float),
    )

    # ── Efficiency difference (unsigned) ──
    df["eff_diff"] = (df["left_count"] - df["right_count"]).abs()
    df["eff_bin"] = df["eff_diff"].apply(assign_eff_bin)

    # ── Location categories ──
    df["both_near"] = df["left_where_near"] & df["right_where_near"]
    df["both_far"] = ~df["left_where_near"] & ~df["right_where_near"]
    df["one_far"] = ~df["both_near"] & ~df["both_far"]
    df["location"] = np.where(
        df["both_near"], "both_near", np.where(df["one_far"], "one_far", "both_far")
    )

    # (Location interactions are built in the design matrix, not here.)

    # ── Nothing hidden ──
    df["nothing_hidden"] = (df["left_hidden"].apply(len) == 0) & (
        df["right_hidden"].apply(len) == 0
    )

    # ── Stranger-stranger (no kin, no self) ──
    df["is_stranger_stranger"] = (
        ~df["left_is_kin"]
        & ~df["right_is_kin"]
        & ~df["left_who_self"]
        & ~df["right_who_self"]
    )

    # ── Controls: identifiable victim and named charity ──
    # Oriented relative to the more-effective option
    for attr, col in [("victim", "who_name"), ("charity", "what_charity_name")]:
        left_has = df[f"left_{col}"].notna()
        right_has = df[f"right_{col}"].notna()

        df[f"{attr}_more_eff"] = np.where(
            df["more_eff_side"] == "left",
            left_has,
            np.where(df["more_eff_side"] == "right", right_has, False),
        ).astype(int)

        df[f"{attr}_less_eff"] = np.where(
            df["more_eff_side"] == "left",
            right_has,
            np.where(df["more_eff_side"] == "right", left_has, False),
        ).astype(int)

    # ── Controls: cause, gender, age on each side ──
    # Oriented relative to the more-effective option (no interactions needed)
    for side_label, side_col in [("more_eff", "more_eff_side"), ("less_eff", None)]:
        # Determine which raw column (left/right) corresponds to this side
        if side_label == "more_eff":
            cause = np.where(
                df["more_eff_side"] == "left", df["left_what_cause"], df["right_what_cause"]
            )
            gender = np.where(
                df["more_eff_side"] == "left", df["left_who_gender"], df["right_who_gender"]
            )
            age = np.where(
                df["more_eff_side"] == "left", df["left_who_age_group"], df["right_who_age_group"]
            )
        else:
            cause = np.where(
                df["more_eff_side"] == "left", df["right_what_cause"], df["left_what_cause"]
            )
            gender = np.where(
                df["more_eff_side"] == "left", df["right_who_gender"], df["left_who_gender"]
            )
            age = np.where(
                df["more_eff_side"] == "left", df["right_who_age_group"], df["left_who_age_group"]
            )
        df[f"cause_{side_label}"] = cause
        df[f"gender_{side_label}"] = gender
        df[f"age_{side_label}"] = age

    # ── Eligible for analysis ──
    # Stranger-stranger, nothing hidden, choice made, efficiency differs
    df["eligible"] = (
        df["is_stranger_stranger"]
        & df["nothing_hidden"]
        & df["chose_more_effective"].notna()
        & (df["eff_diff"] > 0)
    )

    n_eligible = df["eligible"].sum()
    n_people = df.loc[df["eligible"], "session_id"].nunique()
    print(f"  {n_eligible} eligible decisions from {n_people} participants")

    return df


# ── Representative Agent Model ───────────────────────────────────


def build_design_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Build the design matrix for the representative agent model.

    Structure (reference category: both_far for location, diff=0 for bins):

    1. Efficiency bin dummies (all bins, main effects)
    2. Bin x both_near interactions   — diffs 1-12 only
    3. Bin x one_far interactions     — diffs 1-15 only
    4. one_far main effect            — diffs 16+ only
    5. Controls: victim, charity, cause, gender, age (main effects)
    """
    parts = []

    # ── 1. Efficiency bin main effects ──
    bin_dummies = pd.get_dummies(data["eff_bin"], prefix="diff")
    # Drop reference (diff=0) if present
    bin_dummies = bin_dummies.drop(columns=["diff_0"], errors="ignore")
    parts.append(bin_dummies)

    # ── 2. Bin x both_near interactions (diffs 1-12) ──
    for d in range(1, 13):
        col = f"diff_{d}:both_near"
        data[col] = ((data["eff_diff"] == d) & data["both_near"]).astype(int)
        parts.append(data[[col]])

    # ── 3. Bin x one_far interactions (all bins) ──
    for d in range(1, 16):
        col = f"diff_{d}:one_far"
        data[col] = ((data["eff_diff"] == d) & data["one_far"]).astype(int)
        parts.append(data[[col]])

    for lo, hi in EFFICIENCY_BINS:
        if lo < 16:
            continue  # already handled above as individual dummies
        label = bin_label(lo, hi)
        col = f"diff_{label}:one_far"
        data[col] = ((data["eff_diff"] >= lo) & (data["eff_diff"] <= hi) & data["one_far"]).astype(int)
        parts.append(data[[col]])

    # ── 5. Controls ──
    parts.append(
        data[["victim_more_eff", "victim_less_eff",
              "charity_more_eff", "charity_less_eff"]].astype(float)
    )

    for attr in ["cause", "gender", "age"]:
        for side in ["more_eff", "less_eff"]:
            col = f"{attr}_{side}"
            dummies = pd.get_dummies(data[col], prefix=f"{attr}_{side}", drop_first=True)
            parts.append(dummies)

    X = pd.concat(parts, axis=1)

    # Drop columns with zero variance (empty cells)
    X = X.loc[:, X.std() > 0]

    X = sm.add_constant(X)
    return X


def estimate_rep_model(
    df: pd.DataFrame,
    model_type: str = "ols",
) -> tuple[sm.regression.linear_model.RegressionResultsWrapper, list[str]]:
    """Estimate pooled choice model on eligible decisions with clustered SEs.

    Parameters
    ----------
    model_type : "ols" or "logit"
        OLS = linear probability model (faster, used by power simulations).
        Logit = logistic regression (bounded predictions, no clipping needed).
    """
    est = df[df["eligible"]].copy()

    X = build_design_matrix(est)
    y = est["chose_more_effective"].astype(float)

    cluster_kwds = {"groups": est["session_id"].values}

    if model_type == "logit":
        model = sm.Logit(y, X).fit(
            cov_type="cluster", cov_kwds=cluster_kwds, disp=False,
        )
        print(f"\nRepresentative agent model (logit):")
        print(f"  N = {model.nobs:.0f} decisions, {len(X.columns)} parameters")
        print(f"  Pseudo R² = {model.prsquared:.3f}")
    else:
        model = sm.OLS(y, X).fit(
            cov_type="cluster", cov_kwds=cluster_kwds,
        )
        print(f"\nRepresentative agent model (OLS):")
        print(f"  N = {model.nobs:.0f} decisions, {len(X.columns)} parameters, R² = {model.rsquared:.3f}")

    print(f"  Intercept = {model.params['const']:.3f}")

    return model, X.columns.tolist()


def predict_rep(
    df: pd.DataFrame,
    model: sm.regression.linear_model.RegressionResultsWrapper,
    feature_cols: list[str],
) -> pd.Series:
    """Generate representative-agent predictions for all eligible decisions."""
    eligible = df[df["eligible"]].copy()

    X = build_design_matrix(eligible)

    # Align columns: add any missing (new cells), drop any extra
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feature_cols]

    preds = model.predict(X)
    # Clip to [0.01, 0.99] to avoid degenerate probabilities
    preds = preds.clip(0.01, 0.99)

    return pd.Series(preds.values, index=eligible.index, name="y_hat_rep")


# ── Consequentialism Index ───────────────────────────────────────


def compute_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-person Consequentialism Index and Bernoulli plug-in SE.

    For each person i:
        Numerator:   N_i = sum(Y_it - y_hat_rep_it)
        Denominator: D_i = sum(1 - y_hat_rep_it)      [since y_hat_max = 1]
        Index:       C_i = N_i / D_i
        SE:          SE_i = sqrt(sum(p_hat_it * (1 - p_hat_it))) / D_i
    where p_hat_it = clip(y_hat_rep + mean_residual_i, 0.01, 0.99)
    """
    eligible = df[df["eligible"]].copy()

    # Per-decision quantities
    eligible["residual"] = eligible["chose_more_effective"] - eligible["y_hat_rep"]
    eligible["gap"] = 1.0 - eligible["y_hat_rep"]  # y_hat_max - y_hat_rep

    # Group by person (session)
    grouped = eligible.groupby("session_id")

    results = []
    for session_id, group in grouped:
        n_tasks = len(group)
        numerator = group["residual"].sum()
        denominator = group["gap"].sum()

        if denominator <= 0:
            # All predictions are ~1 (everyone would choose efficient).
            # Index is undefined.
            results.append(
                {
                    "session_id": session_id,
                    "n_eligible_tasks": n_tasks,
                    "consequentialism_index": np.nan,
                    "se": np.nan,
                    "denominator": denominator,
                    "flag": "denominator_zero",
                }
            )
            continue

        index = numerator / denominator

        # Bernoulli plug-in SE with person-level adjustment
        alpha_i = group["residual"].mean()  # person-level mean residual
        p_hat = (group["y_hat_rep"] + alpha_i).clip(0.01, 0.99)
        bernoulli_var = (p_hat * (1.0 - p_hat)).sum()
        se = np.sqrt(bernoulli_var) / denominator

        results.append(
            {
                "session_id": session_id,
                "n_eligible_tasks": n_tasks,
                "consequentialism_index": index,
                "se": se,
                "denominator": denominator,
                "flag": "",
            }
        )

    return pd.DataFrame(results)


# ── Summary Statistics ───────────────────────────────────────────


def print_summary(results: pd.DataFrame, sessions: pd.DataFrame) -> None:
    """Print summary statistics for the consequentialism index."""
    # Merge participant IDs
    valid = results[results["consequentialism_index"].notna()].copy()
    flagged = results[results["consequentialism_index"].isna()]

    print(f"\n{'=' * 60}")
    print("CONSEQUENTIALISM INDEX SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total participants:           {len(results)}")
    print(f"  Valid indices:              {len(valid)}")
    print(f"  Flagged (no index):         {len(flagged)}")
    if len(flagged) > 0:
        for flag, count in flagged["flag"].value_counts().items():
            print(f"    {flag}: {count}")

    if len(valid) == 0:
        print("\nNo valid indices to summarize.")
        return

    ci = valid["consequentialism_index"]
    se = valid["se"]
    n_tasks = valid["n_eligible_tasks"]

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

    # Reliability estimate
    var_observed = ci.var()
    mean_se_sq = (se**2).mean()
    if var_observed > 0:
        reliability = max(0, 1.0 - mean_se_sq / var_observed)
        print(f"\n  Reliability (rho):")
        print(f"    Var(C_obs):   {var_observed:.4f}")
        print(f"    Mean(SE²):    {mean_se_sq:.4f}")
        print(f"    rho = 1 - Mean(SE²)/Var(C_obs) = {reliability:.3f}")
        print(f"    (rho = 1 means no measurement error; rho = 0 means all noise)")

    # Fraction of individuals where |C_i| > 2*SE_i
    sig_frac = ((ci.abs() > 2 * se)).mean()
    print(f"\n  Fraction with |C_i| > 2*SE_i: {sig_frac:.1%}")


# ── Main ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-person Consequentialism Index from MyGoodness data."
    )
    parser.add_argument("input", help="Path to JSON export from export_study_data()")
    parser.add_argument(
        "--output",
        "-o",
        default="consequentialism_index.csv",
        help="Output CSV path (default: consequentialism_index.csv)",
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
        help="If set, save representative agent model summary to this path",
    )
    args = parser.parse_args()

    # 1. Load data
    sessions, decisions = load_export(args.input)

    # 2. Construct analysis variables
    df = prepare_decisions(decisions)

    # 3. Estimate representative agent model
    model, feature_cols = estimate_rep_model(df, model_type=args.model_type)

    if args.model_output:
        with open(args.model_output, "w") as f:
            f.write(model.summary().as_text())
        print(f"  Model summary saved to {args.model_output}")

    # 4. Generate predictions
    df.loc[df["eligible"], "y_hat_rep"] = predict_rep(df, model, feature_cols)

    # 5. Compute per-person index and SE
    results = compute_indices(df)

    # 6. Merge participant_id from sessions
    pid_map = sessions.set_index("id")["participant_id"]
    results["participant_id"] = results["session_id"].map(pid_map)

    # Reorder columns
    results = results[
        [
            "participant_id",
            "session_id",
            "n_eligible_tasks",
            "consequentialism_index",
            "se",
            "denominator",
            "flag",
        ]
    ]

    # 7. Print summary
    print_summary(results, sessions)

    # 8. Save output
    results.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
