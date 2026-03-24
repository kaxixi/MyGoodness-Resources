# MyGoodness Analysis Code

Python scripts for analyzing data from MyGoodness studies.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+.

## Data Export

Before running any analysis, export your study data:

```bash
curl -X POST 'https://ipjbjszcshiebjmtmhwr.supabase.co/rest/v1/rpc/export_study_data' \
  -H 'apikey: YOUR_SUPABASE_ANON_KEY' \
  -H 'Content-Type: application/json' \
  -d '{"p_study_slug": "your-study-slug", "p_export_key": "your-export-key"}' \
  -o export.json
```

This produces a JSON file with `sessions`, `decisions`, and `survey_responses` arrays. See `RESEARCHER_GUIDE.md` for details.

---

## consequentialism_index.py

Computes two per-person preference indices from MyGoodness data:

1. **Relative Consequentialism Index** — how much more (or less) each participant responds to efficiency compared to the average participant
2. **Locality Preference Index** — how much more (or less) each participant prefers local charities compared to the average participant

Each index is estimated from its own representative agent model, using the same general framework.

### Quick start

```bash
# Compute both indices (default)
python consequentialism_index.py export.json

# Compute only one index
python consequentialism_index.py export.json --indices consequentialism
python consequentialism_index.py export.json --indices locality
```

This produces `mygoodness_indices.csv` and prints summary statistics to the console.

### Usage

```
python consequentialism_index.py export.json [OPTIONS]

Options:
  -o, --output PATH           Output CSV path (default: mygoodness_indices.csv)
  --indices LIST              Comma-separated indices to compute (default: all)
                              Options: consequentialism, locality
  --model-type {ols,logit}    Choice model (default: ols)
  --model-output PREFIX       Save model summaries to PREFIX_consequentialism.txt, etc.
```

### Examples

```bash
# Both indices, OLS (default)
python consequentialism_index.py export.json -o results.csv

# Only consequentialism, logit model
python consequentialism_index.py export.json --indices consequentialism --model-type logit

# Save model coefficient tables for inspection
python consequentialism_index.py export.json --model-output models
# produces: models_consequentialism.txt, models_locality.txt
```

### Output CSV columns

One row per participant. Columns present depend on `--indices` flag.

| Column | Description |
|---|---|
| `participant_id` | External participant ID (from your survey platform) |
| `session_id` | MyGoodness session UUID |
| `n_elig_consequentialism` | Number of tasks used for the consequentialism index |
| `consequentialism_index` | Relative Consequentialism Index |
| `consequentialism_se` | Standard error of the consequentialism index |
| `consequentialism_eb` | EB-shrunk consequentialism index (recommended for downstream analysis) |
| `consequentialism_shrinkage` | Shrinkage factor B_i (1 = fully shrunk to mean, 0 = no shrinkage) |
| `n_elig_locality` | Number of tasks used for the locality index |
| `locality_index` | Locality Preference Index (raw) |
| `locality_se` | Standard error of the locality index |
| `locality_eb` | EB-shrunk locality index (recommended for downstream analysis) |
| `locality_shrinkage` | Shrinkage factor B_i |

---

## Shared Methodology

Both indices follow the same framework. A separate representative agent model is estimated for each index, then the index measures each participant's deviation from the model's predictions.

### Estimation sample

Both models are estimated on **stranger-stranger decisions with nothing hidden**:

- Excludes kin decisions (either card involves a relative)
- Excludes self decisions (either card is a reward to self)
- Excludes decisions where any attribute was initially hidden ("Click to reveal")

Each index further restricts to decisions where the relevant attribute varies (see below).

### Index formula

For each participant *i* with eligible tasks *t*:

```
              sum(Y_it - Y_hat_rep_it)
    I_i  =  --------------------------
              sum(1 - Y_hat_rep_it)
```

Where:
- `Y_it` = 1 if participant chose the preferred-direction option (e.g., more effective, or local)
- `Y_hat_rep` = predicted probability from the representative agent model
- The maximal agent always chooses the preferred-direction option (`Y_hat_max = 1`)

**Interpretation:**
- `I_i = 0`: behaves like the average participant
- `I_i > 0`: stronger preference than average (more consequentialist, or more local-preferring)
- `I_i < 0`: weaker preference than average
- `I_i = 1`: always chose the preferred-direction option

Values can fall outside [-1, 1] due to noise.

### Standard errors

Per-person SE using a Bernoulli plug-in with person-level adjustment:

```
                sqrt(sum(p_hat_it * (1 - p_hat_it)))
    SE(I_i) =  ------------------------------------
                            D_i
```

Where `p_hat_it = clip(Y_hat_rep + alpha_i, 0.01, 0.99)` and `alpha_i` is participant *i*'s mean residual from the representative agent model.

### Reliability

Each index's summary includes a reliability estimate:

```
    rho = 1 - Mean(SE_i^2) / Var(I_i)
```

- `rho = 1`: measured without error
- `rho = 0`: pure noise
- Key input for power analysis: measurement error inflates residual variance in downstream regressions

### Empirical Bayes shrinkage

Raw indices can be extreme when a participant has few eligible tasks (small denominator). To address this, each index is accompanied by an **empirical Bayes (EB) shrunk** version that pulls imprecise estimates toward the grand mean:

```
    I_i^EB = (1 - B_i) * I_i + B_i * mu
```

Where:
- `mu` = grand mean of the raw index
- `B_i = SE_i^2 / (sigma^2_theta + SE_i^2)` = shrinkage factor
- `sigma^2_theta = Var(I_i) - Mean(SE_i^2)` = estimated variance of true parameters (method of moments)

Participants with large SEs (few tasks, small denominators) are heavily shrunk toward the mean. Participants with precise estimates are barely affected. The shrinkage amount is data-driven, not arbitrary.

**The EB index (`*_eb`) is recommended for downstream regressions.** The raw index (`*_index`) is provided for transparency and robustness checks.

References: Morris (1983, JASA); Kane & Staiger (2008); Chetty, Friedman & Rockoff (2014, AER); Walters (2024, NBER WP 33091).

When both indices are computed, cross-index correlations are also reported.

---

## Index-Specific Details

### Consequentialism Index

**DV:** `chose_more_effective` (1 if participant chose the option saving more lives)

**Additional eligibility:** efficiency must differ between the two options (`eff_diff > 0`)

**Representative agent model:**

- *Efficiency bins:* Individual dummies for differences 1-15, then wider bins (16-20, 21-30, ..., 251-300). Reference: diff = 0.
- *Location interactions* (reference: both far):
  - Diffs 1-12: interacted with both_near and one_far
  - Diffs 13+: interacted with one_far only
- *Controls* (main effects): identifiable victim, named charity, cause, gender, age — each oriented to the more-effective / less-effective side

**Expected eligible tasks per person:** ~8-10 (of 12 total decisions)

### Locality Preference Index

**DV:** `chose_local` (1 if participant chose the local/near option)

**Additional eligibility:** exactly one card must be near and one far

**Representative agent model:**

- *Efficiency control:* cubic polynomial of `local_count - far_count` (signed: positive means the local option saves more lives). Captures nonlinear tradeoff between efficiency and locality preference.
- *Controls* (main effects): identifiable victim, named charity, cause, gender, age — each oriented to the local / far side

**Expected eligible tasks per person:** ~3-5 (depends on location variation in the session)

**Interpretation:** positive index = prefers local more than average; negative = prefers far/efficient more than average.

---

## OLS vs. Logit

| | OLS (linear probability) | Logit |
|---|---|---|
| Predictions | Can exceed [0, 1]; clipped to [0.01, 0.99] | Naturally bounded |
| Speed | Fast | Slower (iterative) |
| Use in simulations | Recommended (faster Monte Carlo) | Not recommended |
| Coefficient interpretation | Marginal probability | Log-odds (less intuitive) |

Both produce valid indices. OLS is the default; logit is available via `--model-type logit`.
