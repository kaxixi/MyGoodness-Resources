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

Computes a per-person **Relative Consequentialism Index** measuring how much more (or less) each participant responds to efficiency compared to the average participant.

### Quick start

```bash
python consequentialism_index.py export.json
```

This produces `consequentialism_index.csv` and prints summary statistics to the console.

### Usage

```
python consequentialism_index.py export.json [OPTIONS]

Options:
  -o, --output PATH        Output CSV path (default: consequentialism_index.csv)
  --model-type {ols,logit}  Choice model for the representative agent (default: ols)
  --model-output PATH      Save model coefficient table to this path
```

### Examples

```bash
# OLS (default) — fast, recommended for most uses
python consequentialism_index.py export.json -o results.csv

# Logit — bounded predictions, no clipping needed
python consequentialism_index.py export.json --model-type logit -o results.csv

# Save the representative agent model coefficients for inspection
python consequentialism_index.py export.json --model-output rep_model.txt
```

### Output CSV columns

| Column | Description |
|---|---|
| `participant_id` | External participant ID (from your survey platform) |
| `session_id` | MyGoodness session UUID |
| `n_eligible_tasks` | Number of tasks used to compute the index |
| `consequentialism_index` | Relative Consequentialism Index (see below) |
| `se` | Standard error of the index |
| `denominator` | Denominator of the index (sum of max - rep gaps) |
| `flag` | Non-empty if the index could not be computed |

### Methodology

#### 1. Estimation sample

The representative agent model is estimated on **stranger-stranger decisions with nothing hidden** and where the two options differ in efficiency. This excludes:

- Kin decisions (either card involves a relative)
- Self decisions (either card is a reward to self)
- Decisions where any attribute was initially hidden ("Click to reveal")
- Decisions where both options save the same number of people

#### 2. Representative agent model

A pooled choice model (OLS or logit) where the dependent variable is 1 if the participant chose the more effective option:

**Efficiency bins:** Individual dummies for differences 1 through 15, then bins for 16-20, 21-30, 31-40, 41-50, 51-75, 76-100, 101-150, 151-200, 201-250, 251-300.

**Location interactions** (reference: both far):
- Differences 1-12: interacted with both_near and one_far
- Differences 13-15 and all higher bins: interacted with one_far only

Both-near interactions stop at 12 because near cards have efficacy 1-20, making both-near differences above ~12 very rare.

**Controls** (main effects, no interactions):
- Identifiable victim on the more/less effective side
- Named charity on the more/less effective side
- Cause (domain) on the more/less effective side
- Gender on the more/less effective side
- Age group on the more/less effective side

Standard errors are clustered at the individual (session) level.

#### 3. Consequentialism index

For each participant *i* with eligible tasks *t*:

```
              sum(Y_it - Y_hat_rep_it)
    C_i  =  --------------------------
             sum(Y_hat_max - Y_hat_rep)
```

Where:
- `Y_it` = 1 if participant chose the more effective option
- `Y_hat_rep` = predicted probability from the representative agent model
- `Y_hat_max` = 1 (the maximal agent always chooses the more effective option)

**Interpretation:**
- `C_i = 0`: behaves like the average participant
- `C_i > 0`: more efficiency-sensitive than average
- `C_i < 0`: less efficiency-sensitive than average
- `C_i = 1`: always chose the more effective option (maximal agent)

Values can fall outside [-1, 1] due to noise.

#### 4. Standard errors

Per-person SE using a Bernoulli plug-in with person-level adjustment:

```
                sqrt(sum(p_hat_it * (1 - p_hat_it)))
    SE(C_i) =  ------------------------------------
                            D_i
```

Where `p_hat_it = clip(Y_hat_rep + alpha_i, 0.01, 0.99)` and `alpha_i` is participant *i*'s mean residual from the representative agent model. This adapts the variance estimate to each person's deviation from the average.

#### 5. Reliability

The summary output includes a reliability estimate:

```
    rho = 1 - Mean(SE_i^2) / Var(C_i)
```

- `rho = 1`: the index is measured without error
- `rho = 0`: the index is pure noise
- Typical values with ~8-10 eligible tasks per person: 0.3-0.6

Reliability is the key input for power analysis: measurement error in the index inflates residual variance when regressing the index on external measures, reducing statistical power.

### OLS vs. logit

| | OLS (linear probability) | Logit |
|---|---|---|
| Predictions | Can exceed [0, 1]; clipped to [0.01, 0.99] | Naturally bounded |
| Speed | Fast | Slower (iterative) |
| Use in simulations | Recommended (faster Monte Carlo) | Not recommended |
| Coefficient interpretation | Marginal probability | Log-odds (less intuitive) |

Both produce valid indices. OLS is the default; logit is available via `--model-type logit` for researchers who prefer bounded predictions.
