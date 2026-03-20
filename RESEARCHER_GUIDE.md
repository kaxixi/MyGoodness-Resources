# Researcher Guide: Embedding MyGoodness in Your Study

This guide explains how to set up the MyGoodness charitable giving game as an embedded component in your survey platform (e.g., Qualtrics, Prolific, Gorilla).

---

## Quick Start

The game is live at **https://my-goodness.vercel.app**

- **Standalone mode** (no parameters): Visit the URL directly to play the game
- **Embed mode**: Add `?study=YOUR_SLUG&pid=PARTICIPANT_ID&return_url=REDIRECT_URL` to embed in a survey

A ready-to-import **Qualtrics test survey** is included at `studies/mygoodness-test.qsf` — see [Section 3](#3-embedding-in-qualtrics) for details.

---

## Overview

The multi-study embed system lets you:

- **Create isolated studies** with their own game configuration
- **Embed the game** inside your survey via an iframe
- **Track participants** using external IDs from your survey platform
- **Export data** for your study only, using a secret export key
- **Customize the game** — consent text, number of rounds, mobile layout, skip consent/survey, etc.

Each study is identified by a URL slug (e.g., `cint-pilot-2026`) and has its own configuration stored in the database.

---

## 1. Creating a Study

### Write a YAML config file

Create a YAML file in the `studies/` directory. See `studies/example.yaml` for a template.

```yaml
_slug: "my-study-2026"
_name: "My Study (Spring 2026)"

# Override any default game settings here.
# Only include fields you want to change from defaults.

totalDecisions: 10
consentTiming: "before"

# Skip the built-in consent/survey screens when your survey platform
# already handles these:
# showConsent: false
# showSurvey: false

consentText:
  heading: "Research Consent"
  sections:
    - title: "1. Purpose"
      body: "This study examines charitable giving decisions."
```

**Required metadata fields** (prefixed with `_`):
- `_slug` — URL-friendly identifier (lowercase, hyphens, no spaces)
- `_name` — Human-readable study name

All other fields override the defaults in `gameConfig.yaml`. See that file for the full list of configurable options.

### Run the create-study script

```bash
# Set environment variables
export SUPABASE_URL="https://ipjbjszcshiebjmtmhwr.supabase.co"
export SUPABASE_SERVICE_ROLE_KEY="eyJ..."  # from Supabase → Settings → API → service_role (secret)

# Optionally set your app URL for the embed snippet
export APP_URL="https://my-goodness.vercel.app"

# Create the study
npx tsx scripts/create-study.ts studies/my-study.yaml
```

The script will output:
- The study **slug** and **ID**
- A secret **export key** (save this — it cannot be retrieved later)
- An **embed snippet** ready to paste into your survey
- A **data export** curl command

---

## 2. URL Parameters

When embedding the game, pass these URL parameters:

| Parameter    | Required | Description |
|---|---|---|
| `study`      | Yes      | The study slug (e.g., `my-study-2026`) |
| `pid`        | No       | Participant ID from your survey platform |
| `return_url` | No       | URL to redirect to after the game finishes |

Example URL:
```
https://my-goodness.vercel.app/?study=my-study-2026&pid=R_abc123&return_url=https://survey.qualtrics.com/next
```

**What happens in embed mode:**
- The game loads the study's custom configuration from the database
- `participant_id` and `return_url` are saved on the session row
- The results page shows **"Return to Survey"** instead of "Redo Game"
- A `postMessage` is sent to the parent window when the game completes (see below)
- The "Others" regression model only uses data from the same study

---

## 3. Embedding in Qualtrics

### Option A: Import the test survey (recommended for testing)

A ready-made Qualtrics survey is included at `studies/mygoodness-test.qsf`. To use it:

1. Log in to Qualtrics
2. Click **Create Project** → **Survey** → **Import a QSF file**
3. Upload `studies/mygoodness-test.qsf`
4. The survey has three pages: intro, game (iframe), and thank-you
5. Preview or publish the survey to test the full flow

The test survey uses the `example-study` study slug. Replace it with your own study's slug for production use.

### Option B: Manual embed setup

In Qualtrics, add a **"Text/Graphic"** question and switch to the **HTML view**. Paste:

```html
<iframe
  id="mygoodness-frame"
  src="https://my-goodness.vercel.app/?study=my-study-2026&pid=${e://Field/ResponseID}&return_url=${e://Field/NextURL}"
  width="100%"
  height="800"
  style="border: none;"
  allow="clipboard-write"
></iframe>
```

### Listening for game completion

To auto-advance when the game finishes, add this JavaScript to the question (click the question → JavaScript editor, or add to the question's "addOnReady" handler):

```javascript
Qualtrics.SurveyEngine.addOnReady(function() {
  window.addEventListener('message', function(event) {
    if (event.data && event.data.type === 'mygoodness:complete') {
      // Store the MyGoodness session ID in an embedded data field
      Qualtrics.SurveyEngine.setEmbeddedData(
        'mygoodness_session_id',
        event.data.sessionId
      );
      // Auto-advance to the next page
      document.querySelector('#NextButton').click();
    }
  });
});
```

**Important:** For the embedded data field to work, you must first define `mygoodness_session_id` in your survey flow:
1. Go to **Survey Flow**
2. Add an **Embedded Data** element at the top of the flow
3. Add a field named `mygoodness_session_id`

### Passing participant ID

Qualtrics provides `${e://Field/ResponseID}` as a unique participant identifier. This gets recorded as `participant_id` in the sessions table, allowing you to join MyGoodness data with your Qualtrics responses.

### Hiding the Next button

If you want the game page to only advance via `postMessage` (not a visible Next button), add this CSS to the question:

```html
<style>#NextButton { display: none; }</style>
```

---

## 4. Data Export

Export all data for your study using the export RPC function:

### Using curl

```bash
curl -X POST 'https://ipjbjszcshiebjmtmhwr.supabase.co/rest/v1/rpc/export_study_data' \
  -H 'apikey: YOUR_SUPABASE_ANON_KEY' \
  -H 'Content-Type: application/json' \
  -d '{"p_study_slug": "my-study-2026", "p_export_key": "YOUR_EXPORT_KEY"}'
```

### Using JavaScript/fetch

```js
const response = await fetch(
  'https://ipjbjszcshiebjmtmhwr.supabase.co/rest/v1/rpc/export_study_data',
  {
    method: 'POST',
    headers: {
      'apikey': 'YOUR_SUPABASE_ANON_KEY',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      p_study_slug: 'my-study-2026',
      p_export_key: 'YOUR_EXPORT_KEY',
    }),
  },
);
const data = await response.json();
// data.sessions — array of session rows
// data.decisions — array of decision rows (10 per session)
// data.survey_responses — array of survey response rows
```

### Response format

The export returns a JSON object with three arrays:

```json
{
  "sessions": [
    {
      "id": "uuid",
      "participant_id": "R_abc123",
      "created_at": "2026-...",
      "ended_at": "2026-...",
      "consent": true,
      "visitor_device_type": "desktop",
      "..."
    }
  ],
  "decisions": [
    {
      "session_id": "uuid",
      "round_number": 0,
      "choice": "left",
      "left_count": 5,
      "right_count": 12,
      "left_what_cause": "water",
      "right_what_cause": "nutrition",
      "left_hidden": ["count"],
      "left_revealed": [{"field": "count", "at": "2026-..."}],
      "..."
    }
  ],
  "survey_responses": [
    {
      "session_id": "uuid",
      "views_political": 0.5,
      "demographic_age": 30,
      "..."
    }
  ]
}
```

### Security

- The export key is a secret — treat it like a password
- The `export_study_data` RPC only returns data for sessions linked to your study
- Data from other studies (or standalone sessions) is never included
- A wrong export key returns an error, not empty data
- **Consent filtering**: sessions where the participant declined consent (`consent = false`) are excluded from exports and from the "Others" regression model. The data is retained in the database for audit purposes but never included in research-facing queries.

---

## 5. Configuration Reference

All fields from `app/src/data/gameConfig.yaml` can be overridden in your study YAML. Key options:

| Field | Default | Description |
|---|---|---|
| `totalDecisions` | `10` | Number of rounds per game |
| `showConsent` | `true` | Show the consent screen? Set `false` to skip entirely |
| `showSurvey` | `true` | Show the pre-results survey? Set `false` to skip entirely |
| `consentTiming` | `"after"` | When to show consent: `"before"` or `"after"` the game (ignored if `showConsent: false`) |
| `mobileLayout` | `"side-by-side"` | Mobile card layout: `"side-by-side"` or `"stacked"` |
| `resultsModelN` | `0` | Number of past users for "Others" model (`0` = all) |
| `consentText` | *(see yaml)* | Consent screen heading and body sections |
| `modifications.*` | *(see yaml)* | Min/max counts for each modification type |
| `efficacy.brackets` | *(see yaml)* | Efficacy sampling brackets and weights |
| `charityNames` | *(see yaml)* | Named charities by cause (famous/unknown pools) |
| `victimNames` | *(see yaml)* | Identifiable victim names by region (male/female pools) |

### Game flow with different settings

| Settings | Flow |
|---|---|
| All defaults | game → consent → survey → results |
| `consentTiming: "before"` | consent → game → survey → results |
| `showConsent: false` | game → survey → results |
| `showSurvey: false` | game → consent → results |
| Both `false` | game → results |

See `gameConfig.yaml` for the full configuration with comments.

---

## 6. Deactivating a Study

To stop accepting new participants, set `active = false` in Supabase:

```sql
UPDATE studies SET active = false WHERE slug = 'my-study-2026';
```

Existing sessions in progress will continue to work. New participants loading the game with this study slug will see an error message.

---

## 7. Standalone Mode

When no `study` URL parameter is present, the game runs in standalone mode using the bundled `gameConfig.yaml`. Data is still recorded to Supabase (if configured), but without a `study_id` or `participant_id`. This is the default behavior for direct visitors to the site.

---

## 8. Troubleshooting

| Issue | Solution |
|---|---|
| "Study not found" error in the game | Verify the slug in the URL matches the study's `_slug` field. Check the study is `active = true` in Supabase. |
| Data not appearing in export | Ensure the session completed (player reached the results page). Check that the export key matches. |
| iframe not loading in Qualtrics | Qualtrics may block certain iframes. Try the survey in Preview mode first. Check browser console for errors. |
| `postMessage` not received | Ensure the JavaScript listener is set up correctly. The message type is `mygoodness:complete` (exact string). |
| Consent/survey appearing when disabled | Double-check your study YAML has `showConsent: false` / `showSurvey: false`. Re-run the create-study script if you changed the YAML after creation. |
