# MATATAG Reading Analysis API

FastAPI service for formative oral-reading measurements in Grade 1-3 activities.
It uses Vosk for word recognition and FFmpeg for audio normalization.

## v4 responsibility

The service measures an oral response. It does not choose the next activity and
does not treat an AI-generated passage as an official CRLA or Phil-IRI test.

The expected application flow is:

1. The generator creates an activity tagged with grade, MATATAG subdomain,
   competency code, activity type, and one selected difficulty.
2. The PHP backend stores that immutable activity context.
3. The backend calls this API only for speech-ready activities.
4. The backend scores tap/type/matching/comprehension responses itself.
5. Raw attempt measurements are sent to the adaptive recommender.

## Measurement profiles

| Profile | Activities | Returned evidence |
|---|---|---|
| `oral_word_accuracy` | phonics, sight words, isolated word reading | word accuracy and miscues |
| `oral_passage_fluency` | sentence, passage, timed, repeated, and oral reading | accuracy, raw WCPM, experimental prosody |
| Backend-scored | multiple choice, true/false, matching, sorting, sequencing, fill-in-the-blank | Do not send to this API |

Difficulty is activity-demand metadata. It does not change the formulas for
word accuracy or WCPM.

## Endpoints

- `GET /` - service summary
- `GET /health` - deployment and model status
- `GET /metadata` - generator-compatible activity contract
- `POST /analyze` - analyze a curriculum-tagged oral response
- `POST /calculate-proficiency` - deprecated fixed-weight heuristic retained
  only for migration

## Analyze request

`POST /analyze` uses `multipart/form-data`.

Required fields:

- `file`
- `grade`
- `subdomain`
- `competency_code`
- `activity_type`
- `difficulty`
- `reference_text`

Optional fields:

- `activity_id`
- `attempt_number` (default `1`)
- `constrain_vocabulary` (default `false`)
- `comprehension_score` supplied by the application backend
- `legacy_proficiency` (default `false`)

Example:

```bash
curl -X POST "http://localhost:10000/analyze" \
  -H "X-App-Key: replace-with-your-internal-key" \
  -F "file=@reading.webm" \
  -F "activity_id=act_123" \
  -F "grade=2" \
  -F "subdomain=Phonics and Word Study" \
  -F "competency_code=EN2PWS-I-3" \
  -F "activity_type=phonics" \
  -F "difficulty=medium" \
  -F "reference_text=ship chat thin ring" \
  -F "attempt_number=1"
```

Important response fields:

```json
{
  "activity_context": {
    "activity_id": "act_123",
    "grade": 2,
    "subdomain": "Phonics and Word Study",
    "competency_code": "EN2PWS-I-3",
    "activity_type": "phonics",
    "difficulty": "medium",
    "attempt_number": 1
  },
  "measurement_profile": "oral_word_accuracy",
  "eligible_metrics": ["word_accuracy"],
  "measurements": {
    "word_accuracy_percent": 75.0,
    "correct_words": 3,
    "substitutions": 1,
    "deletions": 0,
    "insertions": 0,
    "wcpm": null,
    "experimental_prosody_indicator": null,
    "comprehension_score": null
  },
  "quality_flags": [],
  "assessment_use": "formative_practice_not_official_classification",
  "scoring_profile_version": "reading-v4.0"
}
```

The original detailed `accuracy`, `speed`, `prosody`, timestamps, and timing
objects remain in the response for debugging and migration.

## Scoring policy

- Accuracy is word-level Vosk/reference-text agreement, not phoneme-level
  pronunciation scoring.
- WCPM is returned raw. There is no default Grade 1-3 target range.
- `speed_score` stays `null` unless locally validated `TARGET_WPM_MIN` and
  `TARGET_WPM_MAX` values are configured.
- Prosody is an experimental acoustic indicator based on pitch, energy, and
  pauses. It should not be used as a high-stakes score without local validation.
- Comprehension is scored by the application from activity answers, not inferred
  from speech audio.
- Overall proficiency is owned by the adaptive recommender.

## Environment variables

Copy `.env.example` into your deployment settings. Configure `APP_API_KEY` in
production and send it from the PHP backend as `X-App-Key`.

When `APP_API_KEY` is empty, protected endpoints remain open for local testing.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PRELOAD_MODEL=0 uvicorn main:app --reload --port 10000
```

The Docker image downloads the configured Vosk model and preloads it by default.

## Tests

```bash
PRELOAD_MODEL=0 python -m unittest -v
```
