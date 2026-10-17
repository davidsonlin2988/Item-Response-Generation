# IIR simulated student responses: response generation

Code that asks Google Gemini (through Vertex AI) to write answers to the IIR
reading-inference questions as a student at a given IIR level would.

## Files

| File | What it is |
|---|---|
| `prompting_detailed.py` | Detailed (later) version. One API call per story, three per student. Each student also gets an experience level and a personality trait. |
| `prompting_basic.ipynb` | Basic (earlier) version, a Google Colab notebook. One API call per student covering all three stories. |
| `PROMPT_TEMPLATES.txt` | System prompt and user prompt templates for both versions, with the values that fill each placeholder. |
| `PROMPT_TEMPLATES.pdf` | The same content as `PROMPT_TEMPLATES.txt`, formatted for reading: prompts in grey boxes, placeholders in blue. |
| `config/` | Settings read by `prompting_detailed.py`: `experiment.json` (model, sample size, seed, IIR level proportions), `waypoints.json` (IIR level definitions), `student_profile.json` (experience levels, personality traits), `scenarios.json` (questions and output keys). |
| `prompts/` | System prompt, instructions, story texts and comic PDFs. |

## Running `prompting_detailed.py`

Tested with Python 3.13. You need a Google Cloud project with Vertex AI enabled.

```
pip install -r requirements.txt
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT=your-gcp-project-id     # Windows: set GOOGLE_CLOUD_PROJECT=your-gcp-project-id
python prompting_detailed.py
```

Run it from this folder. The results are written to `IIR_outputs/` as a CSV
with one row per simulated student. To change the model, sample size or seed,
edit `config/experiment.json`.

The script uses the `vertexai.generative_models` module, which Google has
deprecated. Newer versions of `google-cloud-aiplatform` may not include it,
so use the version pinned in `requirements.txt`.

## Running the notebook

1. Open the notebook in Google Colab.
2. Upload `Cat.pdf`, `Lying.pdf` and `Stealing.pdf` from `prompts/comics/` to
   the top level of your Google Drive (My Drive).
3. In the `#API call` cell, set `PROJECT_ID` to your Google Cloud project ID.
4. Run all cells. The results are saved to `IIR_outputs/IIR_raw_answers_V2.csv`.
