# -*- coding: utf-8 -*-
# Standalone command-line entry point, extracted from cell 3 of
# IIR_Per_Scenario_Workflow.ipynb, for running without the VS Code Jupyter
# extension. Keep this in sync with that notebook cell if either changes.
#
# Usage (from this folder, after `gcloud auth application-default login`):
#   export GOOGLE_CLOUD_PROJECT=your-gcp-project-id   (Windows: set GOOGLE_CLOUD_PROJECT=...)
#   python prompting_detailed.py
#
# Run parameters (total_n, master_seed, model, etc.) come from
# config/experiment.json. Progress is saved after every scenario call, so
# this can be safely interrupted (Ctrl+C, closed terminal, machine restart)
# and resumed later by just running it again.
"""
Scenario-at-a-Time IIR Pipeline (variant)
-----------------------------------------
This is a structural variant of master_iir_workflow.py. It generates
simulated student responses ONE SCENARIO AT A TIME instead of all three
in a single call. Each student therefore costs 3 API calls instead of 1.

Purpose: produce data comparable to the current all-at-once pipeline so we
can empirically test whether the structural difference matters.

KEY DIFFERENCES vs. the original pipeline:
  - Three API calls per student, one per scenario.
  - Each call only sees ONE story and its 4 questions.
  - The model has no knowledge of the other scenarios when answering.
  - Within-scenario, the (a)/(b) question pair shuffling is preserved.
  - Across-scenario shuffling still happens (random order of Cat/Lying/Stealing).

CACHING NOTE:
  Vertex/Gemini implicit caching has minimum token thresholds (~4096 tokens
  for Gemini, last I checked — verify against current docs). Your system
  prompt + instructions are likely under that threshold, in which case
  caching produces zero savings. We pass `cached_content` only if it would
  plausibly help; otherwise we send normally.

  This script does NOT use explicit context caching (CachedContent objects).
  Implicit caching is the safer default — it's automatic, no setup, no risk
  of paying for a cache that never gets hit.

WHAT IS UNCHANGED FROM THE ORIGINAL:
  - Waypoint definitions, system prompt, output schema.
  - Format pool balancing (~half comic, half text per waypoint).
  - Resume logic via progress.json.
  - Per-student seed determinism (so the same student in both pipelines
    gets the same waypoint, same format, same scenario order).
"""

import os
import json
import time
import random
import pandas as pd
import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

# Google Cloud project to bill Vertex AI calls to. Read from the environment
# so no project ID is stored in the code.
PROJECT_CODE = os.environ.get("GOOGLE_CLOUD_PROJECT")
if not PROJECT_CODE:
    raise SystemExit("Set the GOOGLE_CLOUD_PROJECT environment variable to "
                     "your Google Cloud project ID before running.")

vertexai.init(project=PROJECT_CODE, location="global")
# --- Experiment configuration ---
# Run parameters live in experiment.json. Edit that file to change the model,
# sample size, seed, waypoint proportions, or experience phrasing.
with open("config/experiment.json", encoding="utf-8") as f:
    EXPERIMENT = json.load(f)

OUTPUT_DIR = EXPERIMENT["output_dir"]
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = EXPERIMENT["model_name"]

# Progress file for resume logic — mirrors master_iir_workflow.py's pattern.
# A separate filename so the two pipelines don't stomp on each other's progress.
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "scenario_at_a_time_progress.json")

# Live/partial CSV — rewritten (atomically) after every completed scenario so
# you can open it at any time, even mid-run or after an interruption, to see
# everything generated so far. Removed automatically once a run fully completes.
PARTIAL_CSV_PATH = os.path.join(OUTPUT_DIR, "scenario_at_a_time_partial.csv")

# -----------------------------------------------------------------------------
# Profiles, system prompt, story texts, comic paths
#
# These should mirror your master_iir_workflow.py EXACTLY. To keep this script
# self-contained I'm declaring them here, but in practice you should consider
# importing them from a shared module so the two pipelines can't drift apart.
# -----------------------------------------------------------------------------
# --- Externalized IIR waypoints ---
# Edit waypoints.json to change a level's label/definition or add a new level.
with open("config/waypoints.json", encoding="utf-8") as f:
    waypoints = json.load(f)

# -----------------------------------------------------------------------------
# Experience levels — 4-level scheme matching the original System_Prompt
#
# These are the same 4 levels referenced in the System_Prompt
# ("No experience" through "A lot"). Now they're actually injected into the
# user prompt rather than just mentioned in the system prompt.
#
# INDEPENDENT of the waypoint. Waypoint dictates depth and kind of inference;
# experience affects style, vocabulary, fluency, engagement — NOT reasoning depth.
#
# Format-conditional phrasing: for comic students this is "comic experience";
# for text students this is "reading comprehension experience". Same level
# structure, different framing — applied via FORMAT_PHRASING below.
# -----------------------------------------------------------------------------
with open("config/student_profile.json", "r", encoding="utf-8") as f:
    data = json.load(f)

experience_levels = data["experience_levels"]
personality_types = data["personality_types"]


# How to phrase "experience" depending on which format the student got.
FORMAT_PHRASING = EXPERIMENT["format_phrasing"]

# --- Externalized prompts & stories ---
# Wording lives in prompts/. Edit those files to change prompts; add a new
# story by dropping prompts/stories/<name>.txt and a scenarios_data entry.
PROMPT_DIR = "prompts"

def load_text(*parts):
    """Read a UTF-8 text asset from the prompts/ directory."""
    with open(os.path.join(PROMPT_DIR, *parts), encoding="utf-8") as f:
        return f.read()

def load_story(name):
    """Load a story by name from prompts/stories/<name>.txt."""
    return load_text("stories", f"{name}.txt")

System_Prompt = load_text("system_prompt.txt")

# Story texts — paste from your master_iir_workflow.py
cat_text = load_story("cat")
lying_text = load_story("lying")
stealing_text = load_story("stealing")

# Instructions block — copied verbatim from master_iir_workflow.py.
# Item 7 (continuity between (a) and (b)) still applies in the per-scenario
# context because (a)/(b) pairs are answered together in each call.
# Items 5 and 8 also apply unchanged.
instructions = load_text("instructions.txt")


def build_profile_prompt(level, experience_level=None, format_type=None, personality = None):
    """Profile block — copied verbatim from master_iir_workflow.py, extended
    with an optional experience-level block.

    The experience block uses format-conditional phrasing:
      - For comic students: "COMIC EXPERIENCE"
      - For text students:  "READING COMPREHENSION EXPERIENCE"

    The waypoint takes precedence — experience only affects style/fluency.
    """
    waypoint = waypoints[level]
    block = (
        f"==================== TARGET STUDENT PROFILE ====================\n"
        f"IIR Developmental Level: {waypoint['label']}\n"
        f"CONSTRUCT MAP DEFINITION:\n"
        f"{waypoint['definition']}\n"
    )

    if experience_level is not None:
        if format_type not in FORMAT_PHRASING:
            raise ValueError(
                f"format_type must be one of {list(FORMAT_PHRASING.keys())} "
                f"when experience_level is provided. Got: {format_type!r}"
            )
        exp = experience_levels[experience_level]
        header = FORMAT_PHRASING[format_type]
        block += (
            f"\n"
            f"{header}: {exp['label']}\n"
            f"{exp['definition']}\n"
            f"\n"
            f"IMPORTANT: The experience level above ONLY affects the student's "
            f"writing style, vocabulary, and fluency. It does NOT change the "
            f"reasoning depth or inference type — that is fully determined by "
            f"the IIR Developmental Level above. A 'Low experience' IIR3 student "
            f"still integrates text and world knowledge; they just express it "
            f"in simpler language. An 'High experience' IIR1 student still uses only "
            f"text-base information; they just express it more fluently.\n"
        )

    if personality is not None:
        trait, level_key = personality.split("_", 1)  # e.g. "openness_high" -> ("openness", "high")
        p = personality_types[trait][level_key]
        block += (
            f"\nSTUDENT PERSONALITY (EXPRESSION STYLE ONLY):\n"
            f"Trait: {personality_types[trait]['label']} — {level_key.upper()}\n"
            f"Description: {p['description']}\n"
            f"Writing tendencies: {'; '.join(p['writing_tendencies'])}\n"
            f"Engagement style: {p['engagement_style']}\n"
            f"\nIMPORTANT: Personality ONLY affects tone and surface presentation. "
            f"It does NOT affect reasoning depth or inference type — those are "
            f"fully determined by the IIR Developmental Level above.\n"
        )


    return block


# scenarios_data — uses the SAME structure as master_iir_workflow.py:
# pairs are dicts with 'questions' (pre-formatted string) and 'keys' (list).
# --- Externalized scenarios (single source of scenario identity) ---
# Edit scenarios.json to change questions, add a question pair, or add a whole
# new scenario (also drop prompts/stories/<story>.txt and the comic PDF).
def _load_scenarios(path="config/scenarios.json"):
    """Load scenarios.json and validate the structure the pipeline relies on.
    Rebuilds the runtime shape (text=story string, pairs=[{questions, keys}]).
    Keys must be unique across scenarios — they are the output columns and the
    JSON-parsing contract."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    scenarios, seen_keys = {}, []
    for name, sc in raw.items():
        for field in ("story", "comic", "pairs"):
            if field not in sc:
                raise ValueError(f"scenario {name!r} missing required field {field!r}")
        if len(sc["pairs"]) != 2:
            raise ValueError(f"scenario {name!r} must have exactly 2 question pairs, "
                             f"got {len(sc['pairs'])}")
        built = []
        for p in sc["pairs"]:
            if not isinstance(p.get("keys"), list) or not isinstance(p.get("questions"), list):
                raise ValueError(f"scenario {name!r}: each pair needs list 'keys' and 'questions'")
            seen_keys.extend(p["keys"])
            built.append({"questions": "\n".join(p["questions"]), "keys": p["keys"]})
        scenarios[name] = {"text": load_story(sc["story"]), "comic": sc["comic"], "pairs": built}
    if len(seen_keys) != len(set(seen_keys)):
        raise ValueError(f"duplicate response keys across scenarios: {seen_keys}")
    return scenarios

scenarios_data = _load_scenarios()
N_SCENARIOS = len(scenarios_data)  # used by the resume/completion logic below


# -----------------------------------------------------------------------------
# Per-scenario prompt builder
#
# This mirrors master_iir_workflow.py's build_prompt() as faithfully as possible,
# with the only structural change being: ONE scenario per call instead of three.
# Header style, instruction block, profile block, and JSON schema formatting
# all match the original verbatim. The orienting sentence is adapted for the
# single-scenario context.
# -----------------------------------------------------------------------------
def build_single_scenario_prompt(level: str, scenario_name: str, format_type: str,
                                 pair_seed: int, experience_level: str = None,  personality=None):
    """Build the user prompt for one scenario. Returns (System_Prompt, prompt_user, keys)
    matching the original build_prompt() return shape so call_api can be used
    interchangeably.

    The pair_seed determines the (a)/(b) pair order shuffle within this
    scenario, so it's reproducible.

    experience_level: optional 'low' | 'medium' | 'high'. If provided, an
    experience block is added to the profile prompt with explicit instructions
    that experience only affects style, not reasoning depth.
    """
    s_data = scenarios_data[scenario_name]

    # Shuffle pairs deterministically using a per-call seed
    pairs_rng = random.Random(pair_seed)
    pairs = list(s_data["pairs"])
    pairs_rng.shuffle(pairs)

    # Build questions block and key list — mirrors original line 352-354
    shuffled_questions = f"{pairs[0]['questions']}\n{pairs[1]['questions']}"
    json_keys_ordered = []
    json_keys_ordered.extend(pairs[0]['keys'])
    json_keys_ordered.extend(pairs[1]['keys'])

    # Build per-scenario block — mirrors original line 356-364
    scenarios_prompt_text = f"==================== SCENARIO: {scenario_name.upper()} ====================\n"

    document_parts = []
    if format_type == "text":
        scenarios_prompt_text += f"STORY:\n{s_data['text']}\n\n"
    elif format_type == "comic":
        with open(s_data['comic'], "rb") as f:
            document_parts.append(Part.from_data(data=f.read(), mime_type="application/pdf"))

    scenarios_prompt_text += f"QUESTIONS:\n{shuffled_questions}\n\n"

    # Dynamic JSON schema — mirrors original line 366-377
    json_format_lines = [f'    "{key}": "..."' for key in json_keys_ordered]
    json_format_str = "{\n" + ",\n".join(json_format_lines) + "\n}"

    dynamic_response_format = f"""
==================== RESPONSE FORMAT ====================
Return a single valid JSON object with exactly these {len(json_keys_ordered)} keys in the exact order shown below.
Each value should be a short open-ended text response as the student would write it.
Do not include any explanation or text outside the JSON.

{json_format_str}
"""

    # Orienting sentence adapted for single-scenario context.
    if format_type == "text":
        orienting = "Below is one scenario. Please read the story and answer the questions."
        prompt_text = (
            build_profile_prompt(level, experience_level, format_type,personality) + "\n\n" +
            orienting + "\n\n" +
            scenarios_prompt_text +
            dynamic_response_format + "\n\n" +
            instructions
        )
        prompt_user = prompt_text

    elif format_type == "comic":
        orienting = (
            f"Below is one comic story attached as a document ({scenario_name}). "
            "Please read the comic and answer the corresponding questions."
        )
        prompt_text = (
            build_profile_prompt(level, experience_level, format_type,personality) + "\n\n" +
            orienting + "\n\n" +
            scenarios_prompt_text +
            dynamic_response_format + "\n\n" +
            instructions
        )
        prompt_user = document_parts + [prompt_text]

    return System_Prompt, prompt_user, json_keys_ordered


# -----------------------------------------------------------------------------
# API call with retry
# -----------------------------------------------------------------------------
def call_api(user_payload, max_retries=3):
    """Single API call with exponential backoff retry.

    NOTE on caching: We rely on Vertex's IMPLICIT caching, which automatically
    detects identical prompt prefixes across calls. The system_instruction is
    identical across every call in this pipeline, so it's an ideal candidate
    for implicit caching. We don't pay anything to enable it — Vertex either
    caches the prefix automatically or doesn't, depending on its threshold.
    """
    model = GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=System_Prompt,
    )

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = model.generate_content(
                user_payload,
                generation_config=GenerationConfig(
                    temperature=EXPERIMENT["temperature"],
                    response_mime_type="application/json",
                ),
            )
            return resp.text
        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            print(f"  API error (attempt {attempt+1}): {e}. Waiting {wait}s...")
            time.sleep(wait)
    raise last_err


def parse_response(raw_text, expected_keys):
    """Parse JSON into the expected keys; unparseable or wrong-shaped -> None."""
    try:
        data = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError):
        return {k: None for k in expected_keys}
    # The model occasionally wraps the object in a list (e.g. [ {...} ]) or
    # returns a bare list. Recover the dict if possible; otherwise None it out
    # so one malformed response cannot crash the whole run.
    if isinstance(data, list):
        data = next((d for d in data if isinstance(d, dict)), None)
    if not isinstance(data, dict):
        return {k: None for k in expected_keys}
    return {k: data.get(k, None) for k in expected_keys}


# -----------------------------------------------------------------------------
# Per-student generation: 3 calls, one per scenario
#
# This function supports partial completion: if `existing_state` already
# contains responses for some scenarios, those are skipped and only the
# remaining scenarios are generated. This is what allows resume logic to
# work at the per-scenario level (not just per-student).
# -----------------------------------------------------------------------------
ALL_KEYS = [k for sc in scenarios_data.values() for pair in sc["pairs"] for k in pair["keys"]]  # single source: scenarios.json


def generate_one_student(student_idx: int, level: str, format_type: str,
                         master_seed: int, experience_level: str = None,
                         existing_state=None,
                         personality=None,
                         on_scenario_done=None):
    """Generate all 12 answers for one student via up to 3 separate API calls.

    Args:
        student_idx: 0-based index of this student.
        level: IIR waypoint string (e.g., "IIR1").
        format_type: "comic" or "text".
        master_seed: Used to derive a deterministic per-student RNG.
        experience_level: 'low' | 'medium' | 'high' | None. Independent of
            waypoint; affects style only, not reasoning depth.
        existing_state: Optional dict from a prior interrupted run. Shape:
            {
                "scenario_order": ["Cat", "Lying", "Stealing"],
                "completed_scenarios": ["Cat"],
                "answers": {...partial dict of 12 keys...},
                "raw_json_by_scenario": {"Cat": "<raw>"},
            }
            If provided, scenarios already in completed_scenarios are skipped.
        on_scenario_done: Optional callback(student_idx, scenario_name, state)
            invoked after each scenario completes. Lets the caller persist
            progress between scenarios so an interruption mid-student doesn't
            lose work.

    Returns:
        (full_answers, state) where state is the full per-student state dict
        suitable for serialization to progress file.
    """
    # Per-student RNG drives scenario order and pair shuffles. This is
    # deterministic given (master_seed, student_idx), so a resumed run will
    # produce identical scenario ordering and pair seeds.
    student_rng = random.Random(f"{master_seed}_{student_idx}")
    scenario_order = list(scenarios_data.keys())  # scenario set comes from scenarios.json
    student_rng.shuffle(scenario_order)

    # Pre-generate all pair_seeds so they're stable even when we skip scenarios
    # on resume. Without this, skipping a completed scenario would advance the
    # RNG differently than on the first run, changing the seeds for later
    # scenarios.
    pair_seeds = [student_rng.randint(0, 10**9) for _ in scenario_order]

    # Initialize state — either fresh or from a partial prior run
    if existing_state is None:
        state = {
            "scenario_order": scenario_order,
            "completed_scenarios": [],
            "answers": {},
            "raw_json_by_scenario": {},
        }
    else:
        state = existing_state
        if state.get("scenario_order") != scenario_order:
            print(f"  WARNING: stored scenario_order does not match regenerated "
                  f"order for student {student_idx}. Discarding partial state "
                  f"and starting this student fresh.")
            state = {
                "scenario_order": scenario_order,
                "completed_scenarios": [],
                "answers": {},
                "raw_json_by_scenario": {},
            }

    for scenario_name, pair_seed in zip(scenario_order, pair_seeds):
        if scenario_name in state["completed_scenarios"]:
            continue  # already done in a prior run

        _, user_payload, expected_keys = build_single_scenario_prompt(
            level, scenario_name, format_type, pair_seed,
            experience_level=experience_level,
            personality=personality,
        )

        raw = call_api(user_payload)
        parsed = parse_response(raw, expected_keys)
        state["answers"].update(parsed)
        state["raw_json_by_scenario"][scenario_name] = raw
        state["completed_scenarios"].append(scenario_name)

        if on_scenario_done is not None:
            on_scenario_done(student_idx, scenario_name, state)

        time.sleep(0.3)

    full_answers = {k: state["answers"].get(k, None) for k in ALL_KEYS}
    return full_answers, state


# -----------------------------------------------------------------------------
# Pipeline runner
# -----------------------------------------------------------------------------
def _atomic_save_progress(payload, path=PROGRESS_FILE):
    """Write progress JSON atomically (temp file + rename) so an interrupted
    write can't corrupt the file. The os.replace is atomic on POSIX."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f)
    os.replace(tmp_path, path)


def _atomic_save_csv(df, path):
    """Same pattern for the output CSV."""
    tmp_path = path + ".tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def _build_dataframe(student_states, total_n, level_assignments, format_assignments,
                     experience_assignments, personality_assignments):
    """Build the respondent-level DataFrame from student_states as they stand
    right now. Safe to call mid-run: students with no state yet, or with only
    some scenarios done, get None for any answer not yet generated."""
    rows = []
    for i in range(total_n):
        state = student_states[i]
        if state is None:
            answers = {k: None for k in ALL_KEYS}
            scenario_order = []
        else:
            answers = {k: state["answers"].get(k, None) for k in ALL_KEYS}
            scenario_order = state.get("scenario_order", [])

        rows.append({
            "respondent_id": f"R{i+1:04d}",
            "profile_level": level_assignments[i],
            "format": format_assignments[i],
            "experience": experience_assignments[i],
            "scenario_order": "|".join(scenario_order),
            "personality": personality_assignments[i],
            **answers,
        })
    return pd.DataFrame(rows)


def run_scenario_at_a_time_pipeline(total_n: int = EXPERIMENT["total_n"], master_seed: int = EXPERIMENT["master_seed"],
                                    level_proportions=None,
                                    output_csv: str = None,
                                    resume: bool = True):
    """Run the scenario-at-a-time pipeline.

    Default total_n=96 → 4 students per (waypoint x format x experience) cell
    given 4 waypoints x 2 formats x 3 experience levels = 24 cells.
    For uniform balancing, total_n should be a multiple of 24.

    Cell sizes at common N values (with 24 cells):
        N=24  → 1 per cell  (technically uniform but useless for analysis)
        N=48  → 2 per cell  (very thin)
        N=72  → 3 per cell  (recommended minimum)
        N=96→ 4 per cell
        N=120 → 5 per cell  (smallest size where cell-level patterns are readable)

    Resume behavior:
      - If a progress file exists matching all assignment vectors, it's
        loaded and generation picks up where it left off.
      - Resume granularity is per-scenario: an interruption mid-student
        loses at most one in-flight scenario call, not the whole student.
      - If progress params don't match, the file is ignored and a fresh
        run begins. The script does NOT delete the old file in that case.

    Live progress:
      - IIR_outputs/scenario_at_a_time_partial.csv is rewritten (atomically)
        after every completed scenario. Open it any time, mid-run or after an
        interruption, to see everything generated so far — students not yet
        started, or only partially done, show None in unanswered columns.
      - Removed automatically once the run fully completes, since the
        timestamped output_csv is the definitive final file at that point.
    """
    if level_proportions is None:
        level_proportions = EXPERIMENT["level_proportions"]
    if output_csv is None:
        # Timestamp the filename so each run writes a NEW file instead of
        # overwriting. Toggle with "timestamp_output" in config/experiment.json.
        base = EXPERIMENT["output_csv"]
        if EXPERIMENT.get("timestamp_output", True):
            stem, ext = os.path.splitext(base)
            base = f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}{ext}"
        output_csv = os.path.join(OUTPUT_DIR, base)

    # Validate proportions before spending any API calls. If they don't sum
    # to 1.0, some share of total_n was never assigned to a level (e.g. a
    # level was zeroed out without redistributing its share to the others),
    # which would otherwise get silently dumped onto whichever level happens
    # to be listed last in level_proportions.
    prop_sum = sum(level_proportions.values())
    if abs(prop_sum - 1.0) > 1e-6:
        raise ValueError(
            f"level_proportions in config/experiment.json sum to {prop_sum}, "
            f"not 1.0: {level_proportions}. If you zeroed out a level, "
            f"redistribute its share among the remaining levels so the "
            f"proportions sum to 1.0."
        )

    # Sanity warnings about cell sizes — fire BEFORE any API calls so the
    # user can abort if the design isn't what they wanted. A level with
    # proportion 0.0 doesn't occupy a design cell.
    active_levels = [lvl for lvl, prop in level_proportions.items() if prop > 0]
    n_waypoints = len(active_levels)
    n_formats = 2
    n_exp = 3
    n_cells = n_waypoints * n_formats * n_exp  # 24 with 4 active waypoints
    if total_n % n_cells != 0:
        print(f"WARNING: total_n={total_n} is not a multiple of {n_cells} "
              f"({n_waypoints} active waypoints x {n_formats} formats x {n_exp} "
              f"experience levels). Some cells will have one more student "
              f"than others (within-cell imbalance of at most 1).")
    cell_size = total_n // n_cells
    if cell_size < 2:
        print(f"WARNING: cell size = {cell_size} student(s). Below 2 per cell "
              f"the design is effectively unbalanced for analysis. "
              f"Recommend total_n >= {n_cells * 2}.")

    # ---- Build deterministic level + format assignments ----
    rng = random.Random(master_seed)

    # Apportion total_n across levels via the largest-remainder method, so
    # rounding leftovers are spread across the levels with the largest
    # fractional share instead of dumped entirely onto whichever level is
    # listed last. Proportions are validated to sum to 1.0 above, so this
    # always accounts for exactly total_n students, and a level at 0.0
    # always gets exactly 0.
    exact_counts = {lvl: total_n * prop for lvl, prop in level_proportions.items()}
    level_counts = {lvl: int(exact_counts[lvl]) for lvl in level_proportions}
    shortfall = total_n - sum(level_counts.values())
    by_remainder = sorted(
        level_proportions.keys(),
        key=lambda lvl: exact_counts[lvl] - level_counts[lvl],
        reverse=True,
    )
    for lvl in by_remainder[:shortfall]:
        level_counts[lvl] += 1

    level_assignments = []
    for lvl in level_proportions.keys():
        level_assignments.extend([lvl] * level_counts[lvl])

    format_assignments = [None] * total_n
    for lvl in level_proportions.keys():
        idxs = [i for i, l in enumerate(level_assignments) if l == lvl]
        n_lvl = len(idxs)
        n_comic = n_lvl // 2
        formats = ["comic"] * n_comic + ["text"] * (n_lvl - n_comic)
        rng.shuffle(formats)
        for idx, fmt in zip(idxs, formats):
            format_assignments[idx] = fmt

    # ---- Experience-level assignments ----
    # Balanced uniformly within each (waypoint, format) cell across 3 levels.
    # # With cell_size = 4 (n=96), the split is a perfect 1/1/1 per cell.
    # With cell_size = 8 (n=192), perfect 2/2/2 per cell. Etc.
    #
    # The assignment is deterministic given master_seed because we use the
    # same `rng` that was already seeded above.
    EXP_LEVELS = list(experience_levels.keys())  # from student_profile.json
    assert len(EXP_LEVELS) == 3, (
        f"Experience-level balancing assumes exactly 3 levels (got {len(EXP_LEVELS)}); "
        f"changing the count alters the cell math and total_n divisibility."
    )
    # Personality — randomly sampled, not balanced
    # Personality options derived from student_profile.json — add a trait or pole
    # there and it is picked up automatically. Sampled (not balanced), so the pool
    # size is free; order follows the JSON, which fixes the seeded RNG stream.
    PERSONALITY_OPTIONS = [
        f"{trait}_{pole}"
        for trait, spec in personality_types.items() if isinstance(spec, dict)
        for pole, pdef in spec.items() if isinstance(pdef, dict)
    ]
    personality_assignments = [rng.choice(PERSONALITY_OPTIONS) for _ in range(total_n)]


    experience_assignments = [None] * total_n
    for lvl in level_proportions.keys():
        for fmt in ("comic", "text"):
            idxs = [
                i for i in range(total_n)
                if level_assignments[i] == lvl and format_assignments[i] == fmt
            ]
            n_cell = len(idxs)
            # Build a balanced pool: as even a split across 3 levels as possible.
            base = n_cell // 3
            remainder = n_cell - 3 * base
            pool = []
            for j, lvl_name in enumerate(EXP_LEVELS):
                # Distribute the remainder across the first `remainder` levels.
                # To avoid systematically favoring certain experience levels
                # when n_cell isn't a multiple of 4, shuffle which levels get
                # the +1 across cells. We do this by picking `remainder` levels
                # at random from EXP_LEVELS for this specific cell.
                pass
            # Pick which levels get the extra student in this cell
            extra_levels = rng.sample(EXP_LEVELS, remainder) if remainder else []
            for lvl_name in EXP_LEVELS:
                count = base + (1 if lvl_name in extra_levels else 0)
                pool.extend([lvl_name] * count)
            rng.shuffle(pool)
            for idx, exp in zip(idxs, pool):
                experience_assignments[idx] = exp

    # ---- Resume logic ----
    # student_states is a list of per-student state dicts (None until a student
    # has at least one scenario completed). This is what gets persisted to
    # the progress file.
    student_states = [None] * total_n
    start_i = 0  # Index of the first student that needs *any* work

    if resume and os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r") as f:
                prog = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"==> Progress file unreadable ({e}). Starting fresh.")
            prog = None

        if prog is not None and (
            prog.get("total_n") == total_n
            and prog.get("master_seed") == master_seed
            and prog.get("level_assignments") == level_assignments
            and prog.get("format_assignments") == format_assignments
            and prog.get("experience_assignments") == experience_assignments
            and prog.get("personality_assignments") == personality_assignments
        ):
            student_states = prog["student_states"]
            # Find the first incomplete student. A student is "complete" if
            # all 3 scenarios are in completed_scenarios.
            for idx, state in enumerate(student_states):
                if state is None or len(state.get("completed_scenarios", [])) < N_SCENARIOS:
                    start_i = idx
                    break
            else:
                start_i = total_n  # everything done
            n_complete = sum(
                1 for s in student_states
                if s is not None and len(s.get("completed_scenarios", [])) == N_SCENARIOS
            )
            print(f"==> Resuming from progress file: "
                  f"{n_complete}/{total_n} students fully complete, "
                  f"resuming at student index {start_i}.")
        else:
            print("==> Progress file found but parameters differ from current "
                  "run. Starting fresh. (Old file not deleted — remove it "
                  f"manually at {PROGRESS_FILE} if you want to discard it.)")

    # ---- Sanity report ----
    print(f"Level distribution: "
          f"{pd.Series(level_assignments).value_counts().to_dict()}")
    df_verify = pd.DataFrame({"Level": level_assignments,
                              "Format": format_assignments,
                              "Experience": experience_assignments})
    print("Format breakdown per level:")
    print(df_verify.groupby(['Level', 'Format']).size().unstack(fill_value=0).to_dict('index'))
    print("Experience breakdown per (level, format):")
    print(df_verify.groupby(['Level', 'Format', 'Experience']).size().unstack(fill_value=0))
    print(f"Live progress CSV (updates after every scenario): {PARTIAL_CSV_PATH}")

    # ---- Helper: persist progress after each scenario ----
    def save_progress():
        _atomic_save_progress({
            "total_n": total_n,
            "master_seed": master_seed,
            "level_assignments": level_assignments,
            "format_assignments": format_assignments,
            "experience_assignments": experience_assignments,
            "student_states": student_states,
            "personality_assignments": personality_assignments,
        })

    def save_partial_csv():
        partial_df = _build_dataframe(student_states, total_n, level_assignments,
                                      format_assignments, experience_assignments,
                                      personality_assignments)
        _atomic_save_csv(partial_df, PARTIAL_CSV_PATH)

    def on_scenario_done(student_idx, scenario_name, state):
        student_states[student_idx] = state
        save_progress()
        save_partial_csv()

    # ---- Main generation loop ----
    if start_i >= total_n:
        print("All responses already generated. Skipping API calls.")
    else:
        pipeline_start = time.time()
        student_times = []

        for i in range(start_i, total_n):
            level = level_assignments[i]
            fmt = format_assignments[i]
            exp = experience_assignments[i]
            n_done = (
                len(student_states[i]["completed_scenarios"])
                if student_states[i] is not None else 0
            )
            elapsed_total = time.time() - pipeline_start
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_total))
            base_msg = (f"[{i+1}/{total_n}] R{i+1:04d} level={level} "
                        f"format={fmt} experience={exp} | elapsed={elapsed_str}")
            print(base_msg + (f" (resuming with {n_done}/3 scenarios done)"
                              if n_done > 0 else ""))

            student_start = time.time()
            try:
                answers, state = generate_one_student(
                    i, level, fmt, master_seed,
                    experience_level=exp,
                    existing_state=student_states[i],
                    personality=personality_assignments[i],
                    on_scenario_done=on_scenario_done,
                )
                student_states[i] = state
            except Exception as e:
                print(f"\n  FAILED on student {i+1}: {e}")
                save_progress()
                save_partial_csv()
                raise

            student_elapsed = time.time() - student_start
            student_times.append(student_elapsed)
            students_left = total_n - i - 1
            avg_time = sum(student_times) / len(student_times)
            eta_seconds = avg_time * students_left
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
            total_elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - pipeline_start))
            print(f"  -> done in {student_elapsed:.1f}s | total elapsed={total_elapsed_str} | ETA={eta_str} ({students_left} students left)")

    # ---- Build final CSV from student_states ----
    df = _build_dataframe(student_states, total_n, level_assignments,
                          format_assignments, experience_assignments,
                          personality_assignments)
    _atomic_save_csv(df, output_csv)

    # Clean up progress file and partial CSV only on a fully successful run
    n_complete = sum(
        1 for s in student_states
        if s is not None and len(s.get("completed_scenarios", [])) == N_SCENARIOS
    )
    if n_complete == total_n:
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)
            print(f"All {total_n} students complete. Progress file removed.")
        if os.path.exists(PARTIAL_CSV_PATH):
            os.remove(PARTIAL_CSV_PATH)

    print(f"\nSaved {len(df)} respondents to {output_csv}")
    return df


if __name__ == "__main__":
    # All run parameters come from config/experiment.json (total_n, master_seed,
    # model, temperature, level_proportions, output filename). Edit that file to
    # change the run. Keep total_n a multiple of 24 for uniform balancing across
    # the (waypoint x format x experience) = 4 x 2 x 3 design cells.
    run_scenario_at_a_time_pipeline()
