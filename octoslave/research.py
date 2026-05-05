"""
OctoSlave — autonomous multi-agent long-research pipeline.

Pipeline per round:
  Researcher → HypothesisGenerator → Coder → Debugger → Evaluator → Orchestrator

The Orchestrator synthesises each round and writes the brief for the next one.
Everything is persisted to disk so runs can be inspected or resumed.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from openai import OpenAI, BadRequestError

from . import display
from .agent import _cap_result, _compact_and_trim as _trim_messages
from .tools import TOOL_DEFINITIONS, execute_tool

# ---------------------------------------------------------------------------
# Role registry
# ---------------------------------------------------------------------------

ROLES: dict[str, dict] = {
    "researcher": {
        "label": "Researcher",
        "icon": "🔬",
        "color": "bold cyan",
        "default_model": "deepseek-v3.2-thinking",           # large — fast reading + search
        "max_iter": 15,                             # 15 = budget for ~5 web ops + write
        "tools": ["read_file", "write_file", "web_search", "web_fetch",
                  "list_dir", "glob",
                  # First-class bio/chem connectors — prefer over web_fetch
                  "bio_inspect", "uniprot_lookup", "pubchem_lookup",
                  "chembl_lookup", "geo_search", "ena_fetch",
                  "pdb_fetch", "alphafold_fetch",
                  "pdf_ocr"],  # no bash — researcher surveys, never installs
    },
    "hypothesis": {
        "label": "Experiment Designer",
        "icon": "💡",
        "color": "bold bright_magenta",
        "default_model": "deepseek-v3.2-thinking",  # thinking — commit to the right experiment
        "max_iter": 8,
        "tools": ["read_file", "write_file", "list_dir", "glob",
                  "bio_inspect", "rdkit_describe"],
    },
    "coder": {
        "label": "Coder",
        "icon": "💻",
        "color": "bold green",
        "default_model": "qwen3-coder-30b",         # large code model — fewer mistakes
        "max_iter": 50,
        "tools": ["read_file", "write_file", "edit_file", "bash",
                  "glob", "grep", "list_dir",
                  "bio_inspect", "rdkit_describe", "pdb_fetch",
                  "alphafold_fetch", "uniprot_lookup", "pubchem_lookup",
                  "chembl_lookup", "ena_fetch", "pdf_ocr"],
    },
    "debugger": {
        "label": "Debugger",
        "icon": "🐛",
        "color": "bold red",
        "default_model": "qwen3-coder-30b",         # same coder — knows the code
        "max_iter": 20,
        "tools": ["read_file", "write_file", "edit_file", "bash",
                  "glob", "grep", "list_dir",
                  "bio_inspect", "rdkit_describe"],
    },
    "evaluator": {
        "label": "Evaluator",
        "icon": "⚖️ ",
        "color": "bold yellow",
        "default_model": "deepseek-v3.2-thinking",  # thinking — rigorous scientific judgement
        "max_iter": 15,
        "tools": ["read_file", "bash", "write_file", "list_dir",
                  "web_search", "glob",
                  "bio_inspect", "rdkit_describe"],
    },
    "orchestrator": {
        "label": "Orchestrator",
        "icon": "🧠",
        "color": "bold bright_white",
        "default_model": "deepseek-v3.2",           # strong reasoning — synthesis + direction
        "max_iter": 8,
        "tools": ["read_file", "write_file", "list_dir", "glob"],
    },
    "reporter": {
        "label": "Reporter",
        "icon": "📊",
        "color": "bold bright_cyan",
        "default_model": "deepseek-v3.2",            # large general — clean HTML/writing
        "max_iter": 40,
        "tools": ["read_file", "write_file", "bash", "list_dir", "glob"],
    },
    "merger": {
        "label": "Merger",
        "icon": "🔀",
        "color": "bold bright_cyan",
        "default_model": "deepseek-v3.2",
        "max_iter": 12,
        "tools": ["read_file", "write_file"],
    },
}

# Roles that can run as N independent parallel copies (no intra-round dependencies)
PARALLEL_ROLES: frozenset[str] = frozenset({"researcher", "hypothesis", "evaluator"})

# Per-round pipeline — reporter runs ONCE at the very end, not each round
PIPELINE: list[str] = [
    "researcher",
    "hypothesis",
    "coder",
    "debugger",
    "evaluator",
    "orchestrator",
]

# Expected output paths (relative to round_dir)
OUTPUT_FILES: dict[str, str] = {
    "researcher":    "01_literature.md",
    "hypothesis":    "02_experiment.md",
    "coder":         "03_code/",          # directory
    "debugger":      "04_debug_report.md",
    "evaluator":     "05_evaluation.md",
    "orchestrator":  "06_synthesis.md",
    "reporter":      "07_report.html",
}

FINDINGS_FILE = "findings.md"
CASE_MEMORY_FILE = "case_memory.md"
SKILLS_FILE = "skills.md"
NEXT_BRIEF_MARKER = "## NEXT_ROUND_BRIEF"
COMPLETE_MARKER = "## STATUS: COMPLETE"

# Local input file extensions recognised across the pipeline (researcher discovery,
# handoff stub fallback, etc.). Single source of truth — keep aligned.
LOCAL_DATA_EXTENSIONS: frozenset[str] = frozenset({
    # tabular / structured
    ".csv", ".tsv", ".parquet", ".feather", ".xlsx", ".xls",
    ".json", ".jsonl", ".yaml", ".yml", ".toml",
    # docs
    ".pdf", ".txt", ".md", ".rst",
    # bio sequences / formats
    ".fasta", ".fa", ".faa", ".fna", ".fastq", ".fq",
    ".bed", ".vcf", ".gff", ".gff3", ".gtf",
    # chemistry / structure
    ".pdb", ".cif", ".mol", ".mol2", ".sdf", ".smi", ".xyz",
    # arrays / scientific
    ".npy", ".npz", ".h5", ".hdf5", ".nc",
})


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_SHARED_HEADER = """\
You are the {label} in OctoSlave's multi-agent research pipeline.

TOPIC     : {topic}
ROUND     : {round_num} / {max_rounds}  {final_tag}
ROUND DIR : {round_dir}
RESEARCH  : {research_dir}
WORK DIR  : {working_dir}

BRIEF:
{brief}

EXECUTION RULES — non-negotiable:
- ACT, don't narrate. Zero preamble. "I will now..." costs a tool call. Skip it.
- Read ONLY the section you need (use offset/limit on read_file). Never read a whole file.
- Write each output file ONCE. No drafts, no re-reads, no polish passes.
- INTERMEDIATE FILES (01_literature.md … 06_synthesis.md) are terse HANDOFFS, not reports.
  They exist so the next agent can start fast — not to document your reasoning.
  If it fits in a bullet list, use a bullet list. Prose is waste.
- STOP the moment your required output file is written. Do not make additional tool calls
  to "verify", "review", or "summarise". The next agent will read it directly.
- LONG TASKS (training, data download) are expected to take hours. Do not abort them.
  Pass an appropriate timeout to bash (see PACKAGES / LONG-RUNNING JOBS below).
---
"""

_SCRAPE_RESEARCHER_PROMPT = """\
YOUR MISSION
You are a web scraping specialist. Your job is to crawl the target website's full
category/content tree, extract structured data, and save it for downstream analysis.

STEPS
1. Call crawl_tree on the root URL. Set max_depth and url_pattern to stay focused
   on the content tree (e.g. url_pattern to match category paths).
   Always set output_path to {round_dir}/scraped_tree.json.
2. Review the returned tree — identify the deepest leaf pages that contain actual data.
3. web_fetch up to 5 representative leaf pages to understand the data structure
   (fields, formats, patterns).
4. Write {round_dir}/01_literature.md with:
   ## Scraped Tree Summary
   - Total pages crawled, max depth reached, engine used
   - Tree shape: which URL patterns correspond to which content levels
   ## Data Structure
   - Fields found on leaf pages (name, price, description, etc.)
   - Format of each field (string, number, list, etc.)
   ## Sample Records
   - 3–5 concrete examples extracted from leaf pages
   ## FOR THE EXPERIMENT DESIGNER
   - Recommended approach for full extraction (pagination patterns, rate limits,
     auth requirements, JS rendering needs)
   - Exact output_path where the tree JSON was saved

CONSTRAINTS
- Use crawl_tree ONCE for the initial tree discovery.
- Use web_fetch for up to 5 leaf pages only — do not re-crawl.
- Write 01_literature.md as your LAST action. Stop immediately after.
"""

_ROLE_PROMPTS: dict[str, str] = {

"researcher": """\
YOUR MISSION
Fast, targeted intelligence-gathering pass. Equip the Experiment Designer with
exactly what they need to commit to ONE concrete experiment. 3 sharp sources
beat 10 shallow ones. Total output: under 500 words.

HARD LIMITS — these protect your token budget so you have enough left to write 01_literature.md.
Every limit below is a MAXIMUM, not a target:
  list_dir:    1 call   (step 0 only)
  read_file:   0–2 calls (local data files only — NOT task.md, NOT findings.md twice)
  bio_inspect: 0–3 calls (use on local FASTA / VCF / h5ad / PDB / SDF instead of read_file)
  web_search:  max 2 calls  ← STRICT. Stop searching after 2.
  web_fetch:   max 2 calls  ← STRICT. Stop fetching after 2.
  write_file:  1 call   (your LAST call — always)
  TOTAL: max 8 tool calls. Your 8th or earlier call MUST be write_file for 01_literature.md.

DOMAIN CONNECTORS — prefer these over web_search / web_fetch for biology + chemistry:
  uniprot_lookup     proteins (UniProt accession or query) → name, organism, GO, PDB xrefs
  pubchem_lookup     small molecules by name / CID / SMILES → properties
  chembl_lookup      bioactive / drug-like molecules → max phase, RO5, indications
  geo_search         NCBI GEO / SRA studies → accessions, sample counts, platforms
  ena_fetch          ENA / SRA file report → FASTQ download URLs, read counts
  pdb_fetch          RCSB PDB experimental structures by 4-char ID
  alphafold_fetch    AlphaFold DB predicted structures by UniProt accession
  pdf_ocr            recover numbers from PDF figures (axis ticks, EC50/IC50 values,
                     heat-map legends). USE when read_file says "value reported in
                     Figure N" or you see "EXTRACTION FAILED" — do NOT give up at that
                     point, OCR the relevant pages first. Pass pages="N-M".
These calls do NOT count against web_search / web_fetch budgets, but TOTAL calls
still capped at 8. Use a connector first; fall back to web_fetch only if it fails.

RESEARCHER CONSTRAINTS — non-negotiable:
- Do NOT read task.md. The topic is already in your brief above — reading it again wastes a call.
- Do NOT install packages. You have no bash tool. Survey only.
- Do NOT run code. Do NOT validate datasets programmatically.
- DATASET ACCESSIBILITY RULE (strictly enforced — NO EXCEPTIONS):
  * A dataset is ACCESSIBLE only if YOU PERSONALLY fetched its direct download URL in
    THIS session and got back actual data (>1KB, parseable CSV/TSV/FASTA/JSON).
  * NEVER cite a URL you have not fetched. NEVER cite a URL or dataset "from memory" /
    training data (model knowledge of canonical dataset names is unreliable; URLs go
    stale, hosts move, and download paths change). If you did not fetch it in THIS
    session and confirm a parseable response, it does not exist for our purposes.
  * Fetching just the landing page is NOT sufficient — landing pages always load even for paywalled data.
  * Mark UNAVAILABLE if: download URL returns HTML/error, requires login, or returns <1KB.
  * Mark ACCESSIBLE only if: direct CSV/TSV/FASTA fetch returns parseable data (>1KB).
  * If you cannot confirm a direct download → DO NOT LIST THE DATASET AT ALL. Listing
    a "REQUIRES_SIGNUP" dataset misleads the Designer/Coder into proposing experiments
    that depend on it. Better to recommend NO external dataset than an unverified one.
  * case_memory.md records which datasets FAILED in prior rounds — if listed there as failed,
    mark them UNAVAILABLE immediately without fetching again.
  * PREFERRED ALTERNATIVES when external datasets are unverifiable: HuggingFace
    `datasets.load_dataset(...)` (try the name; library will tell you instantly if it
    exists), GitHub raw CSVs from the cited paper's supplementary repo, and the bio_*
    connector tools (uniprot/pubchem/chembl/pdb/alphafold/geo/ena). These are
    programmatic and verifiable in one call.
- After 2 web_search + 2 web_fetch calls, you have gathered enough. WRITE the file immediately.

STEPS
0. LOCAL DATA FIRST (mandatory, round 1 and every round):
   Call list_dir on {working_dir}. If any PDFs, CSVs, TSVs, FASTA/FA files, or JSON files
   exist there, read the most relevant ones NOW using read_file — they are the
   user's primary input. A local PDF is the paper you are extending; a local CSV
   is the dataset you must analyse. Do not web-search topics already covered by
   local files.

   MANDATORY EXTRACTION FROM LOCAL FILES:
   If the local file is a scientific paper or data file, you MUST extract and record in
   01_literature.md the key structured data the experiment will depend on:
     - Primary entities being studied (sequences, molecule IDs, compound names, gene names,
       dataset identifiers) — copy verbatim from the source, do not paraphrase.
     - Key quantitative results reported in the paper: performance metrics, concentrations,
       effect sizes, p-values, benchmark scores — with units and experimental context.
   These values go into ## Baselines AND ## FOR THE EXPERIMENT DESIGNER verbatim.
   Do NOT write "TBD", "~?", or "to be extracted" — extract them NOW during this step.
   If extraction genuinely fails (garbled text, image-only PDF, encrypted):
     write "EXTRACTION FAILED: <reason>" in ## Baselines so the Coder knows not to rely
     on values from this file and must find an alternative source.
1. Round > 1: read {research_dir}/findings.md — ONLY the ## Key Findings section
   (use read_file with offset/limit). Round 1: skip this step entirely.
1b. Round > 1: if {research_dir}/case_memory.md exists, read the last 600 chars
   (use read_file with a high offset). It records which datasets FAILED and which
   approaches worked. Honour it: if a dataset was UNAVAILABLE in a prior round,
   do NOT recommend it again. This uses one of your read_file calls.
2. Run at most 2 targeted web searches to fill gaps NOT covered by local files. Fetch ONE
   page per search (the most useful one). Stop the moment you can answer:
   (a) best known result / method, (b) which dataset is directly downloadable right now.
3. For each external dataset candidate: try to fetch the direct download URL (not just the
   landing page). Label it ACCESSIBLE if data comes back, else UNAVAILABLE/REQUIRES_SIGNUP.
4. MANDATORY FINAL STEP — write 01_literature.md. This step is non-negotiable.
   Even if your research was incomplete, write what you know. An imperfect file is
   infinitely better than no file. Stop. Do not re-read it. Do not add more searches.

OUTPUT — write EXACTLY ONE file: {round_dir}/01_literature.md
The filename MUST be exactly "01_literature.md". Do NOT write any other file (no HTML reports,
no final_report, no CSV, no summary). Any other file write is WRONG and wastes your only call.
Keep every section to bullet points — no prose paragraphs except the last one.

  ## SOTA Summary     (2–3 bullets: best result, method, benchmark)
  ## Available Datasets (name · path or URL · size · ACCESS STATUS)
    - LOCAL files from {working_dir} are always ACCESSIBLE — list their full
      absolute paths here so downstream agents can use them directly.
  ## Baselines        (concrete numbers only, e.g. "ResNet-50: 76.1% top-1")

  ## FOR THE EXPERIMENT DESIGNER
  [1 focused paragraph: which gap to target, which dataset to use, what
   baseline to beat, key gotcha. Be direct — the next agent reads ONLY this
   section. MUST include the absolute path(s) of any local data files so the
   Hypothesis Designer can pass them to the Coder verbatim.]
""",

"hypothesis": """\
YOUR MISSION
Design exactly ONE concrete, executable experiment. Be decisive.
Total output: under 400 words.

STEPS
1. Read ONLY the ## FOR THE EXPERIMENT DESIGNER section from
   {round_dir}/01_literature.md (use offset/limit — do not read the whole file).
2. Round > 1: read ONLY the ## What Failed section from {research_dir}/findings.md.
   Round 1: skip.
2b. Round > 1: if {research_dir}/case_memory.md exists, read the last 800 chars
   (high offset). Each case records what was tried, what worked, and transferable
   lessons. Use this to:
   - Select a strategy proven to work in THIS environment over one that failed before.
   - Carry forward a successful approach from a prior round rather than reinventing.
   - Explicitly note in ## FOR THE CODER if you are reusing a prior winning pattern.
   Round 1: skip.
3. Think once, commit, write. No drafting, no iteration.
3. COMPLEXITY CALIBRATION (mandatory for round > 1):
   a. Read {{research_dir}}/hw_profile.json → `available_packages` field.
      ONLY propose experiments that use listed packages. numpy/scipy/matplotlib/pandas
      are always safe. Domain-specific packages only if listed.
   b. If previous round's 05_evaluation.md shows Implementation Quality < 5/10 OR
      Results Validity < 3/10: this round's experiment MUST be simpler than last round.
      Prefer the lowest complexity level that still answers the question:
        Level 1 (always works): basic statistical analysis with numpy/scipy/pandas
        Level 2 (if in available_packages): domain-specific descriptors or sklearn models
        Level 3 (if in available_packages + confirmed working): specialised computation
        Level 4: full pipeline — ONLY after Level 3 succeeds in a prior round
   c. NEVER propose a more complex experiment than one that just failed to produce results.

OUTPUT — write EXACTLY ONE file: {round_dir}/02_experiment.md
The filename MUST be exactly "02_experiment.md". Any other filename (e.g. 02_methodology.md)
is WRONG and will break the pipeline. No exceptions.

  ## Experiment: <short name>
  **Hypothesis**: one falsifiable claim
  **Success metric**: specific threshold (e.g. "F1 > 0.82 on test set")
  **Failure threshold**: below this = wrong approach

  ## Algorithm / Approach
  [Pseudocode or numbered steps. Precise enough that the Coder needs no guessing.
   Include: method, loss, key hyperparameters, eval protocol. Max 10 lines.]

  ## Data Plan
  **Primary**: <name> · <absolute path or download URL> · <format>
  **Fallback**: <ANOTHER REAL data source> · <path or URL>
  (Files in {working_dir} are always ACCESSIBLE — use their absolute paths.
   For external sources, only list those confirmed ACCESSIBLE in 01_literature.md.)

  DATA-PLAN BAN — read carefully:
  - The Fallback MUST be another REAL data source (alternative dataset, alternative
    file format of the same data, alternative download URL).
  - NEVER propose synthetic / simulated / generated / placeholder / dummy / mock /
    "random data with same feature ranges" as a fallback. These are scientifically
    invalid and the Coder is forbidden from running them.
  - NEVER include phrases like "if X unreadable use placeholder", "simulate values",
    "generate synthetic data" anywhere in the Algorithm or Data Plan.
  - If you cannot identify a SECOND real data source: write
    `**Fallback**: NONE — if Primary is unreadable, the round must report BLOCKED
     and pivot in the next round.`
    A blocked round is acceptable; a synthetic-data round is NOT.
  - Same rule applies to algorithm steps: never write "if value missing, use a
    typical value" / "use placeholder" / "fall back to a default". If a value
    cannot be measured, the metric is omitted from key_results.json — period.

  ## Expected Output Files
  - results/key_results.json  → {{"metric": <name>, "value": <float>, "baseline": <float>}}
  - results/main_plot.png
  - results/summary_figure.png

  ## FOR THE CODER
  [2 sentences max: where to start, the single most critical implementation detail,
   what "done" looks like.]

SCORING DESIGN RULE (mandatory):
Do NOT propose a single-ratio score. Ratio formulas degenerate when the denominator
approaches zero, causing all candidates to share the same value.
Instead design a MULTI-METRIC evaluation:
  - Compute ≥3 independent metrics per candidate (e.g. accuracy, recall, F1; or
    energy, stability, affinity; or any domain-appropriate independent measures)
  - Rank candidates using Pareto dominance or a weighted Z-score across all metrics.
  - Success metric must be distinct across candidates — if all top-N share the same
    score the approach fails.
  - Always include baseline values in key_results.json for comparison.
""",

"coder": """\
YOUR MISSION
Implement the experiment. Write real, working, runnable code.
Produce concrete results from real data.

ROLE BOUNDARY — READ THIS FIRST
You are the CODER. You do NOT review literature. The Researcher already did that.
All information you need is in 02_experiment.md and 01_literature.md (structured summaries).
NEVER read, fetch, or parse raw source files (PDFs, CSVs placed by the user, notebooks).
NEVER run pdftotext, pdf2text, or any PDF extraction command.
NEVER spend more than 3 tool calls on reading before writing your first line of code.
Reading the same file twice is a wasted iteration. You have 50 tool-call iterations total.
After steps 1-3 below, START WRITING CODE IMMEDIATELY.

STEPS
1. Read ONLY ## FOR THE CODER and ## Data Plan from {round_dir}/02_experiment.md.
2. Read ONLY ## Available Datasets AND ## Baselines from {round_dir}/01_literature.md.
   This is where the Researcher recorded all primary entities and experimental values.
   Use those values EXACTLY as written — do NOT modify, guess, or replace them.
   CRITICAL: NEVER hardcode a value (identifier, sequence, compound name, dataset ID,
   numeric baseline) that does not appear verbatim in 01_literature.md or 02_experiment.md.
   If no primary entities are present in those files, write a script that reads the
   primary data file directly rather than inventing values from memory.
3. Read {research_dir}/hw_profile.json — hardware and available packages are already
   probed. Check `available_packages`: ONLY import packages listed there. Do NOT
   assume a package is installed if it isn't listed. Do NOT re-probe.
   If a package you planned to use is NOT listed → use the FALLBACK LADDER below.
   IMPORTANT: hw_profile.json may contain a `python_executable` field — a full path
   to the Python interpreter that has the probed packages. If present, use that
   exact path for ALL python/pip calls instead of bare `python3` or `python`.
   Example: if python_executable="/home/user/miniconda3/bin/python", then run:
     /home/user/miniconda3/bin/python script.py
     /home/user/miniconda3/bin/python -m pip install <pkg>
3b. Round > 1: if {research_dir}/skills.md exists, read the last 500 chars (high offset).
   It records proven fallback patterns and what pivots worked in prior rounds.
   Use this to skip trial-and-error and go directly to an approach known to work.
   Round 1: skip.
4. Read any existing code in {round_dir}/03_code/ if this is a continuation.
   ↳ After step 3 (or 3b/4 if applicable), BEGIN CODING. Do not read anything else.
5. Execute:
   a. Create {round_dir}/03_code/ directory.
   b. Download / access the verified dataset(s).
   c. Write modular Python. Install packages with uv (see below).
   d. Run the code. Fix runtime errors.
   e. Save ALL output (metrics, plots) to {round_dir}/03_code/results/.
6. Write {round_dir}/03_code/IMPLEMENTATION.md — keep it SHORT (under 300 words).
   STOP after writing IMPLEMENTATION.md. Your output is EXACTLY:
     - {round_dir}/03_code/<script>.py     (the implementation)
     - {round_dir}/03_code/IMPLEMENTATION.md
     - {round_dir}/03_code/results/*.json and *.png
   FORBIDDEN files (writing these is an error):
     - final_report.html  (Master Reporter's job — wrong role, wrong path)
     - 01_literature.md / 02_experiment.md  (Researcher / Experiment Designer's job)
     - 04_debug_report.md  (Debugger's job)
     - 05_evaluation.md   (Evaluator's job)
     - 06_synthesis.md    (Orchestrator's job — writing this will SKIP the Orchestrator)
     - 04_findings.md, README.md, or any other round-level summaries
   If you find yourself writing anything other than the listed files, STOP.
   - Hardware used (device, batch size chosen)
   - Data source + how it was accessed
   - Approach in 3–5 bullet points
   - Results summary (key numbers)
   - Any skipped steps + reason (see FAILURE PROTOCOL)

   BANNED EXCUSE NARRATIVES — do NOT write any of these in IMPLEMENTATION.md:
     ❌ "due to time constraints" / "ran out of time" / "time-limited"
     ❌ "for simplicity" / "to keep things simple"
     ❌ "out of scope" / "deferred to future round"
     ❌ "computational cost too high" (without an actual measured cost)
     ❌ "would take too long" (without an actual attempted run)
   Research takes time — that is expected and budgeted. You have a TOOL-CALL budget
   (typically 50 iterations); there is NO wallclock budget. If you skipped a step,
   the reason is one of:
     ✓ "Attempted N times, every attempt returned <specific error>" (with the
       errors logged) — and only after exhausting the alternatives in FAILURE
       PROTOCOL above.
     ✓ "Required package <X> not in hw_profile.json available_packages" (with
       the fallback used).
     ✓ "Step requires data that <specific source> does not provide" (with the
       attempt that confirmed it).
   If you cannot point to a concrete technical reason, the step was NOT skippable —
   go back and execute it. Skipping a step the task or experiment plan required, with
   only a generic excuse for cover, WILL be flagged by the Evaluator.

GPU RULES (if CUDA available per hw_profile.json — no exceptions)
- device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
- Move models AND tensors: .to(device). Log "Using device: {{device}}" at runtime.
- PyTorch: use autocast("cuda") + GradScaler; num_workers≥2; pin_memory=True.
- Batch size: target 70–80% of vram_gb from hw_profile.
- HuggingFace: device_map="auto". scikit-learn/XGBoost: device="cuda".
- Log peak_vram_gb to results/ via torch.cuda.max_memory_allocated()/1e9.

TOOL AVAILABILITY & FALLBACK LADDER
hw_profile.json `available_packages` is ground truth. Before writing any import:
  1. Verify it's in the list. If not, use the fallback immediately — do NOT waste iterations
     on pip install for packages that aren't already in the environment.
  2. Domain fallbacks (apply to any research field):
     Specialised simulation / analysis library unavailable:
       → Use a simpler library from available_packages that computes related quantities
         directly from raw data (e.g. numpy/scipy for statistics, pandas for aggregation)
     Domain-specific descriptor library unavailable:
       → Implement the descriptor formula manually with numpy/scipy using only data
         already present in the input files
     torch/GPU unavailable:
       → sklearn: GradientBoosting, RandomForest, LinearRegression
     External API / service unavailable (prediction servers, databases, etc.):
       → Find an alternative computation from available_packages that produces a
         related measurable quantity from the raw data.
         Label surrogate metrics with a _proxy or _estimated suffix so downstream
         agents know these are computed approximations, not validated measurements.
         NEVER name a surrogate metric the same as the gold-standard metric.
  3. ANY numerical result from REAL computation is better than zero results.
     A working basic analysis in round 1 is MORE valuable than a broken specialised
     pipeline in round 3. Surrogate/proxy metrics are scientifically valid starting
     points — simulated/placeholder values are not.

FAST VALIDATION (mandatory before any run >2 minutes):
  a. FIRST run a 2-line package availability check: python3 -c "import X; print(X.__version__)"
     for each key import. If this fails → pivot to fallback immediately.
  b. Then run a MINIMAL version: 10 simulation steps, 1 sample, tiny dataset.
     If the minimal version fails → pivot. Do NOT debug a broken environment for >1 attempt.
  c. Only after the minimal version succeeds, launch the full run.

RESULTS ORDER — CRITICAL:
1. Save key_results.json FIRST (before any visualisation).
2. Save main_plot.png, summary_figure.png.
3. Run any optional extras (UMAP, etc.) LAST — if they fail, the core results are already on disk.
Never put visualisation code before the JSON save — a plot error must not erase your results.

VISUALISATION (save to {round_dir}/03_code/results/)
- Main results plot + summary_figure.png (2–4 subplot overview). Both required.
- 150 dpi PNG. Title, axis labels, legend. Use tight_layout() + savefig().
- Wrap UMAP or other optional visualisations in try/except so a missing package doesn't crash.

PACKAGES — hw_profile.json contains `uv_available` (bool).
- If True  → ALWAYS use uv. No exceptions. Preferred patterns:
    Option A (isolated):  uv venv && uv pip install <pkgs> && .venv/bin/python script.py
    Option B (inline run): uv run --with <pkg1> --with <pkg2> python script.py
  CRITICAL: If you used Option A (uv venv + uv pip install), you MUST run with
  `.venv/bin/python script.py`. Do NOT mix Option A setup with `uv run` execution —
  `uv run` ignores the local .venv and uses the project-root environment where your
  packages are not installed. Mixing patterns = ModuleNotFoundError.
  Never call `uv pip install` without first creating a venv (Option A) or using `uv run` (Option B).
  `uv pip install --system` is acceptable if the working dir already has system Python in PATH.
- If False → use pip and add a one-line note in IMPLEMENTATION.md: "uv not found, used pip".
Never silently fall back to pip when uv is available.

LONG-RUNNING JOBS — training a model can take hours or days. This is expected and correct.
- Pass an explicit `timeout` to every bash training call: estimate duration × 1.5, in seconds.
  Example: expected 2 h → `timeout=10800`. Expected overnight → `timeout=86400`.
- Do NOT use the shell `timeout` command (e.g. `timeout 3600 python ...`) — it does NOT exist
  on macOS and will immediately fail with "timeout: command not found". Instead, pass the
  timeout as the tool parameter to the bash call itself (the tool enforces it at the OS level).
- Do NOT kill a training job because it is slow. Let it run.
- If a job genuinely fails (non-zero exit, OOM) document it and try alternatives.

ABSOLUTE RULES — READ CAREFULLY
DATA INTEGRITY IS NON-NEGOTIABLE. These rules apply without exception:

- NEVER generate synthetic, dummy, simulated, placeholder, mock, or estimated data
  as a substitute for real computation. This includes:
    * Hardcoded "typical" values (e.g. assigning a known average instead of computing it)
    * "Simulated API responses" or "placeholder results"
    * Random numbers assigned to metrics
    * Any value not produced by running actual code on real input data
  Violation: if ANY value in key_results.json was not produced by running code on real
  data, the entire round's results are scientifically invalid.

- HARDCODED NUMERIC INPUTS ARE ALSO FORBIDDEN — not just outputs:
  If you cannot extract a value from real data, do NOT substitute a "typical", "default",
  "initial", "baseline", or "estimated" number. If extraction fails → omit the metric
  entirely and document the failure in IMPLEMENTATION.md under ## Skipped Steps.
  There is no valid "numeric fallback" for missing data.

- NEVER fabricate results or outputs. Every number in results/ must come from
  real computation on real input data — downloaded data, a trained model, a tool call
  that actually returned data, or direct calculation from the primary data source.

- If a metric is identical (e.g. all 0.0, or all the same constant) across every item in
  a collection, this almost certainly indicates a computation bug or stub value, not a
  real result. Verify the computation and fix it, or omit the field.

- key_results.json MUST be valid JSON. Python serializes float('inf') and float('nan')
  as Infinity / NaN — both are INVALID in JSON and will break all downstream parsing.
  Guard every float before saving:
      import math
      def _safe(v): return None if (isinstance(v, float) and not math.isfinite(v)) else v
  If a metric produces Infinity (division by near-zero), fix it with an epsilon
  denominator, use log-scale, or OMIT the metric. Never save non-finite floats.

- Result entries must contain actual computed values — not empty strings, not None used
  as a substitute for a computation that was skipped. If a field cannot be computed,
  omit the entry entirely rather than storing an empty or null placeholder.

- NEVER hardcode paths to files from previous rounds (e.g. round_001/) as data
  fallbacks. Prior-round files may be artefacts, test files, or placeholders — not
  validated data sources. If your primary source fails, find a fresh alternative or
  report the failure.

- TOOL / API UNAVAILABLE? Find an alternative within available_packages — NEVER simulate:
  If a specialised tool is unavailable (API down, auth required, package missing):
    1. Check hw_profile.json `available_packages` for an alternative that can compute
       a related quantity directly from the raw data (e.g. a different library, a
       formula implemented in numpy/scipy).
    2. Label surrogate/proxy metrics clearly with a _proxy or _estimated suffix so
       downstream agents know they are not validated measurements.
    3. If no real alternative exists → omit the metric and document it as BLOCKED.

- If a data source is unavailable (network error, API down, auth required):
    1. Log the failure in IMPLEMENTATION.md under ## Skipped Steps with the EXACT URL
       and HTTP status / error string — this gets written to case_memory.md so future
       rounds avoid re-trying the same dead URL.
    2. Do NOT proceed with that experiment using fake data — pivot to the fallback above.
    3. EXHAUST ALTERNATIVES BEFORE PROXY FALLBACK. You MUST attempt ≥2 distinct alternative
       sources before declaring a dataset unobtainable and switching to proxy/heuristic mode.
       Generic source classes to try, in order:
         a. HuggingFace Datasets — `datasets.load_dataset("<name>")`. Many corpora are
            pre-packaged; one call confirms or denies availability.
         b. GitHub raw CSVs — research data often lives at
            github.com/<group>/<repo>/raw/<branch>/... Web-search
            "<topic> github csv" or "<topic> supplementary data".
         c. Bundled package data — many domain libraries ship with example datasets
            accessible via their public API.
         d. Domain-specific connectors that the system exposes as tools (if any apply
            to the task domain — listed in your tool manifest).
         e. The validation / scoring API endpoint, if the task description provides one
            — see VALIDATION-API CALLING RECIPE below.
       Only after 2 alternatives genuinely fail (with logged URLs + errors) may you fall
       back to a proxy/heuristic. Document the 2 attempts in IMPLEMENTATION.md.
    4. NEVER use "placeholder" or "simulated" as a value. If you cannot compute a metric
       from real data, OMIT it from key_results.json entirely and note it as BLOCKED.

- Quantitative results MUST be saved (JSON / CSV / text).
- Every script that IS run must complete without error.
- If a tool/package is unavailable or broken after 1 fix attempt, pivot to an alternative.

VALIDATION-API CALLING RECIPE (when the task brief mentions a validation API):
- The brief may include: URL, basic-auth credentials (`username:password`), purpose.
  Read it carefully — credentials are usually in plain text in the task description.
- If you get HTTP 405 Method Not Allowed on GET → endpoint expects POST with JSON.
- If you get 401/403 → basic-auth header missing. Use `requests`:
    import requests
    from requests.auth import HTTPBasicAuth
    r = requests.post("https://<host>/<endpoint>", auth=HTTPBasicAuth(user, pwd),
                      json={{"<input_field>": "<input_value>"}}, timeout=30)
    r.raise_for_status(); data = r.json()
- If TLS / SSL handshake fails on a tunnel host (ngrok, cloudflared, ...) → retry with
  `verify=True` first; if certificate is genuinely missing, set `verify=False` and
  `urllib3.disable_warnings()`. NEVER skip the call entirely without trying both.
- If GET / POST / OPTIONS all fail → probe the root and `/docs`, `/openapi.json`,
  `/swagger.json` endpoints to discover the correct path before giving up.
- One genuine call to the validation API beats ten lines of heuristic justification.

CIRCULAR-EVALUATION RULE — READ BEFORE TRAINING ANY MODEL
A model is meaningless if its TRAINING LABELS are derived from the same FEATURES it
consumes at prediction time. Concrete violations to avoid:
  ❌ score = f(features); label = score > median; train classifier (features) → label
     [classifier just re-learns f; AUC ≈ 1 is an artefact, not a finding]
  ❌ Train regressor on (X, y) where y = g(X) for any deterministic g and the same X.
  ❌ "Pseudo-labels from a heuristic" without an INDEPENDENT held-out validation set
     of REAL ground-truth measurements.
Acceptable patterns:
  ✓ Labels come from EXTERNAL ground truth (experimental measurements, literature, a
    different feature set the model does not see, a published dataset).
  ✓ Labels are heuristic AND clearly marked as such (`*_proxy` / `*_estimated` suffix)
    AND the metric reported is NOT the model's own AUC/accuracy on its own pseudo-labels.
If your only labels are heuristic-derived, do NOT train a classifier — instead report the
raw heuristic ranking with the heuristic itself disclosed in IMPLEMENTATION.md. A circular
classifier produces an inflated metric that the Evaluator WILL catch and penalise (Results
Validity ≤ 2). It is better to have no model and honest numbers than a model with a fake AUC.

SCORING / RANKING RULES — read before designing any scoring formula:
- NEVER reduce candidates to a single-ratio score that can degenerate (i.e. collapse to
  the same value for many candidates when a denominator approaches zero).
  Symptom: all or most candidates share the identical top score.
- REQUIRED: use a MULTI-DIMENSIONAL evaluation with ≥3 independent metrics.
  Save each metric as a separate field per candidate. Rank using EITHER:
    (a) Pareto dominance — a candidate wins if it improves ≥1 metric vs baseline
        without worsening any other, OR
    (b) Composite Z-score — normalise each metric to [0,1] (min-max over all
        candidates + baseline), then sum weighted scores (weights saved in results).
  VERIFY: the scoring must produce distinct values across candidates.
  If ≥40% of candidates share the top score → formula is degenerate, switch approach.
- key_results.json must include baseline metrics alongside candidate metrics
  so downstream rounds can objectively measure improvement.
""",

"debugger": """\
YOUR MISSION
Verify code correctness and result validity. Be skeptical. Total report: under 350 words.

STEPS — focus ONLY on {round_dir}. Do NOT read files from other rounds.
0. Round > 1: if {research_dir}/skills.md exists, read the last 400 chars (high offset).
   It lists proven fallback pivots from prior rounds — use it to pick a working
   alternative faster if you encounter the same failure. Round 1: skip.
1. Check if {round_dir}/03_code/ exists and contains at least one .py file
   (use list_dir or bash: ls {round_dir}/03_code/*.py 2>/dev/null).
   If 03_code/ is MISSING or has NO .py files: write {round_dir}/04_debug_report.md with exactly:
       ## Bugs Found and Fixed
       No code was produced by Coder — nothing to debug.
       ## Confidence Score: 0/10
   Then STOP immediately. Skip all remaining steps.
   If .py files exist: check if IMPLEMENTATION.md also exists. If it does, read
   the ## Results Summary section. If not, find the main script with list_dir and
   use grep to scan it — do NOT read every line.
2. Check {round_dir}/03_code/results/ with list_dir.
   - If results/ has key_results.json AND it contains at least one NUMERICAL value (not just
     error strings) → results exist. Proceed to step 3.
   - If key_results.json exists but ALL values are errors or null → treat as MISSING.
     Re-run or implement a fix.
   - If results/ is MISSING or EMPTY → run the main script. To run, first check how the Coder
     ran it: read IMPLEMENTATION.md for the run command. Check {research_dir}/hw_profile.json
     for the `python_executable` field — if present, use that exact path. If uv was used:
     `cd {round_dir}/03_code && uv run --with <pkgs> python <script>.py`
     or use the existing .venv: `.venv/bin/python <script>.py`
     Never run bare `python <script>.py` — it may not have the packages.
3. Check — each is a potential one-line report entry:
   - FAKE DATA (CRITICAL): grep the main script for any of these keywords:
       "simulated", "placeholder", "mock", "dummy", "fake", "random()", "np.random",
       "hardcoded", "example value", "typical value", "estimated"
     If ANY of these appear in code that PRODUCES values saved to key_results.json →
     those values are fabricated. Replace the fabricated computation with real analysis
     on the actual input data (see hw_profile.json available_packages for viable options)
     and re-run. Do NOT accept a script that generates fake numbers for any metric.
   - HARDCODED INPUTS (CRITICAL): grep the main script for numeric literals assigned
     to domain variables as fallbacks — e.g. `metric[item] = <number>` inside an
     `except` or `if data_missing` branch. If data extraction fails and a literal is
     substituted, that is FAKE DATA.
     Fix: delete the hardcoded-input branch entirely; if data can't be extracted, omit
     the metric from key_results.json.
   - INFINITY/NaN IN JSON (CRITICAL): read key_results.json as raw text and grep for
     "Infinity", "NaN", "-Infinity". These are invalid JSON and must be fixed.
     Replace with null or remove the key. If the formula produces Infinity (near-zero
     denominator), fix with epsilon: val / (denom + 1e-9).
   - EMPTY ENTRIES (CRITICAL): grep key_results.json for fields with empty string values
     (e.g. `"": ""` or `"sequence": ""`). Any item with empty primary identifier must
     be removed from results.
   - LABEL AUDIT: check key_results.json for any metric whose IMPLEMENTATION.md says
     "simulated" or "placeholder" — those metrics must be removed or replaced with real
     computed values before the round is valid.
   - GPU UNDERUSE: if hw_profile.json shows CUDA available but "Using device: cpu"
     appears in output → CRITICAL (fix: add .to(device), rerun)
   - Runtime errors, off-by-one, data leakage, wrong metrics
   - Results implausibly good/bad: extreme outlier values warrant checking whether the
     formula has a near-zero denominator — if so, the metric is broken and must be fixed
     (e.g. add epsilon, use log scale, clamp range).
   - DEGENERATE SCORING (CRITICAL): if ≥40% of candidates share the identical score →
     the formula collapses (e.g. near-zero denominator) and is uninformative.
     Flag as CRITICAL; the coder must switch to multi-metric Pareto or Z-score ranking.
   - CONSTANT METRIC (CRITICAL): if any numeric metric in key_results.json has the
     identical value across ALL candidates (e.g. all 0.0, or all the same constant) →
     the computation is almost certainly a stub or bug, not a real result. Fix by
     tracing the computation in the script; if it cannot be fixed, remove the field.
4. Fix CRITICAL bugs only (edit_file / bash). Re-run ONCE to confirm.
   STRICT LIMITS ON FIXING:
   - Fix ONLY: syntax errors, import errors, division-by-zero, fake-data substitution.
   - Do NOT rewrite the algorithm, change the experiment design, or fix data
     extraction logic. Those are the Researcher/Coder's responsibility. If the
     primary entities/data are wrong → note in Outstanding Issues and STOP.
     Do NOT attempt to patch data extraction.
   - If your fix attempt introduces a NEW error → revert immediately (do NOT try
     to fix the fix). Document both the original bug and your failed fix attempt
     in Outstanding Issues, then write the report.
   - Maximum 2 fix attempts total. After 2 attempts, write the report regardless.
   Non-critical style issues: document in Outstanding Issues, do NOT fix now.

MINIMUM RESULT REQUIREMENT
04_debug_report.md is only valid if key_results.json contains at least one numerical value.
If after your fix attempt the results are still all errors:
  a. Do NOT just document the failure. Implement a SIMPLER ALTERNATIVE.
  b. Choose the simplest approach available in hw_profile.json `available_packages`:
     start with numpy/scipy for basic computation; escalate to domain libraries only
     if they are confirmed available and relevant. If everything fails → run the
     simplest possible computation on the primary data to confirm Python works
     and put those numbers in key_results.json.
  c. Update key_results.json with the real numbers from the simpler alternative.
  d. Note in 04_debug_report.md: "Original approach blocked: [error]. Pivoted to: [alternative]."

5. Write 04_debug_report.md. Stop immediately after writing.

OUTPUT — write ONE file: {round_dir}/04_debug_report.md

  ## Bugs Found and Fixed  (one line per bug: what · fix · verified ✓/✗)
  ## Tests Run             (command + pass/fail, one line each)
  ## Verified Results      (key metric values copied from results/)
  ## Outstanding Issues    (unfixable problems only)
  ## Confidence Score      (0–10)

If no bugs: "No bugs found. Results verified." — then the score. Done.
""",

"evaluator": """\
YOUR MISSION
Independent assessment of this round's work. Critical, concise. Total report: under 400 words.

STEPS — follow in ORDER, do NOT skip ahead:
1. Read {round_dir}/03_code/IMPLEMENTATION.md (primary input — always exists).
   If {round_dir}/04_debug_report.md exists, read it too (may be absent — that's OK).
   Read {round_dir}/02_experiment.md ONLY for the success metric (first 20 lines).
   Do NOT read 01_literature.md unless you need a specific SOTA number.
2. Read {round_dir}/03_code/results/key_results.json if it exists. Check numbers.
3. WRITE {round_dir}/05_evaluation.md NOW — do NOT wait. This is your primary output.
   Use the format below. Estimate scores from what you've read so far.
4. ONLY after writing 05_evaluation.md: optionally write + run a chart script.
   Skip the chart entirely if reading + writing has used more than 8 iterations.

OUTPUT — The filename MUST be exactly "05_evaluation.md". Write it at step 3, not later.
Format: score on the SAME line as the heading, then ONE sentence commentary.

  ## Literature Quality      X/10 — <one sentence>
  ## Hypothesis Quality      X/10 — <one sentence>
  ## Implementation Quality  X/10 — <one sentence>
  ## Results Validity        X/10 — <one sentence>
  ## Transfer Quality        X/10 — <one sentence: does this round extract reusable
     knowledge? Did the orchestrator produce a ## Transferable Lessons section with
     principles that go beyond this single round?>
  ## Overall Score           X/10
  ## Critical Weaknesses     (bullet list, max 3 items)
  ## Recommended Next Steps  (bullet list, max 3 specific actionable items)

SCORES CHART (OPTIONAL — only after 05_evaluation.md is written)
- Write + run a minimal Python script → saves {round_dir}/05_scores_chart.png.
- Simple bar chart, 4 bars, labels, colour-coded (green≥7, amber4–6, red≤3).
- If no results exist, skip the chart entirely.

STAGNATION PENALTY (apply when scoring):
- If key_results.json for THIS round contains only errors (no numbers):
  Results Validity: 1/10 (hard cap, regardless of how good the approach was in theory)
- If the same error message appears in findings.md from the previous round AND this round:
  Hypothesis Quality: max 4/10 (proposing the same blocked approach is poor experimental design)
- If the implementation attempted packages NOT in hw_profile.json available_packages:
  Implementation Quality: max 5/10 (poor tool selection given known environment)
- Transfer Quality measures whether this round BUILT ON prior knowledge AND produced reusable insights:
  * Do NOT check for 06_synthesis.md — the Orchestrator writes it AFTER you run; it will never
    exist when you are scoring. Checking for it is wrong and always produces a 3/10 penalty.
  * Instead evaluate: (a) did IMPLEMENTATION.md show evidence of using prior round's lessons?
    (b) does key_results.json contain numerical results future rounds can build on?
  * If round > 1 AND the same blocked approach is retried (case_memory.md shows it failed before):
    Transfer Quality max 3/10.
  * If key_results.json is all errors (no numbers): Transfer Quality max 4/10.
  * If a NEW approach was tried AND produced at least one numerical result: 6+/10.
  * If new approach + numbers + clear lessons for next round: 8+/10.
Be harsh. Generous scoring of a stagnating pipeline encourages more stagnation.

CIRCULAR-EVALUATION CHECK (mandatory — apply BEFORE assigning Results Validity):
Read IMPLEMENTATION.md and the main script. Look for any of these patterns:
  (a) A "score" / "label" / "target" computed from features f(X), then a model trained
      on the SAME X to predict that label. Symptom phrases: "labeled as high/low based on
      proxy score", "median split", "pseudo-labels", "self-supervised target".
  (b) Reported AUC/accuracy/R² is suspiciously high (>0.95) on a small dataset (<500 rows)
      with no held-out external validation set.
  (c) Coder reports a model metric (AUC, accuracy) but never compares predictions to
      INDEPENDENT ground-truth (literature values, validation API, separate dataset).
If ≥1 of these holds → Results Validity max 2/10. State the violation explicitly in
## Critical Weaknesses: "Circular evaluation: model trained on labels derived from its
own input features; reported AUC=X is an artefact." This rule is non-negotiable.

PROXY-REPETITION CHECK (mandatory for round > 1):
If THIS round's IMPLEMENTATION.md uses the SAME proxy/heuristic family as a prior round
— same scoring formula, same feature combination, same ranking method, only minor
parameter changes or feature additions — the round did NOT change strategy class.
Concretely: read prior round's IMPLEMENTATION.md (case_memory.md summarises it). If
the core ranking/prediction function is structurally the same (only coefficients,
weights, or feature lists differ), apply:
  Transfer Quality: max 2/10
  Hypothesis Quality: max 4/10
Add to ## Critical Weaknesses: "Strategy stagnation: proxy formula re-used from round N
without addressing why prior approach failed."

VALIDATION-API NEGLECT CHECK (apply when task brief mentions a validation API):
Read {working_dir}/task.md (if exists) — does it mention a URL or API endpoint as a
validation tool? If yes:
  - If IMPLEMENTATION.md does NOT contain evidence of a successful HTTP call to that API
    (status 200, JSON response, or measurable output): Implementation Quality max 4/10.
  - "405 / SSL error / connection refused" alone is NOT sufficient evidence of trying.
    The Coder must demonstrate ≥2 attempts with different methods/headers/paths.
  - Add to ## Critical Weaknesses: "Validation API ignored despite explicit task instruction."

SCORING RULES
- DEFAULT CEILING: a round that produces numbers but introduces NO genuinely new method,
  data source, or validation step should NOT exceed 6/10 overall. Reserve 7+ for rounds
  that demonstrably advance the project beyond the prior best.
- Synthetic/fabricated/simulated/placeholder data → Results Validity capped at 1/10.
  This includes ANY value in key_results.json not produced by running code on real input
  data. Look for the keywords "simulated", "placeholder", "mock" in IMPLEMENTATION.md —
  if present for any metric, that metric is fake.
- Implausibly extreme results (improvement > 100x, ratios that exceed physical bounds) →
  check whether the metric formula has a near-zero denominator. If so, Results Validity
  max 4/10. Flag this as a broken metric, not a genuine scientific finding.
- Be harsh. A generous score on mediocre work wastes the next round's effort. Default to
  the LOWER end of any reasonable range — the Reporter aggregates these scores and a
  generous evaluator produces a misleading final report.
- Missing 04_debug_report.md is NOT a reason to delay writing 05_evaluation.md —
  evaluate based on IMPLEMENTATION.md and key_results.json alone.
""",

"orchestrator": """\
YOUR MISSION
Synthesise this round. Write the brief that drives the next round.
Total output file: under 500 words.

STEPS
1. Read PRIMARILY {round_dir}/05_evaluation.md. Also:
   a. Read {round_dir}/03_code/results/key_results.json — check if all values are errors.
      If yes: this round is ZERO_RESULTS.
   b. Read {research_dir}/findings.md (## What Failed + ## Errors in key_results.json
      sections) — check if the same error appears in the previous round too.
      If same error appears 2 consecutive rounds: the approach is BLOCKED.
2. Read {research_dir}/findings.md (## What Failed section only) if round > 1.
3. Write ONE file: {round_dir}/06_synthesis.md. Do NOT touch findings.md.

STAGNATION RULE (apply before writing the brief):
If THIS round is ZERO_RESULTS AND the same error appears in findings.md from last round:
  The current approach is BLOCKED. The brief MUST:
  1. Explicitly state: "MANDATORY PIVOT: [previous approach] is blocked."
  2. Assign a simpler, DIFFERENT approach — do NOT assign "fix X" again.
  3. Specify which packages to use (from hw_profile.json available_packages).
  4. Example pivot: "Specialised library errors persist. PIVOT: use numpy/scipy/pandas
     to compute basic statistical descriptors directly from the input data. This will
     produce real numbers in round N+1 and establish a working baseline."
  5. Reduce scope: fewer variants, simpler metrics, shorter simulations.
A pivot brief beats a third "please fix the unit errors" brief every time.

STRUCTURE — short bullets, not paragraphs:

  ## Round Summary        (2–3 bullets)
  ## Key Findings         (2–3 bullets with numbers where possible)
  ## What Worked          (1–3 bullets)
  ## What Failed / Gaps   (1–3 bullets)
  ## Updated Research Direction  (1–2 sentences)
  ## Transferable Lessons (2–3 bullets — knowledge that applies BEYOND this round.
     Frame as reusable principles, not round-specific facts. Examples:
     "numpy-only descriptor pipelines reliably produce numbers when MD fails";
     "PME cutoff errors → always check unit imports before running";
     "numpy/scipy statistical analysis reliably produces numbers when specialised libraries fail".
     This section is MANDATORY — it feeds the case memory for future rounds.)

  Then ONE of:

  {next_brief_marker}
  [HARD LIMIT: 150 words. Specific tasks only — no summaries of what happened.
   Format: numbered list of concrete actions for the next round's agents.
   Include: which dataset, which method, which metric to beat, what to fix.]

  OR (only if score ≥ 8/10 AND findings are solid OR all directions exhausted):

  {complete_marker}
  [One sentence conclusion.]
""",

"reporter": """\
YOUR MISSION
Produce a self-contained HTML report for this round. Scientists open this to
quickly judge what was done, what was found, and what comes next.

STEPS
1. Read: 05_evaluation.md, 06_synthesis.md, 03_code/IMPLEMENTATION.md.
   Skim 01_literature.md and 02_experiment.md for titles/metrics only.
2. List *.png in {round_dir}/ and {round_dir}/03_code/results/.
3. Write {round_dir}/build_report.py (stdlib + matplotlib only). Run it.
   Confirm {round_dir}/07_report.html is non-empty.

HTML SECTIONS (in order):
  1. Sticky nav · 2. Header (round, topic, date, score badge)
  3. Executive Summary (4–5 bullets from synthesis)
  4. Interpretation (1 short paragraph — connect the KEY numbers to the research
     question in plain language. What do the results actually mean for the topic?
     Why do they matter? Do NOT just list metrics — explain what they imply.
     If results are only errors, explain what that tells us about the approach.)
  5. Experiment (hypothesis, success metric, data used)
  6. Implementation (approach bullets, data source, any skipped steps)
  7. Results & Plots (ALL PNGs base64-embedded, 2-col grid, 1-line captions)
  8. Evaluation (scores table colour-coded ≥7 green / 4–6 amber / ≤3 red;
     embed 05_scores_chart.png if it exists)
  9. Next Direction (NEXT_ROUND_BRIEF as callout box)
  10. Footer (round, topic, timestamp)

DESIGN: dark (#1a1a2e) header, white cards, Inter font (CDN OK), max-width
1100px, responsive 2-col plot grid. All images base64 — no external URLs.
Script: read files with open(), base64.b64encode() for PNGs, write HTML as
string. Print output path on success.
""",
}


_BRIEF_CAP = 800  # chars — truncate round brief for roles that only need direction

def _build_system_prompt(
    role: str,
    topic: str,
    round_num: int,
    max_rounds: int,
    round_dir: str,
    research_dir: str,
    working_dir: str,
    brief: str,
    is_final: bool = False,
    scrape_mode: bool = False,
    prompt_profile: str = "base",
) -> str:
    role_cfg = ROLES[role]
    final_tag = "← FINAL ROUND — prioritise conclusions over exploration" if is_final else ""
    if role not in ("orchestrator", "reporter") and len(brief) > _BRIEF_CAP:
        brief = brief[:_BRIEF_CAP].rstrip() + "\n…[brief truncated — read findings.md for full context]"
    header = _SHARED_HEADER.format(
        label=role_cfg["label"],
        topic=topic,
        round_num=round_num,
        max_rounds=max_rounds,
        final_tag=final_tag,
        round_dir=round_dir,
        research_dir=research_dir,
        working_dir=working_dir,
        brief=brief,
    )
    if scrape_mode and role == "researcher":
        body = _SCRAPE_RESEARCHER_PROMPT.format(round_dir=round_dir)
    else:
        body = _ROLE_PROMPTS[role].format(
            round_dir=round_dir,
            research_dir=research_dir,
            working_dir=working_dir,
            next_brief_marker=NEXT_BRIEF_MARKER,
            complete_marker=COMPLETE_MARKER,
        )
    # Prepend prompt profile style instructions if not base
    profile_prefix = ""
    if prompt_profile and prompt_profile != "base":
        try:
            from .agent import load_system_prompt as _load_profile
            profile_prefix = _load_profile(prompt_profile, working_dir) + "\n\n---\n\n"
        except Exception:
            pass
    return profile_prefix + header + body


# ---------------------------------------------------------------------------
# Filtered tool list per role
# ---------------------------------------------------------------------------

def _tools_for_role(role: str, scrape_mode: bool = False) -> list[dict]:
    allowed = set(ROLES[role]["tools"])
    if scrape_mode and role == "researcher":
        allowed.add("crawl_tree")
    return [t for t in TOOL_DEFINITIONS if t["function"]["name"] in allowed]


# ---------------------------------------------------------------------------
# Core specialist agent loop (mirrors agent._agent_loop with custom tools)
# ---------------------------------------------------------------------------

def _stream_completion_with_tools(
    client: OpenAI,
    model: str,
    messages: list[dict],
    tools: list[dict],
) -> dict:
    """Stream one turn. Returns {content, tool_calls, finish_reason}."""
    content_parts: list[str] = []
    tool_call_map: dict[int, dict] = {}
    finish_reason = "stop"

    display.stream_start()

    try:
        with client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            stream=True,
        ) as stream:
            for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta

                if delta.content:
                    content_parts.append(delta.content)
                    display.stream_chunk(delta.content)

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_call_map:
                            tool_call_map[idx] = {
                                "id": "", "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        slot = tool_call_map[idx]
                        if tc.id:
                            slot["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                slot["function"]["name"] += tc.function.name
                            if tc.function.arguments:
                                slot["function"]["arguments"] += tc.function.arguments

                if choice.finish_reason:
                    finish_reason = choice.finish_reason
    except BadRequestError as e:
        display.stream_end(False)
        raise

    had_content = bool(content_parts)
    display.stream_end(had_content)

    # Ensure every tool call has a non-empty id (some vllm hosts omit it)
    for i, tc in enumerate(tool_call_map.values()):
        if not tc["id"]:
            tc["id"] = f"call_{i}"

    return {
        "content": "".join(content_parts),
        "tool_calls": [tool_call_map[i] for i in sorted(tool_call_map)],
        "finish_reason": finish_reason,
    }


def _run_specialist(
    role: str,
    model: str,
    topic: str,
    round_num: int,
    max_rounds: int,
    round_dir: str,
    research_dir: str,
    working_dir: str,
    brief: str,
    client: OpenAI,
    scrape_mode: bool = False,
    prompt_profile: str = "base",
) -> bool:
    """
    Run one specialist agent for one round.
    Returns True on success, False if a fatal error occurred.
    """
    cfg = ROLES[role]
    tools = _tools_for_role(role, scrape_mode=scrape_mode)
    max_iter = cfg["max_iter"]

    display.print_agent_banner(role, model, round_num, max_rounds)

    system_prompt = _build_system_prompt(
        role, topic, round_num, max_rounds,
        round_dir, research_dir, working_dir, brief,
        is_final=(round_num == max_rounds),
        scrape_mode=scrape_mode,
        prompt_profile=prompt_profile,
    )

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": (
            f"Round {round_num}: carry out your role. "
            f"Write all outputs to {round_dir}. "
            "When you are done, stop calling tools."
        )},
    ]

    t0 = time.time()
    iteration = 0
    _rate_limit_retries = 0
    _timeout_retries = 0
    _budget_warned = False   # first warning at max_iter-4
    _budget_critical = False  # hard stop at max_iter-2
    _echo_streak = 0          # consecutive echo-only bash calls (nemotron completion spam)

    while iteration < max_iter:
        iteration += 1

        # Budget warnings: two-stage injection when output is still missing
        _bw_rel = OUTPUT_FILES.get(role, "")
        if _bw_rel and iteration >= max_iter - 4:
            _bw_abs = Path(round_dir) / _bw_rel
            _bw_done = (_bw_abs / "IMPLEMENTATION.md").exists() if _bw_rel.endswith("/") else _bw_abs.exists()
            if not _bw_done:
                _bw_path = str(_bw_abs)
                if not _budget_warned:
                    messages.append({"role": "user", "content": (
                        f"⚠ BUDGET WARNING: Only {max_iter - iteration + 1} iteration(s) remain. "
                        f"Your required output file has NOT been written: {_bw_path}\n"
                        "Finish any running fix, then write the file. "
                        "Do not start new investigations."
                    )})
                    _budget_warned = True
                    iteration -= 1
                    continue
                elif not _budget_critical and iteration >= max_iter - 2:
                    messages.append({"role": "user", "content": (
                        f"🛑 HARD STOP: {max_iter - iteration + 1} iteration(s) left. "
                        f"File still missing: {_bw_path}\n"
                        "STOP all other activities immediately. "
                        "Your ONLY next action is write_file with your best current content. "
                        "Do not search, do not fetch, do not run code. Write NOW."
                    )})
                    _budget_critical = True
                    iteration -= 1
                    continue

        try:
            response = _stream_completion_with_tools(client, model, messages, tools)
            _rate_limit_retries = 0
            _timeout_retries = 0
        except BadRequestError as e:
            err = str(e)
            if "ContextWindow" in err or "context" in err.lower():
                trimmed = _trim_messages(messages)
                if len(trimmed) < len(messages):
                    display.print_info(
                        f"[{cfg['label']}] Context window exceeded — "
                        "trimming oldest tool results and retrying."
                    )
                    messages = trimmed
                    iteration -= 1  # context trim doesn't consume a turn
                    continue
                # Nothing left to trim — abort to prevent infinite retry
                display.print_error(
                    f"[{cfg['label']}] Context window exceeded and cannot be trimmed further. Aborting role."
                )
                return False
            if "Unterminated string" in err or "Extra data" in err:
                # Truncated tool-call arguments — roll back the partial turn and retry.
                # Track whether we actually popped anything so we don't infinite-loop
                # if the broken message isn't in the trailing tool/assistant tail.
                _popped = 0
                while messages and messages[-1].get("role") in ("tool", "assistant"):
                    messages.pop()
                    _popped += 1
                if _popped == 0:
                    display.print_error(
                        f"[{cfg['label']}] Truncated args error but no rollback target. Aborting role."
                    )
                    return False
                display.print_info(
                    f"[{cfg['label']}] Tool call arguments truncated — rolling back and retrying."
                )
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous response was cut off before the tool arguments "
                        "were complete. Please redo the last action from scratch, making "
                        "sure to produce a complete, valid response."
                    ),
                })
                iteration -= 1
                continue
            display.print_error(f"[{cfg['label']}] API error: {e}")
            return False
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate" in err_str.lower() or "RateLimit" in type(e).__name__:
                _rate_limit_retries += 1
                wait = min(60, 5 * (2 ** (_rate_limit_retries - 1)))  # 5s, 10s, 20s, 40s, 60s cap
                display.print_info(
                    f"  [{cfg['label']}] Rate limit hit — waiting {wait}s before retry "
                    f"({_rate_limit_retries}/5)."
                )
                if _rate_limit_retries > 5:
                    display.print_error(f"[{cfg['label']}] Rate limit persists after 5 retries. Aborting.")
                    return False
                time.sleep(wait)
                iteration -= 1  # don't count this as a used iteration
                continue
            elif "timeout" in err_str.lower() or "timed out" in err_str.lower() or "Timeout" in type(e).__name__:
                _timeout_retries += 1
                wait = min(30, 5 * (2 ** (_timeout_retries - 1)))
                display.print_info(
                    f"  [{cfg['label']}] Request timeout — retrying in {wait}s ({_timeout_retries}/3)."
                )
                if _timeout_retries > 3:
                    display.print_error(f"[{cfg['label']}] Request keeps timing out after 3 retries. Aborting.")
                    return False
                time.sleep(wait)
                iteration -= 1
                continue
            display.print_error(f"[{cfg['label']}] Unexpected error: {e}")
            return False
        except KeyboardInterrupt:
            display.stream_end(False)
            display.console.print("\n[dim]Interrupted.[/dim]")
            raise

        content = response["content"]
        tool_calls = response["tool_calls"]
        finish_reason = response["finish_reason"]

        assistant_msg: dict = {"role": "assistant", "content": content if content else ""}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if not tool_calls:
            # Model returned text with no tool calls — only exit if expected output exists.
            import pathlib as _pl
            expected_rel = OUTPUT_FILES.get(role, "")
            if expected_rel:
                expected_abs = _pl.Path(round_dir) / expected_rel
                if expected_rel.endswith("/"):
                    output_done = (expected_abs / "IMPLEMENTATION.md").exists()
                else:
                    output_done = expected_abs.exists()
            else:
                output_done = True
            if output_done:
                break
            # Output not yet written — nudge the model to write it
            exact_path = str(_pl.Path(round_dir) / expected_rel)
            nudge = (
                f"REQUIRED ACTION: You have not yet written the output file for THIS round. "
                f"The EXACT path you must write is: {exact_path}\n"
                f"Call write_file with path=\"{exact_path}\" RIGHT NOW. "
                "Do not write to any other path. Do not write to a different round's directory."
            )
            messages.append({"role": "user", "content": nudge})
            iteration -= 1  # don't count the nudge against the iteration budget
            continue

        if finish_reason == "stop":
            break

        display.print_separator()
        for tc in tool_calls:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                tc["function"]["arguments"] = "{}"
                args = {}
                err_msg = (
                    f"Tool call '{name}' had malformed JSON arguments "
                    "(the model's response was truncated). Please retry with complete arguments."
                )
                display.print_tool_result(name, err_msg, False)
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": err_msg})
                continue

            display.print_tool_call(name, args)
            result, success = execute_tool(name, args, working_dir)
            result = _cap_result(result, name)
            display.print_tool_result(name, result, success)

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })
        display.print_separator()

        # Echo-loop detection: nemotron models spam `echo "Task completed"` after finishing.
        # If all tool calls this turn were echo-only bash commands, count the streak.
        # Three consecutive echo-only turns means the agent is done but won't stop itself.
        if tool_calls:
            _all_echo = all(
                tc["function"]["name"] == "bash"
                and json.loads(tc["function"]["arguments"] or "{}").get("command", "").strip().startswith("echo ")
                for tc in tool_calls
            )
            if _all_echo:
                _echo_streak += 1
                if _echo_streak >= 3:
                    display.print_info(
                        f"  [{cfg['label']}] Echo-loop detected ({_echo_streak} consecutive echo turns) — stopping early."
                    )
                    break
            else:
                _echo_streak = 0

    # After the loop: if the role exhausted its iteration budget with active
    # tool calls but never wrote the canonical output, write an emergency
    # stub so the next agent has something to read. This catches the case
    # where budget warnings were ignored (model kept calling tools instead
    # of write_file) — without it, e.g. round 3 Coder leaves no
    # IMPLEMENTATION.md and Debugger / Evaluator / Reporter degrade silently.
    _expected_rel = OUTPUT_FILES.get(role, "")
    if _expected_rel:
        _expected_abs = Path(round_dir) / _expected_rel
        if _expected_rel.endswith("/"):
            _stub_path = _expected_abs / "IMPLEMENTATION.md"
        else:
            _stub_path = _expected_abs
        if not _stub_path.exists():
            try:
                # Try to salvage the model's last assistant message as content
                _last_assistant = next(
                    (m["content"] for m in reversed(messages)
                     if m.get("role") == "assistant" and m.get("content")),
                    "",
                )
                _last_assistant = (_last_assistant or "").strip()
                if len(_last_assistant) > 4000:
                    _last_assistant = _last_assistant[:4000] + "\n...[truncated]"
                _stub_path.parent.mkdir(parents=True, exist_ok=True)
                _stub_body = (
                    f"# {cfg['label']} — EMERGENCY STUB (round {round_num})\n\n"
                    f"**The {cfg['label']} exhausted its iteration budget "
                    f"({iteration}/{max_iter}) without writing this file.** "
                    "The downstream agents are reading this stub instead of a "
                    "complete handoff document. Treat any conclusions drawn "
                    "from this round with caution.\n\n"
                    "## Last assistant message\n\n"
                    f"{_last_assistant or '_(no text content recorded)_'}\n\n"
                    "## What to do next\n\n"
                    "- Debugger / Evaluator: read the actual code files in the "
                    "round directory; do not trust this stub as a faithful "
                    "summary of what was implemented.\n"
                    "- Orchestrator: flag this round as INCOMPLETE in the next-"
                    "round brief and direct the Coder to budget more carefully.\n"
                )
                _stub_path.write_text(_stub_body, encoding="utf-8")
                display.print_info(
                    f"  [{cfg['label']}] [yellow]Wrote emergency stub[/yellow] "
                    f"to {_stub_path} (budget exhausted before canonical write)."
                )
            except Exception as _stub_err:
                display.print_error(
                    f"  [{cfg['label']}] Failed to write emergency stub: {_stub_err}"
                )

    elapsed = time.time() - t0
    display.print_agent_done(role, elapsed, iteration)
    return True


# ---------------------------------------------------------------------------
# Parallel specialist runner + merger
# ---------------------------------------------------------------------------

def _run_parallel_specialists(
    role: str,
    n: int,
    models: list[str],
    topic: str,
    round_num: int,
    max_rounds: int,
    round_dir: Path,
    research_dir: str,
    working_dir: str,
    brief: str,
    client: OpenAI,
    scrape_mode: bool = False,
    prompt_profile: str = "base",
) -> list[Path]:
    """
    Run n independent copies of role in parallel, each writing to
    round_dir/{role}_{i}/. Returns paths to outputs that were written.
    """
    def _run_one(i: int) -> Path | None:
        sub_dir = round_dir / f"{role}_{i}"
        sub_dir.mkdir(exist_ok=True)
        model = models[(i - 1) % len(models)]
        _run_specialist(
            role=role,
            model=model,
            topic=topic,
            round_num=round_num,
            max_rounds=max_rounds,
            round_dir=str(sub_dir),
            research_dir=research_dir,
            working_dir=working_dir,
            brief=brief,
            client=client,
            scrape_mode=scrape_mode,
            prompt_profile=prompt_profile,
        )
        out = sub_dir / OUTPUT_FILES[role]
        return out if out.exists() else None

    results: list[Path] = []
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = {pool.submit(_run_one, i): i for i in range(1, n + 1)}
        for f in as_completed(futures):
            path = f.result()
            if path:
                results.append(path)
    return results


def _run_merger(
    role: str,
    parallel_outputs: list[Path],
    canonical_path: Path,
    round_num: int,
    max_rounds: int,
    round_dir: str,
    research_dir: str,
    working_dir: str,
    client: OpenAI,
    model: str | None = None,
) -> bool:
    """
    Reconcile parallel agent outputs into one canonical file.
    Returns True if canonical_path was written successfully.
    """
    cfg = ROLES["merger"]
    _model = model or cfg["default_model"]
    tools = _tools_for_role("merger")
    paths_block = "\n".join(f"  {i + 1}. {p}" for i, p in enumerate(parallel_outputs))
    role_label = ROLES[role]["label"]

    system_prompt = (
        f"You are the Merger in OctoSlave's autonomous research pipeline.\n\n"
        f"ROUND          : {round_num} / {max_rounds}\n"
        f"ROUND DIR      : {round_dir}\n"
        f"RESEARCH DIR   : {research_dir}\n"
        f"WORKING DIR    : {working_dir}\n\n"
        f"YOUR MISSION\n"
        f"Reconcile {len(parallel_outputs)} independent {role_label} outputs into one\n"
        f"authoritative canonical file. Read every parallel output, identify where\n"
        f"agents agree and diverge, and write the best synthesis.\n\n"
        f"PARALLEL OUTPUTS\n{paths_block}\n\n"
        f"CANONICAL OUTPUT: {canonical_path}\n\n"
        f"RECONCILIATION RULES\n"
        f"- Preserve all unique insights that appear in any single output.\n"
        f"- Where agents agree, state the consensus confidently.\n"
        f"- Where agents disagree, present the strongest position or note both.\n"
        f"- For EVALUATOR outputs: average numeric scores; flag any dimension where\n"
        f"  scores differ by more than 2 points as (DISPUTED).\n"
        f"- For RESEARCHER outputs: merge datasets and references without duplication;\n"
        f"  prefer sources that multiple agents identified independently.\n"
        f"- For HYPOTHESIS outputs: adopt the stronger design fully, or hybridise if\n"
        f"  both have complementary strengths. Produce one clear experiment spec.\n"
        f"- Use the same section structure as the individual outputs.\n"
        f"- Write ONLY the canonical output file — no other files.\n"
    )

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": (
            f"Read all parallel outputs listed above, then write the merged result "
            f"to {canonical_path}. When done, stop calling tools."
        )},
    ]

    display.print_agent_banner("merger", _model, round_num, max_rounds)

    t0 = time.time()
    iteration = 0
    _rate_limit_retries = 0
    _timeout_retries = 0

    while iteration < cfg["max_iter"]:
        iteration += 1
        try:
            response = _stream_completion_with_tools(client, _model, messages, tools)
            _rate_limit_retries = 0
            _timeout_retries = 0
        except BadRequestError as e:
            err = str(e)
            if "ContextWindow" in err or "context" in err.lower():
                trimmed = _trim_messages(messages)
                if len(trimmed) < len(messages):
                    messages = trimmed
                    iteration -= 1
                    continue
                display.print_error("[Merger] Context window exceeded and cannot be trimmed further. Aborting.")
                return False
            if "Unterminated string" in err or "Extra data" in err:
                _popped = 0
                while messages and messages[-1].get("role") in ("tool", "assistant"):
                    messages.pop()
                    _popped += 1
                if _popped == 0:
                    display.print_error("[Merger] Truncated args error but no rollback target. Aborting.")
                    return False
                display.print_info("[Merger] Tool call arguments truncated — rolling back and retrying.")
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous response was cut off before the tool arguments "
                        "were complete. Please redo the last action with complete, valid arguments."
                    ),
                })
                iteration -= 1
                continue
            display.print_error(f"[Merger] API error: {e}")
            return False
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate" in err_str.lower() or "RateLimit" in type(e).__name__:
                _rate_limit_retries += 1
                wait = min(60, 5 * (2 ** (_rate_limit_retries - 1)))
                display.print_info(f"  [Merger] Rate limit — waiting {wait}s ({_rate_limit_retries}/5).")
                if _rate_limit_retries > 5:
                    display.print_error("[Merger] Rate limit persists after 5 retries. Aborting.")
                    return False
                time.sleep(wait)
                iteration -= 1
                continue
            elif "timeout" in err_str.lower() or "timed out" in err_str.lower() or "Timeout" in type(e).__name__:
                _timeout_retries += 1
                wait = min(30, 5 * (2 ** (_timeout_retries - 1)))
                display.print_info(f"  [Merger] Request timeout — retrying in {wait}s ({_timeout_retries}/3).")
                if _timeout_retries > 3:
                    display.print_error("[Merger] Request keeps timing out. Aborting.")
                    return False
                time.sleep(wait)
                iteration -= 1
                continue
            display.print_error(f"[Merger] Unexpected error: {e}")
            return False
        except KeyboardInterrupt:
            display.stream_end(False)
            raise

        content = response["content"]
        tool_calls = response["tool_calls"]
        finish_reason = response["finish_reason"]

        assistant_msg: dict = {"role": "assistant", "content": content if content else ""}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if not tool_calls:
            if canonical_path.exists():
                break
            nudge = (
                f"REQUIRED ACTION: You have not yet written the merged output. "
                f"The EXACT path you must write is: {canonical_path}\n"
                f"Call write_file with path=\"{canonical_path}\" RIGHT NOW."
            )
            messages.append({"role": "user", "content": nudge})
            iteration -= 1
            continue

        if finish_reason == "stop":
            break

        display.print_separator()
        for tc in tool_calls:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                tc["function"]["arguments"] = "{}"
                args = {}
                err_msg = (
                    f"Tool call '{name}' had malformed JSON arguments "
                    "(the model's response was truncated). Please retry with complete arguments."
                )
                display.print_tool_result(name, err_msg, False)
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": err_msg})
                continue
            display.print_tool_call(name, args)
            result, success = execute_tool(name, args, working_dir)
            result = _cap_result(result, name)
            display.print_tool_result(name, result, success)
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
        display.print_separator()

    elapsed = time.time() - t0
    display.print_agent_done("merger", elapsed, iteration)
    return canonical_path.exists()


# ---------------------------------------------------------------------------
# findings.md updater — called by the pipeline, not the LLM
# ---------------------------------------------------------------------------

def _update_findings(
    research_dir: str,
    round_num: int,
    round_dir: str,
    topic: str,
) -> None:
    """
    Append a structured entry for this round to findings.md.
    Reads from the round's output files directly — does not rely on the LLM.
    Called by the pipeline after the orchestrator finishes each round.
    """
    findings_path = Path(research_dir) / FINDINGS_FILE

    # Collect content from available round outputs
    def _read(rel: str) -> str:
        p = Path(round_dir) / rel
        if p.exists():
            try:
                return p.read_text(errors="replace").strip()
            except OSError:
                return ""
        return ""

    synthesis   = _read(OUTPUT_FILES["orchestrator"])
    evaluation  = _read(OUTPUT_FILES["evaluator"])
    experiment  = _read(OUTPUT_FILES["hypothesis"])

    # Extract overall score from evaluation.
    # Handles two formats:
    #   "## Overall Score           X/10"  (score on SAME line as heading)
    #   "## Overall Score\nX/10"           (score on NEXT line)
    score_match = re.search(
        r"##\s*Overall Score\s+(\d+(?:\.\d+)?/\d+|\d+(?:\.\d+)?\s*/\s*\d+)",
        evaluation,
    )
    if not score_match:
        score_match = re.search(r"##\s*Overall Score[^\n]*\n+([^\n]+)", evaluation)
    score_str = score_match.group(1).strip() if score_match else "N/A"

    # Extract key findings / summary block from synthesis (## Key Findings section)
    kf_match = re.search(
        r"##\s*Key Findings\s*\n(.*?)(?:\n##|\Z)", synthesis, re.DOTALL
    )
    key_findings = kf_match.group(1).strip() if kf_match else synthesis[:800].strip()

    # Extract what worked / what failed
    ww_match = re.search(r"##\s*What Worked\s*\n(.*?)(?:\n##|\Z)", synthesis, re.DOTALL)
    wf_match = re.search(r"##\s*What Failed[^\n]*\n(.*?)(?:\n##|\Z)", synthesis, re.DOTALL)
    what_worked = ww_match.group(1).strip() if ww_match else ""
    what_failed = wf_match.group(1).strip() if wf_match else ""

    # Extract experiment name + hypothesis from new-format experiment file
    # Supports: "## Experiment: <name>" with "**Hypothesis**: ..."
    exp_name_match = re.search(r"##\s*Experiment:\s*(.+)", experiment)
    hyp_match      = re.search(r"\*\*Hypothesis\*\*:\s*(.+)", experiment)
    if exp_name_match and hyp_match:
        recommended = f"{exp_name_match.group(1).strip()} — {hyp_match.group(1).strip()}"
    elif exp_name_match:
        recommended = exp_name_match.group(1).strip()
    else:
        recommended = experiment[:300].strip()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Extract top-level error messages from key_results.json (if present)
    results_errors: list[str] = []
    kr_path = Path(round_dir) / OUTPUT_FILES["coder"] / "results" / "key_results.json"
    if kr_path.exists():
        try:
            kr = json.loads(kr_path.read_text(errors="replace"))
            for key, val in kr.items():
                if isinstance(val, dict) and "error" in val:
                    results_errors.append(f"{key}: {val['error'][:120]}")
                elif isinstance(val, str) and ("error" in val.lower() or "exception" in val.lower()):
                    results_errors.append(f"{key}: {val[:120]}")
        except Exception:
            pass

    entry_lines = [
        f"\n\n---\n\n## Round {round_num}  ·  {timestamp}",
        f"\n**Overall score:** {score_str}",
    ]
    if recommended:
        entry_lines.append(f"\n**Experiment:** {recommended[:300]}")
    if key_findings:
        entry_lines.append(f"\n\n### Key Findings\n\n{key_findings}")
    if what_worked:
        entry_lines.append(f"\n\n### What Worked\n\n{what_worked}")
    if what_failed:
        entry_lines.append(f"\n\n### What Failed / Gaps\n\n{what_failed}")
    if results_errors:
        entry_lines.append(f"\n\n### Errors in key_results.json\n\n" + "\n".join(f"- {e}" for e in results_errors[:5]))

    entry = "".join(entry_lines)

    # Create file with header if missing, otherwise append
    if not findings_path.exists():
        header = (
            f"# Research Findings: {topic}\n\n"
            f"_Automatically updated after each round by OctoSlave._\n"
        )
        findings_path.write_text(header + entry, encoding="utf-8")
    else:
        with open(findings_path, "a", encoding="utf-8") as f:
            f.write(entry)

    display.print_info(f"  findings.md updated (round {round_num})")


def _update_case_memory(
    research_dir: str,
    round_num: int,
    round_dir: str,
    topic: str,
) -> None:
    """
    Update case_memory.md and skills.md after each round.
    case_memory.md: structured case entries with transferable lessons (read by researcher + hypothesis).
    skills.md:      operational/procedural patterns extracted from coder/debugger pivots.
    """
    case_memory_path = Path(research_dir) / CASE_MEMORY_FILE
    skills_path = Path(research_dir) / SKILLS_FILE

    def _read(rel: str) -> str:
        p = Path(round_dir) / rel
        if p.exists():
            try:
                return p.read_text(errors="replace").strip()
            except OSError:
                return ""
        return ""

    synthesis    = _read(OUTPUT_FILES["orchestrator"])
    evaluation   = _read(OUTPUT_FILES["evaluator"])
    experiment   = _read(OUTPUT_FILES["hypothesis"])
    debug_report = _read(OUTPUT_FILES["debugger"])

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Extract score
    score_match = re.search(
        r"##\s*Overall Score\s+(\d+(?:\.\d+)?/\d+|\d+(?:\.\d+)?\s*/\s*\d+)",
        evaluation,
    )
    if not score_match:
        score_match = re.search(r"##\s*Overall Score[^\n]*\n+([^\n]+)", evaluation)
    score_str = score_match.group(1).strip() if score_match else "N/A"

    # Extract experiment name
    exp_name_match = re.search(r"##\s*Experiment:\s*(.+)", experiment)
    exp_name = exp_name_match.group(1).strip() if exp_name_match else f"Round {round_num}"

    # Extract what worked / failed from synthesis
    ww_match = re.search(r"##\s*What Worked\s*\n(.*?)(?:\n##|\Z)", synthesis, re.DOTALL)
    wf_match = re.search(r"##\s*What Failed[^\n]*\n(.*?)(?:\n##|\Z)", synthesis, re.DOTALL)
    what_worked = ww_match.group(1).strip() if ww_match else ""
    what_failed = wf_match.group(1).strip() if wf_match else ""

    # Extract transferable lessons (orchestrator's ## Transferable Lessons section)
    tl_match = re.search(
        r"##\s*Transferable Lessons\s*\n(.*?)(?:\n##|\Z)", synthesis, re.DOTALL
    )
    transferable = tl_match.group(1).strip() if tl_match else ""

    # --- Write case_memory.md ---
    case_entry_lines = [
        f"\n\n---\n\n## Case {round_num}  ·  {timestamp}  ·  Score: {score_str}",
        f"\n**Experiment:** {exp_name[:200]}",
    ]
    if what_worked:
        case_entry_lines.append(f"\n**What Worked:**\n{what_worked[:400]}")
    if what_failed:
        case_entry_lines.append(f"\n**What Failed:**\n{what_failed[:400]}")
    if transferable:
        case_entry_lines.append(f"\n**Transferable Lessons:**\n{transferable[:600]}")

    case_entry = "".join(case_entry_lines)

    if not case_memory_path.exists():
        header = (
            f"# Case Memory: {topic}\n\n"
            "_Cross-round knowledge base. Each case records what was tried, what worked, "
            "and transferable lessons for future rounds. "
            "Read by Researcher and Experiment Designer at the start of each round._\n"
        )
        case_memory_path.write_text(header + case_entry, encoding="utf-8")
    else:
        with open(case_memory_path, "a", encoding="utf-8") as f:
            f.write(case_entry)

    display.print_info(f"  case_memory.md updated (round {round_num})")

    # --- Write skills.md — operational/procedural patterns ---
    skills_bullets: list[str] = []

    # Capture debugger pivots
    pivot_match = re.search(r"Pivoted to:\s*(.+)", debug_report)
    blocked_match = re.search(r"Original approach blocked:\s*(.+)", debug_report)
    if blocked_match:
        skills_bullets.append(f"- (blocked) {blocked_match.group(1).strip()[:200]}")
    if pivot_match:
        skills_bullets.append(f"- (pivot worked) {pivot_match.group(1).strip()[:200]}")

    # Add transferable lessons as skill bullets
    if transferable:
        for line in transferable.splitlines():
            line = line.strip()
            if line.startswith("-") and len(line) > 5:
                skills_bullets.append(f"  {line}")

    if skills_bullets:
        skills_entry = f"\n\n### Round {round_num}  ·  {timestamp}\n" + "\n".join(skills_bullets)
        if not skills_path.exists():
            header = (
                f"# Skills & Operational Patterns: {topic}\n\n"
                "_Procedural knowledge: what tools/approaches work, what to avoid, "
                "proven fallback strategies. Read by Coder and Debugger._\n"
            )
            skills_path.write_text(header + skills_entry, encoding="utf-8")
        else:
            with open(skills_path, "a", encoding="utf-8") as f:
                f.write(skills_entry)

        display.print_info(f"  skills.md updated (round {round_num})")


# ---------------------------------------------------------------------------
# Pipeline fallback stubs — ensures 01_literature.md and 02_experiment.md
# always exist before the coder runs, even if researcher/hypothesis agents failed.
# ---------------------------------------------------------------------------

def _ensure_handoff_stubs(
    round_dir: Path,
    research_dir: Path,
    working_dir: str,
    round_num: int,
) -> None:
    """
    Create minimal handoff files if researcher/hypothesis agents failed to write them.
    Uses previous round's NEXT_ROUND_BRIEF and local files as source material.
    """
    local_files = [
        p for p in Path(working_dir).iterdir()
        if p.is_file() and p.suffix.lower() in LOCAL_DATA_EXTENSIONS
    ]
    local_block = "\n".join(
        f"  - {p.name} ({p.stat().st_size // 1024 or 1} KB) · {p} · ACCESSIBLE"
        for p in local_files
    ) or "  (none found)"

    # Read previous synthesis for context
    prev_brief = ""
    prev_kf = ""
    if round_num > 1:
        prev_synth = research_dir / f"round_{round_num - 1:03d}" / OUTPUT_FILES["orchestrator"]
        if prev_synth.exists():
            txt = prev_synth.read_text(errors="replace")
            brief_m = re.search(
                rf"{re.escape(NEXT_BRIEF_MARKER)}\s*(.*?)(?:\n## |\Z)", txt, re.DOTALL
            )
            if brief_m:
                prev_brief = brief_m.group(1).strip()[:1200]
            kf_m = re.search(r"##\s*Key Findings\s*\n(.*?)(?:\n##|\Z)", txt, re.DOTALL)
            if kf_m:
                prev_kf = kf_m.group(1).strip()[:400]

    # --- 01_literature.md stub ---
    lit_path = round_dir / OUTPUT_FILES["researcher"]
    if not lit_path.exists():
        stub = (
            "## SOTA Summary\n"
            "- Stub generated by pipeline (Researcher agent failed to write output)\n"
            f"- Prior round key findings: {prev_kf or 'see findings.md'}\n"
            "- Basic statistical analysis (numpy/scipy/pandas) confirmed working\n\n"
            "## Available Datasets\n"
            f"{local_block}\n\n"
            "## Baselines\n"
            "- (No baselines extracted — Researcher agent failed. Use prior round findings.)\n\n"
            "## FOR THE EXPERIMENT DESIGNER\n"
            f"Fallback stub — proceed using the NEXT_ROUND_BRIEF from the previous synthesis:\n{prev_brief or 'Use computational methods on local data files only.'}\n"
        )
        lit_path.write_text(stub, encoding="utf-8")
        display.print_info(
            "  [yellow]⚠ Created fallback 01_literature.md stub "
            "(Researcher agent produced no output)[/yellow]"
        )

    # --- 02_experiment.md stub ---
    exp_path = round_dir / OUTPUT_FILES["hypothesis"]
    if not exp_path.exists():
        brief_block = prev_brief or (
            "1. Use numpy/scipy/pandas to compute basic descriptors or statistics "
            "from the local data files.\n"
            "2. Rank candidates using multi-metric evaluation (Pareto or Z-score).\n"
            "3. Save results to key_results.json."
        )
        stub = (
            f"## Experiment: Fallback Continuation Round {round_num}\n"
            "**Hypothesis**: Candidates can be ranked using descriptors computed "
            "from the primary data files.\n"
            "**Success metric**: At least one numerical result in key_results.json\n"
            "**Failure threshold**: key_results.json empty or all errors\n\n"
            "## Algorithm / Approach\n"
            f"{brief_block}\n\n"
            "## Data Plan\n"
            f"**Primary**: Local files in {working_dir} · ACCESSIBLE\n"
            "**Fallback**: Use any accessible dataset mentioned in prior round findings\n\n"
            "## Expected Output Files\n"
            '- results/key_results.json → {"metric": "primary_metric", "value": <float>, "baseline": <float>}\n'
            "- results/main_plot.png\n"
            "- results/summary_figure.png\n\n"
            "## FOR THE CODER\n"
            "Fallback spec — use the algorithm above. Minimum viable output: "
            "key_results.json with at least one numerical value.\n"
        )
        exp_path.write_text(stub, encoding="utf-8")
        display.print_info(
            "  [yellow]⚠ Created fallback 02_experiment.md stub "
            "(Hypothesis agent produced no output)[/yellow]"
        )


def _ensure_debug_stub(round_dir: Path) -> None:
    """Write a minimal 04_debug_report.md if the debugger failed to produce one."""
    debug_path = round_dir / OUTPUT_FILES["debugger"]
    if debug_path.exists():
        return
    kr_path = round_dir / OUTPUT_FILES["coder"] / "results" / "key_results.json"
    has_results = kr_path.exists() and kr_path.stat().st_size > 10
    stub = (
        "## Bugs Found and Fixed\n"
        "Debugger agent ran but did not produce this report (budget exhausted or error).\n"
        "Manual review recommended.\n\n"
        "## Tests Run\n"
        "(not run — debugger did not complete)\n\n"
        "## Verified Results\n"
        f"{'key_results.json exists — see results/ directory.' if has_results else 'key_results.json missing or empty.'}\n\n"
        "## Outstanding Issues\n"
        "Debugger failed to write report — full audit skipped this round.\n\n"
        "## Confidence Score\n"
        "0/10\n"
    )
    debug_path.write_text(stub, encoding="utf-8")
    display.print_info(
        "  [yellow]⚠ Created fallback 04_debug_report.md stub "
        "(Debugger agent produced no output)[/yellow]"
    )


# ---------------------------------------------------------------------------
# Overseer: parse synthesis for next brief and completion signal
# ---------------------------------------------------------------------------

def _parse_synthesis(synthesis_path: str) -> tuple[str, bool]:
    """
    Read the orchestrator's synthesis file.
    Returns (next_brief: str, is_complete: bool).
    """
    path = Path(synthesis_path)
    if not path.exists():
        return "Continue the research with improvements based on previous round.", False

    text = path.read_text(errors="replace")

    if COMPLETE_MARKER in text:
        return "", True

    match = re.search(
        rf"{re.escape(NEXT_BRIEF_MARKER)}\s*(.*?)(?:\n## |\Z)",
        text,
        re.DOTALL,
    )
    if match:
        brief = match.group(1).strip()
        if brief:
            return brief, False

    # Fallback: use last 1500 chars of synthesis as implicit brief
    return text[-1500:].strip(), False


# ---------------------------------------------------------------------------
# Context trimmer (last-resort when context window fills up)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Master HTML report (runs once after all rounds complete)
# ---------------------------------------------------------------------------

_MASTER_REPORTER_PROMPT = """\
You are the Master Reporter for an autonomous multi-agent research run.

TOPIC     : {topic}
ROUNDS    : {rounds_done}
RESEARCH  : {research_dir}

YOUR MISSION
One comprehensive, self-contained HTML report covering the full research run.
This is the definitive deliverable — spend your tokens here, not on intermediary prose.

STEPS
1. List round directories under {research_dir}/.
2. For each round read ONLY: round_NNN/05_evaluation.md, round_NNN/06_synthesis.md,
   round_NNN/03_code/IMPLEMENTATION.md (if exists).
   Read round_NNN/02_experiment.md only for the hypothesis name and success metric.
3. Read {research_dir}/findings.md.
4. Collect all summary_figure.png and 05_scores_chart.png from each round.
   Also list any other PNGs in round_NNN/03_code/results/.
5. Write {research_dir}/build_master_report.py. Run it.
   Must produce {research_dir}/final_report.html.

HTML SECTIONS:
  1. Sticky nav
  2. Title block (topic, date, rounds, quality badge)
  3. Abstract (1 paragraph — entire research arc)
  4. Research Interpretation (1 paragraph — what did the pipeline collectively learn?
     Connect accumulated findings to the original research question. What do the
     numbers, methods, and failures mean together? This is the "so what" section.)
  5. Research Timeline table: Round | Hypothesis | Score | Status
  6. Cumulative Findings (from findings.md, as cards)
  7. Accumulated Knowledge (from case_memory.md — transferable lessons across rounds,
     presented as a "what we learned" card deck; skip if file absent)
  8. Round Deep Dives — one <details> per round (NOT <div class="details">):
       hypothesis · implementation summary · ALL result plots
       · scores chart · what worked / failed · transferable lessons
  9. Score Progression chart (generate with matplotlib: round on x, score on y).
     Save as score_progression.png in {research_dir} and embed it.
  10. Key Visualisations Gallery (summary_figure.png from each round, full-width)
  11. Conclusions & Next Steps (from final synthesis)
  Footer: topic · timestamp · "Generated by OctoSlave"

IMAGES — STRICT RULES:
- Embed images via RELATIVE PATHS, e.g. <img src="round_001/03_code/results/main_plot.png" alt="Round 1 main plot">.
  final_report.html lives at {research_dir}/final_report.html, so paths are
  relative to that file. Use the actual files you found in step 4.
- Do NOT base64-inline images. Do NOT write src="data:image/png;base64,..." with ellipsis.
  Truncating base64 with ... is a hard failure — every image will be broken.
- Every <img> MUST have alt text describing the figure.

HTML QUALITY — STRICT RULES:
- Use <details><summary>...</summary>...content...</details> for collapsible round
  sections. NEVER write <div class="details"><summary>...</summary> — <summary> is
  invalid outside a <details> element.
- All <img> tags self-close with no stray punctuation. Example: <img src="x.png" alt="y">
  NOT <img src="x.png" alt="y")>.
- Cards on white background MUST have explicit dark text colour (e.g. .card {{ color: #0d1117 }}).
  Body text on dark background can stay white. Verify contrast for every block.

DESIGN: dark page background (#0d1117), white cards (#ffffff with color: #0d1117),
Inter font (CDN OK), max-width 1200px, collapsible rounds via <details>/<summary>.
Script: stdlib + matplotlib only. Print output path on success.

A Python-side post-processor will repair common failures (truncated base64,
malformed <details>, missing contrast). Aim for clean output anyway — the
post-processor is a safety net, not a substitute for valid HTML.
"""

_MASTER_REPORTER_SYSTEM = """\
You are an expert scientific report writer. You produce polished, self-contained
HTML research reports. You write clean Python scripts that generate these reports.
Working directory: {working_dir}
"""


def _run_master_reporter(
    topic: str,
    research_dir: str,
    rounds_done: int,
    working_dir: str,
    client: OpenAI,
    model: str,
) -> None:
    """Generate the final master HTML report covering all rounds."""
    import pathlib as _pl
    cfg = ROLES["reporter"]
    tools = _tools_for_role("reporter")

    # Remove stale report so the reporter always regenerates fresh
    stale = _pl.Path(research_dir) / "final_report.html"
    if stale.exists():
        stale.unlink()

    display.print_agent_banner("reporter", model, rounds_done, rounds_done)
    display.print_info("  Generating master report…")

    system = _MASTER_REPORTER_SYSTEM.format(working_dir=working_dir)
    user_task = _MASTER_REPORTER_PROMPT.format(
        topic=topic,
        rounds_done=rounds_done,
        research_dir=research_dir,
    )

    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_task},
    ]

    t0 = time.time()
    iteration = 0
    _rate_limit_retries = 0
    _timeout_retries = 0

    while iteration < cfg["max_iter"]:
        iteration += 1
        try:
            response = _stream_completion_with_tools(client, model, messages, tools)
            _rate_limit_retries = 0
            _timeout_retries = 0
        except BadRequestError as e:
            err = str(e)
            if "ContextWindow" in err or "context" in err.lower():
                trimmed = _trim_messages(messages)
                if len(trimmed) < len(messages):
                    messages = trimmed
                    iteration -= 1
                    continue
                display.print_error("[Master Reporter] Context window exceeded and cannot be trimmed further. Aborting.")
                return
            if "Unterminated string" in err or "Extra data" in err:
                _popped = 0
                while messages and messages[-1].get("role") in ("tool", "assistant"):
                    messages.pop()
                    _popped += 1
                if _popped == 0:
                    display.print_error("[Master Reporter] Truncated args error but no rollback target. Aborting.")
                    return
                display.print_info("[Master Reporter] Tool call arguments truncated — rolling back and retrying.")
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous response was cut off before the tool arguments "
                        "were complete. Please redo the last action with complete, valid arguments."
                    ),
                })
                iteration -= 1
                continue
            display.print_error(f"[Master Reporter] API error: {e}")
            return
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate" in err_str.lower() or "RateLimit" in type(e).__name__:
                _rate_limit_retries += 1
                wait = min(60, 5 * (2 ** (_rate_limit_retries - 1)))
                display.print_info(f"[Master Reporter] Rate limit — waiting {wait}s ({_rate_limit_retries}/5).")
                if _rate_limit_retries > 5:
                    display.print_error("[Master Reporter] Rate limit persists. Aborting.")
                    return
                time.sleep(wait)
                iteration -= 1
                continue
            elif "timeout" in err_str.lower() or "timed out" in err_str.lower() or "Timeout" in type(e).__name__:
                _timeout_retries += 1
                wait = min(30, 5 * (2 ** (_timeout_retries - 1)))
                display.print_info(f"[Master Reporter] Request timeout — retrying in {wait}s ({_timeout_retries}/3).")
                if _timeout_retries > 3:
                    display.print_error("[Master Reporter] Request keeps timing out. Aborting.")
                    return
                time.sleep(wait)
                iteration -= 1
                continue
            display.print_error(f"[Master Reporter] Unexpected error: {e}")
            return
        except KeyboardInterrupt:
            display.stream_end(False)
            display.console.print("\n[dim]Master report interrupted.[/dim]")
            return

        content = response["content"]
        tool_calls = response["tool_calls"]
        finish_reason = response["finish_reason"]

        assistant_msg: dict = {"role": "assistant", "content": content if content else ""}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if not tool_calls:
            # Only exit early if final_report.html has been produced.
            import pathlib as _pl2
            if (_pl2.Path(research_dir) / "final_report.html").exists():
                break
            # Not written yet — nudge the model (decrement so the nudge counts as a free slot)
            nudge = (
                f"You have not yet written {research_dir}/final_report.html. "
                "Write build_master_report.py and run it, OR write final_report.html directly. "
                "Call write_file or bash now."
            )
            messages.append({"role": "user", "content": nudge})
            iteration -= 1
            continue

        if finish_reason == "stop":
            break

        display.print_separator()
        for tc in tool_calls:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                tc["function"]["arguments"] = "{}"
                args = {}
                err_msg = (
                    f"Tool call '{name}' had malformed JSON arguments "
                    "(the model's response was truncated). Please retry with complete arguments."
                )
                display.print_tool_result(name, err_msg, False)
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": err_msg})
                continue
            display.print_tool_call(name, args)
            result, success = execute_tool(name, args, working_dir)
            result = _cap_result(result, name)
            display.print_tool_result(name, result, success)
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
        display.print_separator()

    elapsed = time.time() - t0
    display.print_agent_done("reporter", elapsed, iteration)

    final_report = Path(research_dir) / "final_report.html"
    if final_report.exists():
        try:
            n_fixes = _postprocess_report_html(final_report)
            if n_fixes:
                display.print_info(
                    f"  [yellow]Report post-processor[/yellow] applied {n_fixes} fix(es) "
                    "(placeholder images / malformed HTML / contrast)."
                )
        except Exception as _pp_err:
            display.print_info(f"  [yellow]Report post-processor failed:[/yellow] {_pp_err}")

        display.print_info(
            f"  [bold bright_cyan]Master report → {final_report}[/bold bright_cyan]"
        )


# ---------------------------------------------------------------------------
# Master report post-processor
# ---------------------------------------------------------------------------

# Tag inserted before </head> so the post-processor's CSS overrides take precedence.
_POSTPROCESSOR_CSS = """
<style id="ots-postprocessor-overrides">
  /* Fix white-on-white text in cards/abstracts (common 49B mistake). */
  .card, .abstract, .timeline-table, .visuals-gallery,
  .research-interpretation, .details {
      color: #0d1117;
  }
  .card *, .abstract *, .timeline-table *, .visuals-gallery *,
  .research-interpretation * { color: inherit; }
  .timeline-table table { width: 100%; border-collapse: collapse; background: #fff;
      border-radius: 8px; overflow: hidden; color: #0d1117; }
  .timeline-table th { background: #161b22; color: #fff; padding: 0.75rem 1rem;
      text-align: left; }
  .timeline-table td { padding: 0.75rem 1rem; border-top: 1px solid #e1e4e8; }
  .details { background: #fff !important; color: #0d1117; }
  .details summary { cursor: pointer; font-weight: 600; padding: 0.5rem 0; }
  .visuals-gallery img, .chart img { max-width: 100%; height: auto;
      border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }
  body { line-height: 1.6; }
  h1, h2, h3 { letter-spacing: -0.01em; }
  /* Fallback box for images that failed to load (caught by onerror below) */
  .ots-img-missing { display: inline-block; padding: 1.5rem 2rem;
      background: #fff7d6; color: #5a4500; border: 1px dashed #c4a000;
      border-radius: 8px; font-style: italic; }
</style>
"""


def _wrap_details_divs(html: str) -> tuple[str, int]:
    """
    Convert <div class="details"><summary>...</summary>...</div> into
    <details class="details" open><summary>...</summary>...</details>,
    matching the OUTER </div> using a div-depth counter so nested <div>s
    (e.g. <div class="chart">) don't get mistaken for the close tag.
    Returns (new_html, n_fixes_applied). Idempotent — already-correct HTML
    passes through unchanged.
    """
    import re

    open_re = re.compile(
        r'<div\s+class=(["\'])details\1\s*>(\s*)<summary>([^<]+)</summary>',
        re.IGNORECASE,
    )
    out_parts: list[str] = []
    cursor = 0
    n_fixes = 0
    tag_re = re.compile(r'<\s*(/?)\s*(div|details)\b[^>]*>', re.IGNORECASE)

    for m in open_re.finditer(html):
        # Emit text up to the match unchanged
        out_parts.append(html[cursor:m.start()])
        # Walk forward from m.end() balancing div depth (we entered at depth 1)
        depth = 1
        i = m.end()
        close_match = None
        for sub in tag_re.finditer(html, i):
            slash, name = sub.group(1), sub.group(2).lower()
            if name == "div":
                if not slash:
                    depth += 1
                else:
                    depth -= 1
                    if depth == 0:
                        close_match = sub
                        break
            # Ignore nested details; if model already used <details>, we treat
            # them as opaque — we still only count <div> depth.
        if close_match is None:
            # Unbalanced — leave this block alone, copy original text and move on.
            out_parts.append(html[m.start():m.end()])
            cursor = m.end()
            continue

        # Replace open tag with <details ...> and the matched </div> with </details>
        out_parts.append(
            f'<details class="details" open>{m.group(2)}<summary>{m.group(3)}</summary>'
        )
        out_parts.append(html[m.end():close_match.start()])
        out_parts.append("</details>")
        cursor = close_match.end()
        n_fixes += 1

    out_parts.append(html[cursor:])
    return "".join(out_parts), n_fixes


def _postprocess_report_html(report_path: Path) -> int:
    """Repair common Reporter failures on the written HTML. Returns n fixes applied."""
    import re

    research_dir = report_path.parent
    html = report_path.read_text(encoding="utf-8", errors="replace")
    fixes = 0

    # Collect all real PNGs available on disk (relative to research_dir, sorted by round).
    available_pngs: list[Path] = sorted(research_dir.glob("round_*/03_code/results/*.png"))
    available_pngs += sorted(research_dir.glob("round_*/05_scores_chart.png"))
    available_pngs += sorted(research_dir.glob("*.png"))
    rel_pngs: list[str] = [str(p.relative_to(research_dir)) for p in available_pngs]

    # 1. Replace truncated base64 placeholders with relative paths to real PNGs.
    #    A real base64 PNG src is thousands of chars long; if we see one ending in
    #    `...` or `…` or shorter than 500 chars, it's a placeholder.
    placeholder_re = re.compile(
        r'src=(?P<q>["\'])data:image/[^;]+;base64,(?P<data>[^"\']*?)(?P=q)',
        re.IGNORECASE,
    )

    pool = list(rel_pngs)  # consume in order
    used: set[str] = set()

    def _replace_placeholder(m: re.Match) -> str:
        nonlocal fixes
        data = m.group("data")
        # Heuristic: real base64 of a chart PNG is usually > 5000 chars; ellipsis/truncation
        # markers are dead giveaways.
        if (len(data) >= 500 and not data.endswith("...") and not data.endswith("…")
                and "..." not in data[-10:]):
            return m.group(0)  # looks legit, keep it
        if not pool:
            # No real PNG to substitute — make the failure visible as a labelled box.
            fixes += 1
            return ('src="data:image/svg+xml;utf8,'
                    '<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22600%22 height=%22120%22>'
                    '<rect width=%22100%25%22 height=%22100%25%22 fill=%22%23fff7d6%22/>'
                    '<text x=%2250%25%22 y=%2255%25%22 font-family=%22sans-serif%22 '
                    'font-size=%2218%22 text-anchor=%22middle%22 fill=%22%235a4500%22>'
                    'Image missing — Reporter wrote a placeholder; no PNG available'
                    '</text></svg>"')
        path = pool.pop(0)
        used.add(path)
        fixes += 1
        return f'src="{path}"'

    new_html = placeholder_re.sub(_replace_placeholder, html)

    # 2. Fix the stray `)>` HTML attribute syntax error this Reporter is fond of.
    new_html, n_paren = re.subn(r'(["\'])\)>', r'\1>', new_html)
    fixes += n_paren

    # 3. Wrap loose <summary> ... </summary> inside <div class="details"> ... </div>
    #    in proper <details> ... </details> elements. Depth-correct: closes on
    #    the OUTER </div>, not on a nested one (e.g. <div class="chart">...</div>).
    new_html, n_details = _wrap_details_divs(new_html)
    fixes += n_details

    # 4. Inject our overrides CSS so contrast / table / image styles are sane,
    #    regardless of what the model wrote.
    if "ots-postprocessor-overrides" not in new_html and "</head>" in new_html:
        new_html = new_html.replace("</head>", _POSTPROCESSOR_CSS + "</head>", 1)
        fixes += 1

    # 5. Add img onerror that converts a failed image to a labelled missing-box,
    #    so future runs degrade visibly instead of showing a broken-image icon.
    new_html, n_img = re.subn(
        r'<img\b(?![^>]*onerror=)([^>]*)>',
        r'<img\1 onerror="this.outerHTML=\'<div class=&quot;ots-img-missing&quot;>'
        r'image not found: \'+(this.alt||this.src)+\'</div>\'">',
        new_html,
    )
    if n_img:
        fixes += 1  # count as one logical fix

    if new_html != html:
        report_path.write_text(new_html, encoding="utf-8")
    return fixes


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _probe_hardware(research_dir: str) -> dict:
    """
    Run a hardware probe and write hw_profile.json to research_dir.
    Returns the profile dict. Safe to call even if torch/psutil are absent.
    """
    import subprocess as _sp
    hw_path = Path(research_dir) / "hw_profile.json"

    script = (
        "import json, platform, os, sys, shutil, subprocess as _sp\n"
        "info = {'python': sys.version.split()[0], 'platform': platform.platform(), "
        "'cpu_count': os.cpu_count()}\n"
        # UV availability
        "uv_path = shutil.which('uv')\n"
        "if uv_path:\n"
        "    try:\n"
        "        v = _sp.run(['uv', '--version'], capture_output=True, text=True, timeout=5)\n"
        "        info['uv_available'] = True\n"
        "        info['uv_version'] = v.stdout.strip()\n"
        "    except Exception:\n"
        "        info['uv_available'] = True\n"
        "        info['uv_version'] = 'unknown'\n"
        "else:\n"
        "    info['uv_available'] = False\n"
        "    info['uv_version'] = None\n"
        "try:\n"
        "    import psutil; m = psutil.virtual_memory()\n"
        "    info['ram_total_gb'] = round(m.total/1e9,1)\n"
        "    info['ram_available_gb'] = round(m.available/1e9,1)\n"
        "except ImportError: pass\n"
        "try:\n"
        "    import torch\n"
        "    info['torch_version'] = torch.__version__\n"
        "    info['cuda_available'] = torch.cuda.is_available()\n"
        "    if torch.cuda.is_available():\n"
        "        info['cuda_device_count'] = torch.cuda.device_count()\n"
        "        info['cuda_devices'] = [{'name': torch.cuda.get_device_name(i), "
        "'vram_gb': round(torch.cuda.get_device_properties(i).total_memory/1e9,1)} "
        "for i in range(torch.cuda.device_count())]\n"
        "        info['cuda_version'] = torch.version.cuda\n"
        "except ImportError:\n"
        "    info['torch_available'] = False\n"
        "try:\n"
        "    r = _sp.run(['nvidia-smi','--query-gpu=name,memory.total,memory.free',"
        "'--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)\n"
        "    if r.returncode==0: info['nvidia_smi'] = r.stdout.strip()\n"
        "except Exception: pass\n"
        "# Package availability probe — find_spec only (no actual import) so we\n"
        "# do NOT load slow giants like torch / tensorflow / transformers and\n"
        "# blow past the subprocess timeout.\n"
        "import importlib.util as _ilu\n"
        "try:\n"
        "    from importlib.metadata import version as _ver, PackageNotFoundError as _PNF\n"
        "except Exception:\n"
        "    _ver = None; _PNF = Exception  # type: ignore\n"
        # (import_name, distribution_name_for_version_lookup)
        "_sci_pkgs = [\n"
        "    ('numpy','numpy'),('scipy','scipy'),('matplotlib','matplotlib'),\n"
        "    ('pandas','pandas'),('sklearn','scikit-learn'),\n"
        "    ('seaborn','seaborn'),('plotly','plotly'),('statsmodels','statsmodels'),\n"
        "    ('networkx','networkx'),('umap','umap-learn'),('hdbscan','hdbscan'),\n"
        "    ('Bio','biopython'),('biotite','biotite'),('prody','ProDy'),\n"
        "    ('pdbfixer','pdbfixer'),('nglview','nglview'),('pymol','pymol'),\n"
        "    ('rdkit','rdkit'),('openmm','openmm'),('mdtraj','mdtraj'),\n"
        "    ('MDAnalysis','MDAnalysis'),\n"
        "    ('torch','torch'),('tensorflow','tensorflow'),('keras','keras'),\n"
        "    ('transformers','transformers'),\n"
        "    ('sentence_transformers','sentence-transformers'),\n"
        "    ('xgboost','xgboost'),('lightgbm','lightgbm'),('catboost','catboost'),\n"
        "    ('duckdb','duckdb'),('pyarrow','pyarrow'),('polars','polars'),\n"
        "    ('sqlalchemy','SQLAlchemy'),\n"
        "    ('requests','requests'),('httpx','httpx'),\n"
        "    ('bs4','beautifulsoup4'),('lxml','lxml'),\n"
        "    ('anndata','anndata'),('scanpy','scanpy'),\n"
        "]\n"
        "_avail = []\n"
        "for _imp, _dist in _sci_pkgs:\n"
        "    try:\n"
        "        if _ilu.find_spec(_imp) is None:\n"
        "            continue\n"
        "    except Exception:\n"
        "        continue\n"
        "    _v = None\n"
        "    if _ver is not None:\n"
        "        try:\n"
        "            _v = _ver(_dist)\n"
        "        except _PNF:\n"
        "            _v = None\n"
        "        except Exception:\n"
        "            _v = None\n"
        "    _avail.append(f\"{_imp}=={_v}\" if _v else _imp)\n"
        "info['available_packages'] = _avail\n"
        "print(json.dumps(info))\n"
    )

    profile: dict = {}
    try:
        import shutil as _shutil, sys as _sys, os as _os
        # Build candidate list: prefer pythons that have the most science packages.
        # The OctoSlave venv python often lacks numpy/sklearn/etc., so we search
        # common conda/system locations and pick the richest environment.
        _home = _os.path.expanduser("~")
        _candidates = []
        for _p in [
            _sys.executable,
            _shutil.which("python3"),
            _shutil.which("python"),
            f"{_home}/miniconda3/bin/python",
            f"{_home}/miniconda3/bin/python3",
            f"{_home}/anaconda3/bin/python",
            f"{_home}/anaconda3/bin/python3",
            f"{_home}/miniforge3/bin/python",
            f"{_home}/miniforge3/bin/python3",
            "/usr/local/bin/python3",
            "/usr/bin/python3",
        ]:
            if _p and _os.path.isfile(_p) and _p not in _candidates:
                _candidates.append(_p)

        _best_py = _candidates[0] if _candidates else "python3"
        _best_count = -1
        for _py in _candidates:
            try:
                _r = _sp.run(
                    [_py, "-c",
                     "import importlib, json; "
                     "pkgs=['numpy','scipy','matplotlib','pandas','sklearn','biopython','torch']; "
                     "found=[p for p in pkgs if importlib.util.find_spec(p) is not None]; "
                     "print(len(found))"],
                    capture_output=True, text=True, timeout=5,
                )
                _count = int(_r.stdout.strip()) if _r.returncode == 0 and _r.stdout.strip().isdigit() else 0
                if _count > _best_count:
                    _best_count = _count
                    _best_py = _py
            except Exception:
                continue

        probe_err: str | None = None
        try:
            result = _sp.run(
                [_best_py, "-c", script],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode == 0 and result.stdout.strip():
                profile = json.loads(result.stdout.strip())
                profile["python_executable"] = _best_py
            else:
                probe_err = (
                    f"hw probe rc={result.returncode}; "
                    f"stderr_tail={(result.stderr or '').strip()[-300:]!r}"
                )
        except _sp.TimeoutExpired:
            probe_err = f"hw probe timed out after 60s on {_best_py}"
        except json.JSONDecodeError as _je:
            probe_err = f"hw probe stdout not JSON: {_je} | head={result.stdout[:200]!r}"
        if probe_err:
            try:
                display.print_info(f"[yellow]hw_profile fallback:[/yellow] {probe_err}")
            except Exception:
                pass
            # Always include at least basic info from the parent process so
            # downstream agents have *something* to read instead of `{}`.
            import platform as _plat
            profile.setdefault("python", _sys.version.split()[0])
            profile.setdefault("platform", _plat.platform())
            profile.setdefault("cpu_count", _os.cpu_count())
            profile.setdefault("python_executable", _best_py)
            profile.setdefault("available_packages", [])
            profile.setdefault("probe_error", probe_err)
    except Exception as _e:
        try:
            display.print_info(f"[yellow]hw_profile fatal:[/yellow] {_e}")
        except Exception:
            pass

    hw_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    # Pretty-print hardware summary
    cuda = profile.get("cuda_available", False)
    devices = profile.get("cuda_devices", [])
    ram = profile.get("ram_total_gb", "?")
    cpus = profile.get("cpu_count", "?")
    uv_ok = profile.get("uv_available", False)
    uv_ver = profile.get("uv_version") or ""

    uv_tag = (
        f"[bold bright_green]uv ✓[/bold bright_green] ({uv_ver})"
        if uv_ok
        else "[bold red]uv ✗ — agents will fall back to pip[/bold red]"
    )

    if cuda and devices:
        gpu_str = ", ".join(f"{d['name']} ({d['vram_gb']} GB)" for d in devices)
        display.print_info(f"  Hardware: {cpus} CPU cores, {ram} GB RAM, "
                           f"[bold bright_green]CUDA ✓[/bold bright_green] {gpu_str}  |  {uv_tag}")
    else:
        display.print_info(f"  Hardware: {cpus} CPU cores, {ram} GB RAM, "
                           f"[dim]no CUDA GPU detected[/dim]  |  {uv_tag}")

    pkgs = profile.get("available_packages", [])
    if pkgs:
        display.print_info(f"  Available packages: {', '.join(pkgs[:8])}{'…' if len(pkgs) > 8 else ''}")

    return profile


_SCRAPE_KEYWORDS = frozenset({
    "scrape", "scraping", "crawl", "crawling", "spider", "spidering",
    "harvest", "harvesting", "extract from website", "extract from site",
    "extract from url", "web extraction", "data extraction from",
})


def _sanitize_key_results(round_dir: Path) -> None:
    """Fix non-finite floats in key_results.json (Python writes Infinity/NaN; standard JSON forbids them)."""
    import math, re as _re
    kr_path = round_dir / OUTPUT_FILES["coder"] / "results" / "key_results.json"
    if not kr_path.exists():
        return
    raw = kr_path.read_text(errors="replace")
    if not any(tok in raw for tok in ("Infinity", "NaN", "-Infinity")):
        return  # already valid
    cleaned = _re.sub(r'\b(-?Infinity|NaN)\b', 'null', raw)
    try:
        json.loads(cleaned)  # validate
        kr_path.write_text(cleaned, encoding="utf-8")
    except Exception:
        pass  # leave as-is if still broken; debugger will catch it


def _has_numerical_results(round_dir: Path) -> bool:
    """Return True if the round's key_results.json contains at least one real number."""
    kr_path = round_dir / OUTPUT_FILES["coder"] / "results" / "key_results.json"
    if not kr_path.exists():
        return False
    try:
        kr = json.loads(kr_path.read_text(errors="replace"))
        def _has_num(obj) -> bool:
            if isinstance(obj, (int, float)) and not isinstance(obj, bool):
                return True
            if isinstance(obj, dict):
                return any(_has_num(v) for v in obj.values())
            if isinstance(obj, list):
                return any(_has_num(v) for v in obj)
            return False
        return _has_num(kr)
    except Exception:
        return False


def run_long_research(
    topic: str,
    working_dir: str,
    client: OpenAI,
    max_rounds: int = 5,
    model_overrides: dict[str, str] | None = None,
    resume: bool = False,
    num_parallel: int = 1,
    scrape_mode: bool = False,
    prompt_profile: str = "base",
) -> None:
    """
    Run the full autonomous multi-agent research pipeline.

    Args:
        topic:           The research topic / goal.
        working_dir:     The project working directory.
        client:          Authenticated OpenAI client.
        max_rounds:      Maximum number of research rounds.
        model_overrides: Per-role model overrides, e.g. {"coder": "qwen3-coder-30b"}.
        resume:          If True, skip rounds whose output files already exist.
        num_parallel:    Number of independent agent copies to run for parallelisable
                         roles (researcher, hypothesis, evaluator). Default 1 = sequential.
        scrape_mode:     If True (or auto-detected from topic keywords), gives the
                         researcher the crawl_tree tool and a scraping-focused prompt.
    """
    # Auto-detect scrape intent from topic
    topic_lower = topic.lower()
    if not scrape_mode:
        scrape_mode = any(kw in topic_lower for kw in _SCRAPE_KEYWORDS)
    if scrape_mode:
        display.print_info("  [bold yellow]🕷  Scrape mode active — researcher has crawl_tree tool.[/bold yellow]")
    overrides = model_overrides or {}
    research_dir = Path(working_dir) / "research"
    research_dir.mkdir(parents=True, exist_ok=True)

    display.print_research_start(topic, max_rounds, ROLES, overrides)

    # Probe hardware once; result is written to research_dir/hw_profile.json
    # and read by the coder/debugger agents in every subsequent round.
    _probe_hardware(str(research_dir))

    # Scan working directory for user-supplied local files (PDFs, CSVs, data, etc.)
    # Include them in the brief so every agent knows they exist from round 1.
    local_files = [
        p for p in Path(working_dir).iterdir()
        if p.is_file() and p.suffix.lower() in LOCAL_DATA_EXTENSIONS
    ]
    local_files_block = ""
    if local_files:
        file_list = "\n".join(f"  - {p.name}  ({p.stat().st_size // 1024 or 1} KB)" for p in local_files)
        local_files_block = (
            f"\n\nLOCAL FILES IN WORKING DIR — provided by the user as primary input:\n"
            f"{file_list}\n"
            "Agents MUST read these files before doing any web searches. "
            "They take precedence over anything found online."
        )
        display.print_info(
            f"  Local data files detected: {', '.join(p.name for p in local_files)}"
        )

    # Initial brief
    if scrape_mode:
        brief = (
            f"Scraping task: {topic}\n"
            "Use crawl_tree to map the full site structure, then web_fetch leaf pages "
            "to understand data fields. Save the tree JSON to the round dir. "
            "Subsequent rounds will implement and run the full extractor."
            f"{local_files_block}"
        )
    else:
        brief = (
            f"Initial research round. Conduct a broad literature survey on: {topic}\n"
            "Identify key papers, available datasets, existing methods, and open problems.\n"
            "Generate first hypotheses and implement the most promising experiment."
            f"{local_files_block}"
        )

    completed_early = False

    for round_num in range(1, max_rounds + 1):
        round_dir = research_dir / f"round_{round_num:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        display.print_round_header(round_num, max_rounds, str(round_dir))

        for role in PIPELINE:
            model = overrides.get(role) or ROLES[role]["default_model"]

            # Resumability: skip if canonical output already exists
            expected_path = round_dir / OUTPUT_FILES[role]
            output_complete = expected_path.exists()
            if output_complete and role == "coder":
                # For the coder, the directory must contain IMPLEMENTATION.md to be valid
                output_complete = (expected_path / "IMPLEMENTATION.md").exists()
            if resume and output_complete:
                display.print_info(
                    f"  ↩  {ROLES[role]['label']} output found — skipping."
                )
                continue

            ok = True
            try:
                if num_parallel > 1 and role in PARALLEL_ROLES:
                    display.print_info(
                        f"  ⚡ Spawning {num_parallel} parallel "
                        f"{ROLES[role]['label']} agents…"
                    )
                    parallel_outputs = _run_parallel_specialists(
                        role=role,
                        n=num_parallel,
                        models=[model] * num_parallel,
                        topic=topic,
                        round_num=round_num,
                        max_rounds=max_rounds,
                        round_dir=round_dir,
                        research_dir=str(research_dir),
                        working_dir=working_dir,
                        brief=brief,
                        client=client,
                        scrape_mode=scrape_mode,
                        prompt_profile=prompt_profile,
                    )
                    if parallel_outputs:
                        _run_merger(
                            role=role,
                            parallel_outputs=parallel_outputs,
                            canonical_path=expected_path,
                            round_num=round_num,
                            max_rounds=max_rounds,
                            round_dir=str(round_dir),
                            research_dir=str(research_dir),
                            working_dir=working_dir,
                            client=client,
                            model=overrides.get("merger") or ROLES["merger"]["default_model"],
                        )
                    else:
                        ok = False
                        display.print_error(
                            f"All parallel {ROLES[role]['label']} agents failed "
                            f"in round {round_num}. Continuing."
                        )
                else:
                    ok = _run_specialist(
                        role=role,
                        model=model,
                        topic=topic,
                        round_num=round_num,
                        max_rounds=max_rounds,
                        round_dir=str(round_dir),
                        research_dir=str(research_dir),
                        working_dir=working_dir,
                        brief=brief,
                        client=client,
                        scrape_mode=scrape_mode,
                        prompt_profile=prompt_profile,
                    )
            except KeyboardInterrupt:
                display.console.print(
                    "\n[bold yellow]Research paused.[/bold yellow] "
                    f"Progress saved to [dim]{research_dir}[/dim]\n"
                    "Re-run with [cyan]/long-research ... --resume[/cyan] to continue."
                )
                return

            if not ok:
                display.print_error(
                    f"{ROLES[role]['label']} failed in round {round_num}. "
                    "Continuing with next agent."
                )

            # Structural integrity check: warn if expected output file is missing
            expected_out = round_dir / OUTPUT_FILES[role]
            if role != "coder":  # coder output is a directory, not a single file
                if not expected_out.exists():
                    display.print_info(
                        f"  [yellow]⚠ {ROLES[role]['label']}: expected output "
                        f"{OUTPUT_FILES[role]} not found after {ROLES[role]['max_iter']} "
                        f"iterations. Next role will proceed without it.[/yellow]"
                    )

            # After coder: sanitize key_results.json (Infinity/NaN → null)
            if role == "coder":
                _sanitize_key_results(round_dir)

            # After debugger: ensure 04_debug_report.md exists (pipeline stub if agent failed)
            if role == "debugger":
                _ensure_debug_stub(round_dir)

            # After hypothesis: ensure 01_literature.md and 02_experiment.md exist
            # before the coder runs. Creates minimal stubs if agents failed.
            if role == "hypothesis":
                _ensure_handoff_stubs(round_dir, research_dir, working_dir, round_num)

        # Update findings.md and case_memory.md from round outputs — pipeline-owned, not LLM-owned
        _update_findings(
            research_dir=str(research_dir),
            round_num=round_num,
            round_dir=str(round_dir),
            topic=topic,
        )
        _update_case_memory(
            research_dir=str(research_dir),
            round_num=round_num,
            round_dir=str(round_dir),
            topic=topic,
        )

        # Parse orchestrator synthesis → next brief
        synthesis_path = round_dir / OUTPUT_FILES["orchestrator"]
        brief, is_complete = _parse_synthesis(str(synthesis_path))

        # Stagnation detection: two consecutive rounds with zero numerical results
        if round_num >= 2:
            prev_dir = research_dir / f"round_{round_num:03d}"
            prev2_dir = research_dir / f"round_{max(1, round_num - 1):03d}"
            if not _has_numerical_results(prev_dir) and not _has_numerical_results(prev2_dir):
                pivot_prefix = (
                    "⚠ MANDATORY PIVOT: Two consecutive rounds produced zero numerical results.\n"
                    "The current experimental approach is blocked by tooling or environment issues.\n"
                    "This round MUST use packages confirmed in hw_profile.json available_packages.\n"
                    "Prefer the simplest approach that produces any numerical output — even just\n"
                    "basic statistics (mean, std, counts, correlations) computed with numpy/scipy/pandas.\n"
                    "ANY number in key_results.json is more valuable than another failed run.\n\n"
                )
                brief = pivot_prefix + brief

        if is_complete:
            _run_master_reporter(
                topic=topic,
                research_dir=str(research_dir),
                rounds_done=round_num,
                working_dir=working_dir,
                client=client,
                model=overrides.get("reporter") or ROLES["reporter"]["default_model"],
            )
            display.print_research_complete(round_num, str(research_dir))
            completed_early = True
            break

        display.print_round_done(round_num, str(round_dir))

    if not completed_early:
        _run_master_reporter(
            topic=topic,
            research_dir=str(research_dir),
            rounds_done=max_rounds,
            working_dir=working_dir,
            client=client,
            model=overrides.get("reporter") or ROLES["reporter"]["default_model"],
        )
        display.print_research_complete(max_rounds, str(research_dir))
