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
from datetime import datetime
from pathlib import Path

from openai import OpenAI, BadRequestError

from . import display
from .agent import _cap_result
from .tools import TOOL_DEFINITIONS, execute_tool

# ---------------------------------------------------------------------------
# Role registry
# ---------------------------------------------------------------------------

ROLES: dict[str, dict] = {
    "researcher": {
        "label": "Researcher",
        "icon": "🔬",
        "color": "bold cyan",
        "default_model": "qwen3.5-122b",           # large — fast reading + search
        "max_iter": 10,                             # targeted scout, not full survey
        "tools": ["read_file", "write_file", "web_search", "web_fetch",
                  "list_dir", "glob", "bash"],
    },
    "hypothesis": {
        "label": "Experiment Designer",
        "icon": "💡",
        "color": "bold bright_magenta",
        "default_model": "deepseek-v3.2-thinking",  # thinking — commit to the right experiment
        "max_iter": 8,
        "tools": ["read_file", "write_file", "list_dir", "glob"],
    },
    "coder": {
        "label": "Coder",
        "icon": "💻",
        "color": "bold green",
        "default_model": "qwen3-coder-30b",         # large code model — fewer mistakes
        "max_iter": 50,
        "tools": ["read_file", "write_file", "edit_file", "bash",
                  "glob", "grep", "list_dir"],
    },
    "debugger": {
        "label": "Debugger",
        "icon": "🐛",
        "color": "bold red",
        "default_model": "qwen3-coder-30b",         # same coder — knows the code
        "max_iter": 20,
        "tools": ["read_file", "write_file", "edit_file", "bash",
                  "glob", "grep", "list_dir"],
    },
    "evaluator": {
        "label": "Evaluator",
        "icon": "⚖️ ",
        "color": "bold yellow",
        "default_model": "deepseek-v3.2-thinking",  # thinking — rigorous scientific judgement
        "max_iter": 10,
        "tools": ["read_file", "bash", "write_file", "list_dir",
                  "web_search", "glob"],
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
        "default_model": "gpt-oss-120b",            # large general — clean HTML/writing
        "max_iter": 25,
        "tools": ["read_file", "write_file", "bash", "list_dir", "glob"],
    },
}

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
NEXT_BRIEF_MARKER = "## NEXT_ROUND_BRIEF"
COMPLETE_MARKER = "## STATUS: COMPLETE"


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

EFFICIENCY RULES (critical — small models, limited context):
- Act immediately. No preamble, no "I will now...", no narration.
- Read only the specific section you need from each file, not the whole file.
- Write output files once, concisely. Do not draft, then rewrite.
- Stop as soon as your required output file is written and verified non-empty.
---
"""

_ROLE_PROMPTS: dict[str, str] = {

"researcher": """\
YOUR MISSION
Fast, targeted intelligence-gathering pass. Equip the Experiment Designer with
exactly what they need to commit to ONE concrete experiment. 3 sharp sources
beat 10 shallow ones. Total output: under 500 words.

STEPS
1. Round > 1: read {research_dir}/findings.md (## Key Findings section only) to
   know what was tried. Round 1: skip.
2. Run 2–3 targeted web searches on the round brief. Fetch the single most
   useful page per search. Stop when you can answer: (a) best known result /
   method, (b) which dataset is accessible right now.
3. For each dataset candidate: fetch its landing page. Label it:
   ACCESSIBLE | REQUIRES_SIGNUP | PAYWALLED | UNAVAILABLE. Only list confirmed ones.

OUTPUT — write ONE file: {round_dir}/01_literature.md
Keep every section to bullet points — no prose paragraphs except the last one.

  ## SOTA Summary     (2–3 bullets: best result, method, benchmark)
  ## Available Datasets (name · direct URL · size · licence · ACCESS STATUS)
  ## Baselines        (concrete numbers only, e.g. "ResNet-50: 76.1% top-1")

  ## FOR THE EXPERIMENT DESIGNER
  [1 focused paragraph: which gap to target, which dataset to use (URL),
   what baseline to beat, key gotcha. Be direct — the next agent reads ONLY
   this section.]
""",

"hypothesis": """\
YOUR MISSION
Design exactly ONE concrete, executable experiment. Be decisive.
Total output: under 400 words.

STEPS
1. Read ONLY the ## FOR THE EXPERIMENT DESIGNER section from
   {round_dir}/01_literature.md — that is your entire input.
2. Round > 1: also read {research_dir}/findings.md (## What Failed section only)
   to avoid repeating failures.
3. Think, commit, write. No drafting.

OUTPUT — write ONE file: {round_dir}/02_experiment.md

  ## Experiment: <short name>
  **Hypothesis**: one falsifiable claim
  **Success metric**: specific threshold (e.g. "F1 > 0.82 on test set")
  **Failure threshold**: below this = wrong approach

  ## Algorithm / Approach
  [Pseudocode or numbered steps. Precise enough that the Coder needs no guessing.
   Include: method, loss, key hyperparameters, eval protocol. Max 10 lines.]

  ## Data Plan
  **Primary**: <name> · <direct download URL> · <format>
  **Fallback**: <alternative> · <URL>
  (Only sources confirmed ACCESSIBLE in 01_literature.md.)

  ## Expected Output Files
  - results/key_results.json  → {{"metric": <name>, "value": <float>, "baseline": <float>}}
  - results/main_plot.png
  - results/summary_figure.png

  ## FOR THE CODER
  [2 sentences max: where to start, the single most critical implementation detail,
   what "done" looks like.]
""",

"coder": """\
YOUR MISSION
Implement the experiment. Write real, working, runnable code.
Produce concrete results from real data.

STEPS
1. Read ONLY ## FOR THE CODER and ## Data Plan from {round_dir}/02_experiment.md.
2. Read ONLY ## Available Datasets from {round_dir}/01_literature.md to confirm
   which dataset URLs are VERIFIED ACCESSIBLE.
3. Read {research_dir}/hw_profile.json — hardware is already probed by the
   pipeline. Use cuda_available, cuda_devices[].vram_gb, ram_total_gb, cpu_count
   to set batch sizes, device placement, and parallelism. Do NOT re-probe.
4. Read any existing code in {round_dir}/03_code/ if this is a continuation.
5. Execute:
   a. Create {round_dir}/03_code/ directory.
   b. Download / access the verified dataset(s).
   c. Write modular Python. Install packages with uv (see below).
   d. Run the code. Fix runtime errors.
   e. Save ALL output (metrics, plots) to {round_dir}/03_code/results/.
6. Write {round_dir}/03_code/IMPLEMENTATION.md — keep it SHORT (under 300 words):
   - Hardware used (device, batch size chosen)
   - Data source + how it was accessed
   - Approach in 3–5 bullet points
   - Results summary (key numbers)
   - Any skipped steps + reason (see FAILURE PROTOCOL)

GPU RULES (if CUDA available per hw_profile.json — no exceptions)
- device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
- Move models AND tensors: .to(device). Log "Using device: {device}" at runtime.
- PyTorch: use autocast("cuda") + GradScaler; num_workers≥2; pin_memory=True.
- Batch size: target 70–80% of vram_gb from hw_profile.
- HuggingFace: device_map="auto". scikit-learn/XGBoost: device="cuda".
- Log peak_vram_gb to results/ via torch.cuda.max_memory_allocated()/1e9.

VISUALISATION (save to {round_dir}/03_code/results/)
- Main results plot + summary_figure.png (2–4 subplot overview). Both required.
- 150 dpi PNG. Title, axis labels, legend. Use tight_layout() + savefig().

PACKAGES — use `uv pip install <pkg>`. Fallback: pip. Note failures in IMPLEMENTATION.md.

ABSOLUTE RULES — READ CAREFULLY
- NEVER generate synthetic or dummy data as a substitute for real data.
  Synthetic stand-ins are scientifically invalid and mislead future agents.
- NEVER fabricate results or outputs. Every number in results/ must come from
  real computation on real data.
- If a data source is unavailable (network error, API down, auth required):
    1. Log the failure clearly in IMPLEMENTATION.md under ## Skipped Steps.
    2. Do NOT proceed with that experiment using fake data.
    3. Instead: search for an alternative real dataset that tests the same
       hypothesis (web_search). Try at least 2–3 alternatives.
    4. If NO real data can be obtained for a given hypothesis, mark that
       experiment as BLOCKED in IMPLEMENTATION.md and pivot to a different
       hypothesis from {round_dir}/02_experiment.md that CAN use available data.
    5. If ALL hypotheses are blocked due to data access, implement the
       methodological scaffolding (data loading, model, evaluation pipeline)
       using a small well-known public benchmark (e.g. UCI, HuggingFace, NCBI)
       that IS accessible — document the substitution clearly.
- Quantitative results MUST be saved (JSON / CSV / text).
- Every script that IS run must complete without error.
- If an approach fails after 3 debugging attempts, pivot and document why.
""",

"debugger": """\
YOUR MISSION
Verify code correctness and result validity. Be skeptical. Total report: under 350 words.

STEPS
1. Read IMPLEMENTATION.md and the main script(s) under {round_dir}/03_code/.
2. Run the main script. Inspect output in 03_code/results/.
3. Check — each is a potential one-line report entry:
   - SYNTHETIC DATA: any fabricated/placeholder data instead of real → CRITICAL
   - GPU UNDERUSE: if hw_profile.json shows CUDA available but "Using device: cpu"
     appears in output → CRITICAL (fix: add .to(device), rerun)
   - Runtime errors, off-by-one, data leakage, wrong metrics
   - Results implausibly good/bad (may indicate fake data)
4. Fix each bug (edit_file / bash). Re-run to confirm.

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

STEPS
1. Read: {round_dir}/03_code/IMPLEMENTATION.md and {round_dir}/04_debug_report.md.
   These are your primary inputs. Read {round_dir}/02_experiment.md only for
   the success metric. Read {round_dir}/01_literature.md ONLY if you need a
   SOTA number for comparison.
2. Check results/ for key_results.json and plots. Verify numbers are plausible.
3. One web_search max if you need a SOTA reference.

OUTPUT — write ONE file: {round_dir}/05_evaluation.md
Format: score on the SAME line as the heading, then ONE sentence commentary.

  ## Literature Quality      X/10 — <one sentence>
  ## Hypothesis Quality      X/10 — <one sentence>
  ## Implementation Quality  X/10 — <one sentence>
  ## Results Validity        X/10 — <one sentence>
  ## Overall Score           X/10
  ## Critical Weaknesses     (bullet list, max 3 items)
  ## Recommended Next Steps  (bullet list, max 3 specific actionable items)

SCORES CHART (only if results/key_results.json exists)
- Write + run a minimal Python script → saves {round_dir}/05_scores_chart.png.
- Simple bar chart, 4 bars, labels, colour-coded (green≥7, amber4–6, red≤3).
- If no results exist, skip the chart entirely.

SCORING RULES
- Synthetic/fabricated data → Results Validity capped at 1/10.
- Be harsh. A generous score on mediocre work wastes the next round's effort.
""",

"orchestrator": """\
YOUR MISSION
Synthesise this round. Write the brief that drives the next round.
Total output file: under 500 words.

STEPS
1. Read PRIMARILY {round_dir}/05_evaluation.md — it already summarises the work.
   Read {round_dir}/03_code/IMPLEMENTATION.md for specific technical details only
   if the evaluation references something you need to clarify.
2. Read {research_dir}/findings.md (## What Failed section only) if round > 1.
3. Write ONE file: {round_dir}/06_synthesis.md. Do NOT touch findings.md.

STRUCTURE — short bullets, not paragraphs:

  ## Round Summary        (2–3 bullets)
  ## Key Findings         (2–3 bullets with numbers where possible)
  ## What Worked          (1–3 bullets)
  ## What Failed / Gaps   (1–3 bullets)
  ## Updated Research Direction  (1–2 sentences)

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
  4. Experiment (hypothesis, success metric, data used)
  5. Implementation (approach bullets, data source, any skipped steps)
  6. Results & Plots (ALL PNGs base64-embedded, 2-col grid, 1-line captions)
  7. Evaluation (scores table colour-coded ≥7 green / 4–6 amber / ≤3 red;
     embed 05_scores_chart.png if it exists)
  8. Next Direction (NEXT_ROUND_BRIEF as callout box)
  9. Footer (round, topic, timestamp)

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
) -> str:
    role_cfg = ROLES[role]
    final_tag = "← FINAL ROUND — prioritise conclusions over exploration" if is_final else ""
    # Cap brief length for roles that only need the direction, not full synthesis prose
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
    body = _ROLE_PROMPTS[role].format(
        round_dir=round_dir,
        research_dir=research_dir,
        working_dir=working_dir,
        next_brief_marker=NEXT_BRIEF_MARKER,
        complete_marker=COMPLETE_MARKER,
    )
    return header + body


# ---------------------------------------------------------------------------
# Filtered tool list per role
# ---------------------------------------------------------------------------

def _tools_for_role(role: str) -> list[dict]:
    allowed = set(ROLES[role]["tools"])
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
) -> bool:
    """
    Run one specialist agent for one round.
    Returns True on success, False if a fatal error occurred.
    """
    cfg = ROLES[role]
    tools = _tools_for_role(role)
    max_iter = cfg["max_iter"]

    display.print_agent_banner(role, model, round_num, max_rounds)

    system_prompt = _build_system_prompt(
        role, topic, round_num, max_rounds,
        round_dir, research_dir, working_dir, brief,
        is_final=(round_num == max_rounds),
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

    for iteration in range(1, max_iter + 1):
        try:
            response = _stream_completion_with_tools(client, model, messages, tools)
        except BadRequestError as e:
            err = str(e)
            if "ContextWindow" in err or "context" in err.lower():
                display.print_error(
                    f"[{cfg['label']}] Context window exceeded — "
                    "trimming oldest tool results and retrying."
                )
                messages = _trim_messages(messages)
                continue
            display.print_error(f"[{cfg['label']}] API error: {e}")
            return False
        except KeyboardInterrupt:
            display.stream_end(False)
            display.console.print("\n[dim]Interrupted.[/dim]")
            raise

        content = response["content"]
        tool_calls = response["tool_calls"]
        finish_reason = response["finish_reason"]

        assistant_msg: dict = {"role": "assistant", "content": content or None}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if not tool_calls or finish_reason == "stop":
            break

        display.print_separator()
        for tc in tool_calls:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}

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

    elapsed = time.time() - t0
    display.print_agent_done(role, elapsed, iteration)
    return True


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

    # Extract overall score from evaluation
    score_match = re.search(r"##\s*Overall Score[^\n]*\n+([^\n]+)", evaluation)
    score_str   = score_match.group(1).strip() if score_match else "N/A"

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

def _trim_messages(messages: list[dict]) -> list[dict]:
    """
    Remove the oldest tool-result messages (pairs) to free context space.
    Always preserve system + first user message.
    """
    system = messages[:2]
    rest = messages[2:]

    # Drop oldest tool result
    for i, m in enumerate(rest):
        if m.get("role") == "tool":
            rest = rest[:max(0, i - 1)] + rest[i + 1:]
            break

    return system + rest


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
  4. Research Timeline table: Round | Hypothesis | Score | Status
  5. Cumulative Findings (from findings.md, as cards)
  6. Round Deep Dives — one <details> per round:
       hypothesis · implementation summary · ALL result plots (2-col, base64)
       · scores chart · what worked / failed
  7. Score Progression chart (generate with matplotlib: round on x, score on y)
  8. Key Visualisations Gallery (summary_figure.png from each round, full-width)
  9. Conclusions & Next Steps (from final synthesis)
  Footer: topic · timestamp · "Generated by OctoSlave"

DESIGN: dark header (#0d1117), white cards, Inter (CDN OK), max-width 1200px,
base64 all images, collapsible rounds via <details>/<summary>.
Script: stdlib + matplotlib only. Print output path on success.
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
    cfg = ROLES["reporter"]
    tools = _tools_for_role("reporter")

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
    for iteration in range(1, cfg["max_iter"] + 1):
        try:
            response = _stream_completion_with_tools(client, model, messages, tools)
        except BadRequestError as e:
            err = str(e)
            if "ContextWindow" in err or "context" in err.lower():
                messages = _trim_messages(messages)
                continue
            display.print_error(f"[Master Reporter] API error: {e}")
            return
        except KeyboardInterrupt:
            display.stream_end(False)
            display.console.print("\n[dim]Master report interrupted.[/dim]")
            return

        content = response["content"]
        tool_calls = response["tool_calls"]
        finish_reason = response["finish_reason"]

        assistant_msg: dict = {"role": "assistant", "content": content or None}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if not tool_calls or finish_reason == "stop":
            break

        display.print_separator()
        for tc in tool_calls:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
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
        display.print_info(
            f"  [bold bright_cyan]Master report → {final_report}[/bold bright_cyan]"
        )


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
        "import json, platform, os, sys\n"
        "info = {'python': sys.version.split()[0], 'platform': platform.platform(), "
        "'cpu_count': os.cpu_count()}\n"
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
        "    r = __import__('subprocess').run(['nvidia-smi','--query-gpu=name,memory.total,memory.free',"
        "'--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)\n"
        "    if r.returncode==0: info['nvidia_smi'] = r.stdout.strip()\n"
        "except Exception: pass\n"
        "print(json.dumps(info))\n"
    )

    profile: dict = {}
    try:
        result = _sp.run(
            ["python3", "-c", script],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            profile = json.loads(result.stdout.strip())
    except Exception:
        pass

    hw_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    # Pretty-print hardware summary
    cuda = profile.get("cuda_available", False)
    devices = profile.get("cuda_devices", [])
    ram = profile.get("ram_total_gb", "?")
    cpus = profile.get("cpu_count", "?")

    if cuda and devices:
        gpu_str = ", ".join(f"{d['name']} ({d['vram_gb']} GB)" for d in devices)
        display.print_info(f"  Hardware: {cpus} CPU cores, {ram} GB RAM, "
                           f"[bold bright_green]CUDA ✓[/bold bright_green] {gpu_str}")
    else:
        display.print_info(f"  Hardware: {cpus} CPU cores, {ram} GB RAM, "
                           f"[dim]no CUDA GPU detected[/dim]")

    return profile


def run_long_research(
    topic: str,
    working_dir: str,
    client: OpenAI,
    max_rounds: int = 5,
    model_overrides: dict[str, str] | None = None,
    resume: bool = False,
) -> None:
    """
    Run the full autonomous multi-agent research pipeline.

    Args:
        topic:          The research topic / goal.
        working_dir:    The project working directory.
        client:         Authenticated OpenAI client.
        max_rounds:     Maximum number of research rounds.
        model_overrides: Per-role model overrides, e.g. {"coder": "qwen3-coder-30b"}.
        resume:         If True, skip rounds whose output files already exist.
    """
    overrides = model_overrides or {}
    research_dir = Path(working_dir) / "research"
    research_dir.mkdir(parents=True, exist_ok=True)

    display.print_research_start(topic, max_rounds, ROLES, overrides)

    # Probe hardware once; result is written to research_dir/hw_profile.json
    # and read by the coder/debugger agents in every subsequent round.
    _probe_hardware(str(research_dir))

    # Initial brief
    brief = (
        f"Initial research round. Conduct a broad literature survey on: {topic}\n"
        "Identify key papers, available datasets, existing methods, and open problems.\n"
        "Generate first hypotheses and implement the most promising experiment."
    )

    completed_early = False

    for round_num in range(1, max_rounds + 1):
        round_dir = research_dir / f"round_{round_num:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        display.print_round_header(round_num, max_rounds, str(round_dir))

        for role in PIPELINE:
            model = overrides.get(role) or ROLES[role]["default_model"]

            # Resumability: skip if output already exists
            expected = OUTPUT_FILES[role]
            expected_path = round_dir / expected
            if resume and expected_path.exists():
                display.print_info(
                    f"  ↩  {ROLES[role]['label']} output found — skipping."
                )
                continue

            try:
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

        # Update findings.md from round outputs — pipeline-owned, not LLM-owned
        _update_findings(
            research_dir=str(research_dir),
            round_num=round_num,
            round_dir=str(round_dir),
            topic=topic,
        )

        # Parse orchestrator synthesis → next brief
        synthesis_path = round_dir / OUTPUT_FILES["orchestrator"]
        brief, is_complete = _parse_synthesis(str(synthesis_path))

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
