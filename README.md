<div align="center">

```
       ██████████████████
    ████████████████████████
   ██████████████████████████
  ████████████████████████████
  ██ ███████████ ███████████ ██
  ██ ██ ██◉██ ██ ██ ██◉██ ██ ██
  ██ ███████████ ███████████ ██
  ████████████████████████████
    █████  ▄▄▄▄▄▄▄▄▄  █████
    ████████████████████████
     ██████████████████████
  ████   ████   ████   ████
  ████   ████   ████   ████
 ╰████╯ ╰████╯ ╰████╯ ╰████╯
```

# OctoSlave 🐙

**Autonomous AI research & coding assistant powered by the [e-INFRA CZ](https://llm.ai.e-infra.cz) LLM platform**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-e--INFRA%20CZ-purple?style=flat-square)](https://llm.ai.e-infra.cz)

</div>

---

OctoSlave is a terminal-based autonomous agent that works like Claude Code — but runs entirely on the **e-INFRA CZ academic LLM infrastructure**. Give it a task or a research topic and it explores, searches the web, writes and runs code, debugs, and iterates until the job is done.

It also ships a **multi-agent long-research pipeline** (`/long-research`) that spawns a population of specialist agents — Researcher, Hypothesis Generator, Coder, Debugger, Evaluator, Orchestrator — and runs them in a loop over multiple rounds, producing rigorous, reproducible research with real data only.

---

## Contents

- [Features](#features)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Interactive TUI](#interactive-tui)
- [Slash commands](#slash-commands)
- [One-shot mode](#one-shot-mode)
- [Long-research pipeline](#long-research-pipeline)
- [Available models](#available-models)
- [Tools reference](#tools-reference)
- [Configuration](#configuration)
- [Project structure](#project-structure)
- [License](#license)

---

## Features

| | |
|---|---|
| **Autonomous agent loop** | Runs up to 80 iterations end-to-end until the task is complete |
| **Web research** | DuckDuckGo search + full-page text extraction from any URL |
| **File system & shell** | Read, write, edit files; run arbitrary shell commands; install packages |
| **PDF support** | Automatically extracts text from PDFs (no binary noise sent to the model) |
| **Streaming output** | See the model's reasoning and tool calls in real time |
| **Interactive TUI** | Persistent prompt with command history, bottom toolbar, keyboard shortcuts |
| **Multi-model** | Switch between Mistral, Qwen, DeepSeek, GPT-OSS and more mid-session |
| **Multi-agent research** | Full autonomous research pipeline with 6 specialist roles |
| **Context safety** | 50 K-char tool-result cap + graceful context-window-exceeded recovery |
| **Resumable** | Research runs persist to disk and can be resumed after interruption |

---

## Installation

**Requirements:** Python 3.10+, an [e-INFRA CZ LLM](https://llm.ai.e-infra.cz) API key

```bash
git clone https://github.com/karatedava/octoslave.git
cd octoslave
pip install -e .
```

> **Tip:** Use a virtual environment to keep dependencies isolated.
> ```bash
> python -m venv .venv && source .venv/bin/activate
> pip install -e .
> ```

### Set your API key

```bash
# Interactive setup wizard
ots config

# Or pass the key directly
ots config --api-key sk-YOUR_KEY

# Or export it for the session
export OCTOSLAVE_API_KEY=sk-YOUR_KEY
```

Config is stored at `~/.octoslave/config.json`. Environment variables always take precedence.

---

## Quick start

```bash
# Launch the interactive TUI
ots

# Run a single task and exit
ots run "build a Flask REST API for a todo app"

# Run against a specific directory and model
ots run "add pytest unit tests" --dir ~/my-project --model qwen3-coder

# Start a fully autonomous multi-round research session
ots
◆ /long-research "calibration methods for large language models" --rounds 4
```

---

## Interactive TUI

Running `ots` launches the full TUI:

```
╭─────────────────────────────────────────╮
│                                         │
│        ██████████████████               │
│     ████████████████████████            │
│   ██████████████████████████████        │
│   ...  (octopus mascot)  ...            │
│                                         │
│  OCTOSLAVE                              │
│  model mistral-small-4   dir ~/project  │
│  /help for commands                     │
│                                         │
╰─────────────────────────────────────────╯

◆ [mistral-small-4] _
```

- Type any task in natural language and press **Enter**
- The agent streams its thinking and tool calls live
- Follow up with more instructions in the same session — context is preserved
- Use `/` commands to control the session (see below)

**Keyboard shortcuts**

| Key | Action |
|-----|--------|
| `↑` / `↓` | Recall previous prompts from history |
| `Ctrl+C` | Cancel current generation (history preserved) |
| `Ctrl+D` | Exit |
| `Ctrl+L` | Clear terminal screen |

---

## Slash commands

| Command | Description |
|---------|-------------|
| `/model [name]` | Switch model; lists all available if no name given |
| `/dir [path]` | Change the active working directory |
| `/clear` | Clear screen and reset conversation history |
| `/compact` | Summarise history into a compact context block (saves tokens) |
| `/long-research TOPIC [flags]` | Launch the multi-agent research pipeline |
| `/help` | Show all commands and flags |
| `/exit` | Quit (also `Ctrl+D`) |

---

## One-shot mode

```bash
# Basic task
ots run "build a REST API for a todo app"

# Specify model and working directory
ots run "refactor the authentication module" \
  --model qwen3-coder \
  --dir /path/to/project

# Stay interactive after the task completes
ots run "set up a data processing pipeline for CSV files" -i

# Full flag reference
ots run --help
```

---

## Long-research pipeline

`/long-research` spawns a **population of 6 specialist agents** that collaborate over multiple rounds of fully autonomous research:

```
Round 1 ──────────────────────────────────────────────────────
  🔬 Researcher         Literature survey, verified datasets
  💡 Hypothesis Gen.    3–5 ranked, falsifiable hypotheses
  💻 Coder              Implements the recommended experiment (real data only)
  🐛 Debugger           Independently verifies code and results
  ⚖️  Evaluator          Critical scoring against SOTA benchmarks
  🧠 Orchestrator       Synthesises findings → writes brief for Round 2
Round 2 ──────────────────────────────────────────────────────
  ... (loop until complete or max rounds)
```

**Data integrity guarantee:** agents are explicitly forbidden from generating synthetic or dummy data as a substitute for real sources. If a dataset is unavailable, the step is logged as skipped, alternatives are searched, and the pipeline pivots — it never fabricates results.

### Usage

```
/long-research TOPIC [--rounds N] [--all MODEL] [--overseer MODEL] [--resume]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--rounds N` | `5` | Maximum number of research rounds |
| `--all MODEL` | *(per-role defaults)* | Use a single model for all 6 agents |
| `--overseer MODEL` | `mistral-small-4` | Override the orchestrator model only |
| `--resume` | off | Resume an interrupted run (skips agents whose output files exist) |

### Examples

```
/long-research "effect of batch size on transformer generalisation" --rounds 3

/long-research "protein folding accuracy of ESMFold vs AlphaFold2" \
  --rounds 5 \
  --all qwen3-coder \
  --overseer gpt-oss-120b

/long-research "RAG retrieval strategies for long documents" --resume
```

### Output structure

Each run creates a self-contained directory tree under `research/` in your working directory:

```
research/
├── findings.md                    ← cumulative findings across all rounds
├── round_001/
│   ├── 01_literature.md           ← papers, datasets (with access status)
│   ├── 02_hypotheses.md           ← ranked hypotheses + recommended experiment
│   ├── 03_code/
│   │   ├── *.py                   ← experiment scripts
│   │   ├── IMPLEMENTATION.md      ← approach, skipped steps, results summary
│   │   └── results/               ← all outputs (CSV, JSON, logs)
│   ├── 04_debug_report.md         ← bugs found/fixed, confidence score
│   ├── 05_evaluation.md           ← independent scores against SOTA
│   └── 06_synthesis.md            ← round summary + brief for next round
├── round_002/
│   └── ...
```

---

## Available models

Run `ots models` to see the current list. As of writing:

| Model | Strengths |
|-------|-----------|
| `mistral-small-4` | Fast, general tasks — **default** |
| `qwen3-coder` | Strong coding and tool use |
| `qwen3-coder-30b` | Larger coding model |
| `qwen3-coder-next` | Latest Qwen coder generation |
| `qwen3.5` | Balanced general + coding |
| `qwen3.5-122b` | Large general model |
| `gpt-oss-120b` | Large general model |
| `deepseek-v3.2` | Strong reasoning |
| `deepseek-v3.2-thinking` | Extended chain-of-thought |
| `kimi-k2.5` | Long context |
| `llama-4-scout-17b-16e-instruct` | Meta Llama 4 |
| `gemma4` | Google Gemma 4 |
| `glm-4.7` / `glm-5` | GLM series |
| `thinker` / `coder` / `agentic` / `mini` | Alias shortcuts |
| `redhatai-scout` | Red Hat AI Scout |

Switch mid-session with `/model qwen3-coder` or pass `-m MODEL` to any command.

---

## Tools reference

| Tool | Description |
|------|-------------|
| `read_file` | Read file contents (with optional offset/limit for large files); PDFs automatically extracted to text |
| `write_file` | Create or fully overwrite a file |
| `edit_file` | Targeted string replacement — safer than rewriting the whole file |
| `bash` | Run any shell command (installs, tests, builds, git, data processing) |
| `glob` | Find files by pattern, e.g. `**/*.py` |
| `grep` | Regex search across files with context lines |
| `list_dir` | List directory contents with sizes and modification times |
| `web_search` | DuckDuckGo search → titles, URLs, one-line snippets |
| `web_fetch` | Fetch URL → extracted readable text (strips JS/CSS/ads) |

---

## Configuration

| Mechanism | Precedence | Notes |
|-----------|-----------|-------|
| Environment variable | **Highest** | Overrides everything |
| `~/.octoslave/config.json` | Medium | Written by `ots config` |
| Built-in default | Lowest | `mistral-small-4`, e-INFRA CZ endpoint |

**Environment variables**

| Variable | Description |
|----------|-------------|
| `OCTOSLAVE_API_KEY` | API key |
| `OCTOSLAVE_BASE_URL` | Override the API base URL (default: `https://llm.ai.e-infra.cz/v1`) |
| `OCTOSLAVE_MODEL` | Override the default model |

**Interactive config wizard**

```bash
ots config          # guided setup
ots config --show   # print current config (key is masked)
```

---

## Project structure

```
octoslave/
├── octoslave/
│   ├── agent.py      # Core agent loop, system prompt, context management
│   ├── config.py     # Config loading/saving, model list
│   ├── display.py    # Rich TUI: mascot, banners, streaming, research display
│   ├── main.py       # Click CLI, interactive REPL, slash-command handler
│   ├── research.py   # Multi-agent long-research pipeline
│   └── tools.py      # All tool definitions and implementations
└── pyproject.toml
```

---

## License

MIT — see [LICENSE](LICENSE).

---
