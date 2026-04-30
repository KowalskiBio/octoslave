import json
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# e-INFRA CZ defaults
# ---------------------------------------------------------------------------

CONFIG_DIR = Path.home() / ".octoslave"
CONFIG_FILE = CONFIG_DIR / "config.json"

BASE_URL = "https://llm.ai.e-infra.cz/v1"
DEFAULT_MODEL = "deepseek-v3.2"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NIM_DEFAULT_MODEL = "meta/llama-3.3-70b-instruct"

KNOWN_MODELS = [
    "mistral-small-4",
    "qwen3.5",
    "qwen3.5-122b",
    "qwen3-coder",
    "qwen3-coder-30b",
    "qwen3-coder-next",
    "gpt-oss-120b",
    "deepseek-v3.2",
    "deepseek-v3.2-thinking",
    "kimi-k2.5",
    "llama-4-scout-17b-16e-instruct",
    "gemma4",
    "glm-4.7",
    "glm-5",
    "thinker",
    "coder",
    "agentic",
    "mini",
    "redhatai-scout",
]

# Static fallback list — used when the API key is absent or the /models call fails.
# Excludes known-EOL models (deepseek-ai/deepseek-r1 EOL 2026-01-26, etc.)
NIM_KNOWN_MODELS = [
    # Llama 4 (released April 2025)
    "meta/llama-4-scout-17b-16e-instruct",
    "meta/llama-4-maverick-17b-128e-instruct",
    # NVIDIA Nemotron (NVIDIA-tuned SOTA)
    "nvidia/llama-3.3-nemotron-super-49b-v1",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    # Llama 3.x
    "meta/llama-3.3-70b-instruct",
    "meta/llama-3.1-405b-instruct",
    # Mistral
    "mistralai/mistral-large-2-instruct",
    # Qwen
    "qwen/qwen2.5-72b-instruct",
    # Phi / Gemma
    "microsoft/phi-4",
    "google/gemma-3-27b-it",
    # DeepSeek distilled (R1 base is EOL; distilled variants may still be live)
    "deepseek-ai/deepseek-r1-distill-llama-70b",
]


# ---------------------------------------------------------------------------
# NVIDIA NIM helpers
# ---------------------------------------------------------------------------

def nim_list_models(nim_url: str = NIM_BASE_URL, nim_api_key: str = "") -> list[str]:
    """
    Query NIM's OpenAI-compatible /v1/models endpoint and return model IDs.
    Returns an empty list on any error so callers can fall back to NIM_KNOWN_MODELS.
    """
    if not nim_api_key:
        return []
    try:
        import urllib.request, json as _json
        req = urllib.request.Request(
            f"{nim_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {nim_api_key}"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = _json.loads(resp.read())
            ids = [m["id"] for m in data.get("data", []) if m.get("id")]
            return sorted(ids) if ids else []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

def ollama_is_running(ollama_url: str = OLLAMA_BASE_URL) -> bool:
    """Return True if an Ollama instance is reachable at ollama_url."""
    try:
        import urllib.request
        base = ollama_url.rstrip("/").removesuffix("/v1")
        req = urllib.request.urlopen(f"{base}/api/tags", timeout=3)
        return req.status == 200
    except Exception:
        return False


def ollama_list_models(ollama_url: str = OLLAMA_BASE_URL) -> list[str]:
    """
    Return the list of model names already pulled in Ollama.
    Returns an empty list if Ollama is unreachable.
    """
    try:
        import urllib.request, json as _json
        base = ollama_url.rstrip("/").removesuffix("/v1")
        with urllib.request.urlopen(f"{base}/api/tags", timeout=5) as resp:
            data = _json.loads(resp.read())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def ollama_pull_model(model_name: str, ollama_url: str = OLLAMA_BASE_URL) -> bool:
    """
    Pull a model via the Ollama REST API (streaming).
    Prints progress lines to stdout. Returns True on success.
    """
    try:
        import urllib.request, json as _json
        base = ollama_url.rstrip("/").removesuffix("/v1")
        payload = _json.dumps({"name": model_name, "stream": True}).encode()
        req = urllib.request.Request(
            f"{base}/api/pull",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw_line in resp:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    obj = _json.loads(line)
                    status = obj.get("status", "")
                    if "total" in obj and "completed" in obj:
                        pct = int(obj["completed"] / obj["total"] * 100)
                        print(f"\r  {status}: {pct}%", end="", flush=True)
                    else:
                        print(f"\r  {status}        ", end="", flush=True)
                except Exception:
                    pass
        print()  # newline after progress
        return True
    except Exception as e:
        print(f"\n  Pull failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Role → local model assignment for long-research
# ---------------------------------------------------------------------------

# How the 7 pipeline roles are mapped when ≤3 local models are available.
# Priority tiers: if 3 models available, tier-A gets model[0], tier-B gets
# model[1], tier-C gets model[2].  Fewer models collapse tiers.
_ROLE_TIERS: dict[str, int] = {
    # Tier A — primary reasoning (best model)
    "orchestrator": 0,
    "evaluator":    0,
    # Tier B — coding / implementation
    "coder":        1,
    "debugger":     1,
    "reporter":     1,
    # Tier C — reading / writing (lightest model is fine)
    "researcher":   2,
    "hypothesis":   2,
}


def assign_local_models(pulled_models: list[str]) -> dict[str, str]:
    """
    Given a list of pulled Ollama model names (up to 3 are used),
    return a role → model mapping for the research pipeline.
    """
    if not pulled_models:
        raise ValueError("No Ollama models available.")

    # Use at most 3 distinct models
    available = pulled_models[:3]
    n = len(available)

    mapping: dict[str, str] = {}
    for role, tier in _ROLE_TIERS.items():
        # Collapse tiers if fewer than 3 models
        idx = min(tier, n - 1)
        mapping[role] = available[idx]
    return mapping


# ---------------------------------------------------------------------------
# Config load / save
# ---------------------------------------------------------------------------

def list_models(cfg: dict | None = None) -> list[str]:
    """
    Return available model names.
    - einfra: static KNOWN_MODELS (no /models endpoint)
    - ollama: poll the local server
    - nim: query /v1/models live; fall back to NIM_KNOWN_MODELS on failure
    """
    if cfg is None:
        cfg = {}
    if cfg.get("backend") == "ollama":
        pulled = ollama_list_models(cfg.get("ollama_url", OLLAMA_BASE_URL))
        return pulled if pulled else KNOWN_MODELS
    if cfg.get("backend") == "nim":
        live = nim_list_models(
            cfg.get("nim_url", NIM_BASE_URL),
            cfg.get("nim_api_key", ""),
        )
        return live if live else list(NIM_KNOWN_MODELS)
    return list(KNOWN_MODELS)


def load_config() -> dict:
    config = {
        "api_key": "",
        "base_url": BASE_URL,
        "default_model": DEFAULT_MODEL,
        "backend": "einfra",        # "einfra" | "ollama" | "nim"
        "ollama_url": OLLAMA_BASE_URL,
        "permission_mode": "autonomous",  # "autonomous" | "controlled" | "supervised"
        "nim_api_key": "",
        "nim_url": NIM_BASE_URL,
    }
    # Env vars override config file
    if os.environ.get("OCTOSLAVE_API_KEY"):
        config["api_key"] = os.environ["OCTOSLAVE_API_KEY"]
    if os.environ.get("OCTOSLAVE_BASE_URL"):
        config["base_url"] = os.environ["OCTOSLAVE_BASE_URL"]
    if os.environ.get("OCTOSLAVE_MODEL"):
        config["default_model"] = os.environ["OCTOSLAVE_MODEL"]
    if os.environ.get("OCTOSLAVE_BACKEND"):
        config["backend"] = os.environ["OCTOSLAVE_BACKEND"]
    if os.environ.get("OCTOSLAVE_OLLAMA_URL"):
        config["ollama_url"] = os.environ["OCTOSLAVE_OLLAMA_URL"]
    if os.environ.get("OCTOSLAVE_PERMISSION_MODE"):
        config["permission_mode"] = os.environ["OCTOSLAVE_PERMISSION_MODE"]
    if os.environ.get("OCTOSLAVE_NIM_API_KEY"):
        config["nim_api_key"] = os.environ["OCTOSLAVE_NIM_API_KEY"]
    if os.environ.get("OCTOSLAVE_NIM_URL"):
        config["nim_url"] = os.environ["OCTOSLAVE_NIM_URL"]

    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                saved = json.load(f)
            _env_keys = {
                "api_key":          "OCTOSLAVE_API_KEY",
                "base_url":         "OCTOSLAVE_BASE_URL",
                "default_model":    "OCTOSLAVE_MODEL",
                "backend":          "OCTOSLAVE_BACKEND",
                "ollama_url":       "OCTOSLAVE_OLLAMA_URL",
                "permission_mode":  "OCTOSLAVE_PERMISSION_MODE",
                "nim_api_key":      "OCTOSLAVE_NIM_API_KEY",
                "nim_url":          "OCTOSLAVE_NIM_URL",
            }
            for key, env_var in _env_keys.items():
                if not os.environ.get(env_var) and saved.get(key):
                    config[key] = saved[key]
        except (json.JSONDecodeError, OSError):
            pass

    return config


def save_config(
    api_key: str,
    base_url: str = BASE_URL,
    default_model: str = DEFAULT_MODEL,
    backend: str = "einfra",
    ollama_url: str = OLLAMA_BASE_URL,
    permission_mode: str = "autonomous",
    nim_api_key: str | None = None,
    nim_url: str | None = None,
):
    # Load existing config to preserve nim values when not explicitly provided
    _existing: dict = {}
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                _existing = json.load(f)
        except Exception:
            pass

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "api_key": api_key,
        "base_url": base_url,
        "default_model": default_model,
        "backend": backend,
        "ollama_url": ollama_url,
        "permission_mode": permission_mode,
        "nim_api_key": nim_api_key if nim_api_key is not None else _existing.get("nim_api_key", ""),
        "nim_url": nim_url if nim_url is not None else _existing.get("nim_url", NIM_BASE_URL),
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(data, f, indent=2)
