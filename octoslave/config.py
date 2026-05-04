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
NIM_DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b"

KNOWN_MODELS = [
    "deepseek-v3.2",
    "deepseek-v3.2-thinking",
    "qwen3.5",
    "qwen3.5-122b",
    "qwen3-coder",
    "qwen3-coder-30b",
    "qwen3-coder-next",
    "gpt-oss-120b",
    "kimi-k2.5",
    "kimi-k2.6",
    "mistral-medium-3.5",
    "llama-4-scout-17b-16e-instruct",
    "gemma4",
    "glm-4.7",
    "glm-5",
    "glm-5.1",
    "thinker",
    "coder",
    "agentic",
    "mini",
    "redhatai-scout",
]

# Static fallback list — used when the API key is absent or the /models call fails.
# Only includes models confirmed accessible on a standard (free) NIM account.
# NOTE: The live /v1/models catalog contains 100+ models; many require paid tiers
#       or specific account permissions. This list is curated for general access.
NIM_KNOWN_MODELS = [
    # Llama 4
    "meta/llama-4-maverick-17b-128e-instruct",
    # NVIDIA Nemotron (free tier — confirmed working)
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/llama-3.3-nemotron-super-49b-v1",
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "nvidia/llama-3.1-nemotron-nano-8b-v1",
    # Llama 3.x
    "meta/llama-3.3-70b-instruct",
    "meta/llama-3.1-405b-instruct",
    "meta/llama-3.1-70b-instruct",
    # Qwen 3
    "qwen/qwen3.5-122b-a10b",
    "qwen/qwen3-coder-480b-a35b-instruct",
    # DeepSeek (correct NIM model IDs)
    "deepseek-ai/deepseek-v3.2",
    # Gemma / Phi
    "google/gemma-3-27b-it",
    "google/gemma-4-31b-it",
    "microsoft/phi-4-mini-instruct",
    # Mistral (may require paid tier on some accounts)
    "mistralai/mistral-large-2-instruct",
    "mistralai/mistral-small-4-119b-2603",
]

# ---------------------------------------------------------------------------
# Per-backend default role models for the research pipeline
# ---------------------------------------------------------------------------

PIPELINE_ROLES = (
    "researcher", "hypothesis", "coder", "debugger",
    "evaluator", "orchestrator", "reporter", "merger",
)

EINFRA_ROLE_MODELS: dict[str, str] = {
    "researcher":   "deepseek-v3.2-thinking",
    "hypothesis":   "deepseek-v3.2-thinking",
    "coder":        "qwen3-coder-30b",
    "debugger":     "qwen3-coder-30b",
    "evaluator":    "deepseek-v3.2-thinking",
    "orchestrator": "deepseek-v3.2",
    "reporter":     "deepseek-v3.2",
    "merger":       "deepseek-v3.2",
}

NIM_ROLE_MODELS: dict[str, str] = {
    # Heavy reasoning roles — largest confirmed-stable NVIDIA model
    "researcher":   "nvidia/nemotron-3-super-120b-a12b",
    "hypothesis":   "nvidia/nemotron-3-super-120b-a12b",
    "coder":        "nvidia/nemotron-3-super-120b-a12b",
    "debugger":     "nvidia/nemotron-3-super-120b-a12b",
    "evaluator":    "nvidia/nemotron-3-super-120b-a12b",
    # Lighter synthesis/writing roles — 49B, confirmed stable
    "orchestrator": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "reporter":     "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "merger":       "nvidia/llama-3.3-nemotron-super-49b-v1.5",
}


# ---------------------------------------------------------------------------
# e-INFRA CZ live model query
# ---------------------------------------------------------------------------

def einfra_list_models(base_url: str = BASE_URL, api_key: str = "") -> list[str]:
    """
    Query e-INFRA CZ's OpenAI-compatible /v1/models endpoint and return model IDs.
    Returns an empty list on any error so callers can fall back to KNOWN_MODELS.
    """
    if not api_key:
        return []
    try:
        import urllib.request, json as _json
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = _json.loads(resp.read())
            ids = [m["id"] for m in data.get("data", []) if m.get("id")]
            return sorted(ids) if ids else []
    except Exception:
        return []


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
# Tool-calling capability ordering for local models
# ---------------------------------------------------------------------------

# Model name prefixes ranked best→worst for OpenAI-format tool calling.
# Models not matching any prefix are treated as lowest priority.
_TOOL_CALL_RANK: list[str] = [
    "qwen2.5",
    "qwen3",
    "llama3.1",
    "llama3.2",
    "llama3.3",
    "mistral-nemo",
    "mistral-small",
    "granite3",
    "phi4",
    "phi3",
    "llama3",
    # Below here: known poor/no tool-calling support
    "gemma",
    "mistral",   # plain mistral:7b doesn't support JSON tool calls reliably
]


def _tool_call_score(model_name: str) -> int:
    """Lower score = better tool calling. Unknown models get worst score."""
    name = model_name.lower().split(":")[0]  # strip tag like :7b
    for i, prefix in enumerate(_TOOL_CALL_RANK):
        if name.startswith(prefix):
            return i
    return len(_TOOL_CALL_RANK)


def sort_by_tool_calling(models: list[str]) -> list[str]:
    """Return models sorted best-first for tool calling support."""
    return sorted(models, key=_tool_call_score)


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
# Role model helpers
# ---------------------------------------------------------------------------

def get_role_models(cfg: dict | None = None) -> dict[str, str]:
    """
    Return role → model mapping for the research pipeline.
    Starts from per-backend defaults, then applies any custom overrides
    stored in the config under 'role_models_<backend>'.
    """
    if cfg is None:
        cfg = {}
    backend = cfg.get("backend", "einfra")

    if backend == "nim":
        base = dict(NIM_ROLE_MODELS)
    elif backend == "ollama":
        pulled = ollama_list_models(cfg.get("ollama_url", OLLAMA_BASE_URL))
        base = assign_local_models(pulled) if pulled else dict(EINFRA_ROLE_MODELS)
    else:  # einfra
        base = dict(EINFRA_ROLE_MODELS)

    # Apply any per-backend custom overrides saved by the user
    custom = cfg.get(f"role_models_{backend}") or {}
    base.update(custom)
    return base


def save_role_model(role: str, model: str, backend: str) -> None:
    """Persist a single per-role model override for the given backend."""
    cfg = load_config()
    key = f"role_models_{backend}"
    role_models = dict(cfg.get(key) or {})
    role_models[role] = model
    save_config(
        api_key=cfg.get("api_key", ""),
        base_url=cfg.get("base_url", BASE_URL),
        default_model=cfg.get("default_model", DEFAULT_MODEL),
        backend=cfg.get("backend", "einfra"),
        ollama_url=cfg.get("ollama_url", OLLAMA_BASE_URL),
        permission_mode=cfg.get("permission_mode", "autonomous"),
        nim_api_key=cfg.get("nim_api_key"),
        nim_url=cfg.get("nim_url"),
        **{key: role_models},
    )


def reset_role_models(backend: str) -> None:
    """Clear all custom per-role overrides for the given backend."""
    cfg = load_config()
    key = f"role_models_{backend}"
    save_config(
        api_key=cfg.get("api_key", ""),
        base_url=cfg.get("base_url", BASE_URL),
        default_model=cfg.get("default_model", DEFAULT_MODEL),
        backend=cfg.get("backend", "einfra"),
        ollama_url=cfg.get("ollama_url", OLLAMA_BASE_URL),
        permission_mode=cfg.get("permission_mode", "autonomous"),
        nim_api_key=cfg.get("nim_api_key"),
        nim_url=cfg.get("nim_url"),
        **{key: {}},
    )


# ---------------------------------------------------------------------------
# Config load / save
# ---------------------------------------------------------------------------

def list_models(cfg: dict | None = None) -> list[str]:
    """
    Return available model names for the active backend.
    - einfra: query /v1/models live; fall back to KNOWN_MODELS on failure
    - ollama: poll the local server; fall back to KNOWN_MODELS
    - nim:    query /v1/models live; fall back to NIM_KNOWN_MODELS on failure
    """
    if cfg is None:
        cfg = {}
    if cfg.get("backend") == "ollama":
        pulled = ollama_list_models(cfg.get("ollama_url", OLLAMA_BASE_URL))
        return pulled if pulled else list(KNOWN_MODELS)
    if cfg.get("backend") == "nim":
        live = nim_list_models(
            cfg.get("nim_url", NIM_BASE_URL),
            cfg.get("nim_api_key", ""),
        )
        return live if live else list(NIM_KNOWN_MODELS)
    # einfra
    live = einfra_list_models(
        cfg.get("base_url", BASE_URL),
        cfg.get("api_key", ""),
    )
    return live if live else list(KNOWN_MODELS)


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
        "role_models_einfra": {},
        "role_models_nim": {},
        "role_models_ollama": {},
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
            # role_models_* are dicts — no env-var override, just load from file
            for rm_key in ("role_models_einfra", "role_models_nim", "role_models_ollama"):
                if isinstance(saved.get(rm_key), dict):
                    config[rm_key] = saved[rm_key]
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
    role_models_einfra: dict | None = None,
    role_models_nim: dict | None = None,
    role_models_ollama: dict | None = None,
):
    # Load existing config to preserve values not explicitly provided
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
        "role_models_einfra":  role_models_einfra  if role_models_einfra  is not None else _existing.get("role_models_einfra",  {}),
        "role_models_nim":     role_models_nim      if role_models_nim     is not None else _existing.get("role_models_nim",     {}),
        "role_models_ollama":  role_models_ollama   if role_models_ollama  is not None else _existing.get("role_models_ollama",  {}),
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(data, f, indent=2)
